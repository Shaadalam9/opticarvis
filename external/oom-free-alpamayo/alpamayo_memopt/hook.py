"""DoubleBufHook: Pipelined Demand Layering with Double Flat Buffer (DFB).

Implements asynchronous Double Flat Buffer prefetching for transformer layer
parameters. Two contiguous GPU buffer slots alternate per layer, enabling
overlap between H2D parameter transfer and previous-layer execution.

Synchronization:
    - WAR (Write-After-Read): prefetch_stream waits for compute_done[slot]
                              before initiating DMA into that slot.
    - RAW (Read-After-Write): default_stream waits for dma_done[idx] before
                              executing the layer.

Architecture (paper Section IV):
    - allocate() / set_bufs(): create or share two GPU buffer slots.
    - pin():     move offloaded layer parameters to pinned host memory and
                 compute flat-buffer layout (offset/shape/numel).
    - register(): attach forward pre/post hooks to each offloaded layer.
    - start():   submit DMA for the first two offloaded layers.
                 Subsequent DMAs are chained from the post-hook of layer pos
                 to layer pos+2 (same slot reuse).
    - reset():   clear DMA state between inference iterations.
    - remove():  unregister hooks.

Bit-exact equivalence with the un-hooked baseline has been verified.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch


class DoubleBufHook:
    """Asynchronous Double Flat Buffer hook for layer-wise parameter prefetch.

    Args:
        auto_restart: When True, the post-hook of the last offloaded layer
                      automatically resubmits DMAs for the next iteration.
                      Set True for repeated modules (VLM Decode, Diffusion
                      Expert) and False for single-shot modules (ViT).
    """

    def __init__(self, auto_restart: bool = True,
                 enable_timing: bool = False) -> None:
        # CPU-side pinned storage
        self.cpu_params: dict = {}     # idx -> {name: pinned_tensor}
        self.cpu_buffers: dict = {}    # idx -> {name: pinned_tensor}

        # Flat-buffer layout: offset/shape/numel within a slot
        self.param_layout: dict = {}   # idx -> {name: (offset, shape, numel)}
        self.buf_layout: dict = {}     # idx -> {name: (offset, shape, numel)}

        # Two GPU buffer slots (ping-pong)
        self.gpu_bufs: List[Optional[torch.Tensor]] = [None, None]

        # Offload bookkeeping
        self.offload_indices: List[int] = []
        self._pos: dict = {}           # idx -> position in offload_indices

        # CUDA streams and events
        self.prefetch_stream: torch.cuda.Stream = torch.cuda.Stream()
        self.dma_done: dict = {}                       # idx -> event
        self.compute_done: List[torch.cuda.Event] = [
            torch.cuda.Event(), torch.cuda.Event()
        ]

        # Registered PyTorch hooks (for cleanup)
        self.hooks: List = []

        self.auto_restart: bool = auto_restart

        # Optional per-call forward-time instrumentation
        self.enable_timing: bool = enable_timing
        self._pending_starts: List = []          # stack of (idx, start_event)
        self.timings: List = []                  # [(idx, elapsed_ms), ...]
                                                  #   filled lazily; call get_timings_ms()

    # -------------------------------------------------------------------
    # Buffer allocation
    # -------------------------------------------------------------------

    def allocate(self, max_elements: int) -> None:
        """Allocate two independent GPU buffers of `max_elements` BF16 elements."""
        for s in range(2):
            self.gpu_bufs[s] = torch.empty(
                max_elements, dtype=torch.bfloat16, device="cuda"
            )
        for ev in self.compute_done:
            ev.record()

    def set_bufs(
        self,
        bufs: Sequence[torch.Tensor],
        compute_done: Optional[Sequence[torch.cuda.Event]] = None,
        prefetch_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        """Share GPU buffers (and optionally events/stream) across modules.

        Sharing prefetch_stream serializes all DMAs across modules, eliminating
        cross-module race conditions when modules execute sequentially.
        """
        self.gpu_bufs = list(bufs)
        if compute_done is not None:
            self.compute_done = list(compute_done)
        if prefetch_stream is not None:
            self.prefetch_stream = prefetch_stream
        for ev in self.compute_done:
            ev.record()

    # -------------------------------------------------------------------
    # Layer pinning
    # -------------------------------------------------------------------

    def pin(self, layers: Sequence[torch.nn.Module], indices: Sequence[int]) -> None:
        """Move specified layers to CPU pinned memory and compute flat layout.

        Args:
            layers:  List/sequence of nn.Module (e.g., model.vlm.layers).
            indices: Indices of layers to be offloaded (non-resident).
        """
        for i in indices:
            ly = layers[i]
            if next(ly.parameters()).device.type == "cuda":
                ly.to("cpu")

            # Pin parameters: replace original storage with empty, hold pinned tensor
            self.cpu_params[i] = {}
            for n, p in ly.named_parameters():
                d = p.data.contiguous()
                p.data = torch.empty(0)
                pinned = d.pin_memory() if not d.is_pinned() else d
                del d
                self.cpu_params[i][n] = pinned
                p.data = pinned

            # Pin non-parameter buffers
            self.cpu_buffers[i] = {}
            for n, b in ly.named_buffers():
                if b.device.type == "cpu":
                    d = b.data.contiguous()
                    b.data = torch.empty(0)
                    pinned = d.pin_memory() if not d.is_pinned() else d
                    del d
                    self.cpu_buffers[i][n] = pinned
                    b.data = pinned

        self.offload_indices = sorted(self.cpu_params.keys())
        self._pos = {idx: pos for pos, idx in enumerate(self.offload_indices)}

        # Compute flat-buffer layout per layer
        for i in self.offload_indices:
            offset = 0
            self.param_layout[i] = {}
            for n, p in self.cpu_params[i].items():
                nu = p.numel()
                self.param_layout[i][n] = (offset, p.shape, nu)
                offset += nu
            self.buf_layout[i] = {}
            if i in self.cpu_buffers:
                for n, b in self.cpu_buffers[i].items():
                    nu = b.numel()
                    self.buf_layout[i][n] = (offset, b.shape, nu)
                    offset += nu

    def max_elements(self) -> int:
        """Element count of the largest offloaded layer (for buffer sizing)."""
        mx = 0
        for i in self.offload_indices:
            total = sum(p.numel() for p in self.cpu_params[i].values())
            if i in self.cpu_buffers:
                total += sum(b.numel() for b in self.cpu_buffers[i].values())
            mx = max(mx, total)
        return mx

    # -------------------------------------------------------------------
    # DMA submission
    # -------------------------------------------------------------------

    def _slot(self, idx: int) -> int:
        """Slot number (0 or 1) for layer `idx` (ping-pong)."""
        return self._pos[idx] % 2

    def _submit_dma(self, idx: int) -> None:
        """Enqueue an H2D DMA for layer `idx` on the prefetch_stream."""
        if idx not in self.cpu_params or idx in self.dma_done:
            return
        s = self._slot(idx)
        buf = self.gpu_bufs[s]

        with torch.cuda.stream(self.prefetch_stream):
            # WAR: wait for previous compute on this slot to finish
            self.prefetch_stream.wait_event(self.compute_done[s])

            for n, pinned in self.cpu_params[idx].items():
                o, _, nu = self.param_layout[idx][n]
                buf[o:o + nu].copy_(pinned.view(-1), non_blocking=True)

            if idx in self.cpu_buffers and idx in self.buf_layout:
                for n, pinned in self.cpu_buffers[idx].items():
                    if n in self.buf_layout[idx]:
                        o, _, nu = self.buf_layout[idx][n]
                        buf[o:o + nu].copy_(pinned.view(-1), non_blocking=True)

            ev = torch.cuda.Event()
            ev.record(self.prefetch_stream)
        self.dma_done[idx] = ev

    def start(self) -> None:
        """Submit DMAs for the first two offloaded layers (DFB priming)."""
        self.dma_done.clear()
        if len(self.offload_indices) >= 1:
            self._submit_dma(self.offload_indices[0])
        if len(self.offload_indices) >= 2:
            self._submit_dma(self.offload_indices[1])

    # -------------------------------------------------------------------
    # PyTorch hook registration
    # -------------------------------------------------------------------

    def register(self, layers: Sequence[torch.nn.Module], indices: Sequence[int]) -> None:
        """Attach forward pre/post hooks to each offloaded layer."""
        for i in indices:
            if i not in self.cpu_params:
                continue
            ly = layers[i]

            def make_pre(idx):
                def hook_fn(module, inp):
                    s = self._slot(idx)
                    buf = self.gpu_bufs[s]

                    # RAW: wait until DMA is done for this layer
                    if idx in self.dma_done:
                        torch.cuda.current_stream().wait_event(self.dma_done[idx])
                        del self.dma_done[idx]
                    else:
                        # Fallback: synchronous copy if DMA not yet submitted
                        self.prefetch_stream.synchronize()
                        for n, pinned in self.cpu_params[idx].items():
                            o, _, nu = self.param_layout[idx][n]
                            buf[o:o + nu].copy_(pinned.view(-1))
                        if idx in self.cpu_buffers and idx in self.buf_layout:
                            for n, pinned in self.cpu_buffers[idx].items():
                                if n in self.buf_layout[idx]:
                                    o, _, nu = self.buf_layout[idx][n]
                                    buf[o:o + nu].copy_(pinned.view(-1))

                    # Re-bind module parameter pointers to the GPU buffer slot
                    pd = dict(module.named_parameters())
                    for n in self.cpu_params[idx]:
                        if n in pd:
                            o, sh, nu = self.param_layout[idx][n]
                            pd[n].data = buf[o:o + nu].view(sh)
                    if idx in self.buf_layout:
                        bd = dict(module.named_buffers())
                        for n in self.cpu_buffers.get(idx, {}):
                            if n in bd and n in self.buf_layout[idx]:
                                o, sh, nu = self.buf_layout[idx][n]
                                bd[n].data = buf[o:o + nu].view(sh)

                    # Optional timing: record forward-start event
                    if self.enable_timing:
                        start_ev = torch.cuda.Event(enable_timing=True)
                        start_ev.record()
                        self._pending_starts.append((idx, start_ev))
                return hook_fn

            def make_post(idx):
                def hook_fn(module, inp, out):
                    s = self._slot(idx)
                    # Mark slot's compute as done (for next prefetch WAR check)
                    self.compute_done[s].record()

                    # Optional timing: pair with the matching pre-hook start event
                    if self.enable_timing and self._pending_starts:
                        p_idx, start_ev = self._pending_starts.pop()
                        end_ev = torch.cuda.Event(enable_timing=True)
                        end_ev.record()
                        self.timings.append((p_idx, start_ev, end_ev))

                    # Restore CPU pinned pointer references
                    pd = dict(module.named_parameters())
                    for n, pinned in self.cpu_params[idx].items():
                        if n in pd and pd[n].device.type == "cuda":
                            pd[n].data = pinned
                    if idx in self.cpu_buffers:
                        bd = dict(module.named_buffers())
                        for n, pinned in self.cpu_buffers[idx].items():
                            if n in bd and bd[n].device.type == "cuda":
                                bd[n].data = pinned

                    # Chain: submit DMA for layer at pos+2 (same slot reuse)
                    pos = self._pos[idx]
                    if pos + 2 < len(self.offload_indices):
                        self._submit_dma(self.offload_indices[pos + 2])

                    # Last offloaded layer of a repeated module: restart DMA
                    if self.auto_restart and pos == len(self.offload_indices) - 1:
                        self.start()
                return hook_fn

            self.hooks.append(ly.register_forward_pre_hook(make_pre(i)))
            self.hooks.append(ly.register_forward_hook(make_post(i)))

    # -------------------------------------------------------------------
    # State management
    # -------------------------------------------------------------------

    def reset(self) -> None:
        """Clear DMA state between inference iterations."""
        self.dma_done.clear()
        for ev in self.compute_done:
            ev.record()

    def remove(self) -> None:
        """Remove all registered forward hooks."""
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def get_timings_ms(self) -> List:
        """Return per-call forward times collected when enable_timing=True.

        Each element is (layer_idx, elapsed_ms). Synchronizes the device once.
        """
        if not self.enable_timing:
            return []
        torch.cuda.synchronize()
        return [(idx, start.elapsed_time(end)) for idx, start, end in self.timings]

    def clear_timings(self) -> None:
        """Discard collected timings without disabling instrumentation."""
        self.timings.clear()
        self._pending_starts.clear()
