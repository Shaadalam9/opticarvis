"""Profiler: System spec detection + per-layer profiling + residency planning.

Provides building blocks invoked by scripts/profile.py:
    - detect_cpu_dram_gb()             system DRAM total
    - detect_gpu_info()                GPU name + VRAM total
    - get_model_weight_size_gb(model)  total model weight size
    - get_vlm_layer_size_mb(layers)    per-VLM-layer size (assumes homogeneous)
    - measure_h2d_bandwidth_gbps()     pinned-memory H2D PCIe BW
    - measure_layer_exe_ms(layer, ...) single-layer forward time
    - measure_non_vlm_vram_gb(model)   VRAM consumed after non-layer setup
    - compute_max_resident(...)        VRAM-budget-bound resident count
    - interleaved_placement(k, N)      paper Eq. 8 with first-in / last-out
    - run_sequential_dl_profile(...)   one-shot Nr=0 inference for full timing
"""

from __future__ import annotations

import time
from typing import List, Optional, Sequence

import torch

try:
    import psutil
except ImportError:  # graceful fallback
    psutil = None


# ─────────────────────────────────────────────────────────────────────
# System detection
# ─────────────────────────────────────────────────────────────────────

def detect_cpu_dram_gb() -> float:
    """Total system RAM in GB."""
    if psutil is None:
        # Fallback: parse /proc/meminfo
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return kb / (1024 ** 2)
        raise RuntimeError("Cannot determine CPU DRAM total")
    return psutil.virtual_memory().total / (1024 ** 3)


def detect_gpu_info(device_index: int = 0) -> dict:
    """GPU name, VRAM total (GB), compute capability."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device not available")
    props = torch.cuda.get_device_properties(device_index)
    return {
        "name": props.name,
        "vram_total_gb": props.total_memory / (1024 ** 3),
        "compute_capability": f"{props.major}.{props.minor}",
        "device_index": device_index,
    }


# ─────────────────────────────────────────────────────────────────────
# Model size detection
# ─────────────────────────────────────────────────────────────────────

def get_model_weight_size_gb(model: torch.nn.Module) -> float:
    """Total parameter + buffer size of `model` in GB."""
    p_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    b_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    return (p_bytes + b_bytes) / (1024 ** 3)


def get_vlm_layer_size_mb(vlm_layers: Sequence[torch.nn.Module]) -> float:
    """Size of one VLM layer in MB (assumes homogeneous transformer layers)."""
    if not vlm_layers:
        raise ValueError("Empty VLM layer list")
    layer = vlm_layers[0]
    p_bytes = sum(p.numel() * p.element_size() for p in layer.parameters())
    b_bytes = sum(b.numel() * b.element_size() for b in layer.buffers())
    return (p_bytes + b_bytes) / (1024 ** 2)


def verify_cpu_can_hold_weights(model: torch.nn.Module,
                                cpu_dram_gb: Optional[float] = None,
                                safety_margin_gb: float = 1.0) -> None:
    """Raise RuntimeError if CPU DRAM is too small for model weights."""
    if cpu_dram_gb is None:
        cpu_dram_gb = detect_cpu_dram_gb()
    weights_gb = get_model_weight_size_gb(model)
    if cpu_dram_gb < weights_gb + safety_margin_gb:
        raise RuntimeError(
            f"Insufficient CPU DRAM: model weights {weights_gb:.2f} GB "
            f"+ {safety_margin_gb:.1f} GB safety > available {cpu_dram_gb:.2f} GB. "
            "This framework requires CPU memory to hold all model parameters."
        )


# ─────────────────────────────────────────────────────────────────────
# Bandwidth / latency measurement
# ─────────────────────────────────────────────────────────────────────

def measure_h2d_bandwidth_gbps(
    size_mb: int = 256,
    num_trials: int = 5,
    warmup: int = 2,
) -> float:
    """Measure pinned-memory H2D PCIe bandwidth in GB/s."""
    elements = size_mb * 1024 * 1024 // 2  # bfloat16 = 2 bytes
    cpu_pinned = torch.empty(elements, dtype=torch.bfloat16).pin_memory()
    cpu_pinned.uniform_(-1, 1)
    gpu_buf = torch.empty(elements, dtype=torch.bfloat16, device="cuda")

    # Warmup
    for _ in range(warmup):
        gpu_buf.copy_(cpu_pinned, non_blocking=True)
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(num_trials)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(num_trials)]
    for i in range(num_trials):
        starts[i].record()
        gpu_buf.copy_(cpu_pinned, non_blocking=True)
        ends[i].record()
    torch.cuda.synchronize()

    elapsed_ms = [starts[i].elapsed_time(ends[i]) for i in range(num_trials)]
    avg_ms = sum(elapsed_ms) / num_trials
    return (size_mb / 1024) / (avg_ms / 1000)


def measure_layer_exe_ms(
    layer: torch.nn.Module,
    forward_fn,
    num_trials: int = 5,
    warmup: int = 3,
) -> float:
    """Measure single-layer forward execution time in ms.

    Args:
        layer:        nn.Module already moved to GPU.
        forward_fn:   Callable that invokes the layer once with appropriate inputs.
                      Signature: () -> output.
        num_trials:   Timed iterations.
        warmup:       Untimed warmup iterations.
    """
    for _ in range(warmup):
        forward_fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(num_trials)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(num_trials)]
    for i in range(num_trials):
        starts[i].record()
        forward_fn()
        ends[i].record()
    torch.cuda.synchronize()

    return sum(starts[i].elapsed_time(ends[i]) for i in range(num_trials)) / num_trials


def measure_current_vram_gb(device_index: int = 0) -> float:
    """Currently allocated VRAM in GB (not just peak; reported by torch)."""
    return torch.cuda.memory_allocated(device_index) / (1024 ** 3)


# ─────────────────────────────────────────────────────────────────────
# Sequential DL one-shot profiling
# ─────────────────────────────────────────────────────────────────────

def run_sequential_dl_profile(inference_fn) -> dict:
    """Execute `inference_fn` once with timing instrumentation.

    `inference_fn` must perform a full inference pass with all VLM layers
    offloaded (Nr=0) using DoubleBufHook. Returns wall-clock total + peak VRAM.
    """
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    inference_fn()
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    return {
        "full_offload_time_s": t1 - t0,
        "peak_vram_gb": torch.cuda.max_memory_allocated() / (1024 ** 3),
    }


# ─────────────────────────────────────────────────────────────────────
# Residency planning (paper Eq. 8 + conservative margin)
# ─────────────────────────────────────────────────────────────────────

def compute_max_resident(
    vram_budget_gb: float,
    vlm_layer_size_mb: float,
    non_vlm_overhead_gb: float,
    dfb_buffer_overhead_mb: Optional[float] = None,
) -> int:
    """Maximum number of VLM layers that fit in the VRAM budget.

    Args:
        vram_budget_gb:        User-specified VRAM budget (≤ total VRAM).
        vlm_layer_size_mb:     Per-VLM-layer size (homogeneous assumption).
        non_vlm_overhead_gb:   VRAM used by non-VLM components (ViT/Expert
                               essentials, KV cache, activations, etc.).
        dfb_buffer_overhead_mb: DFB ping-pong buffer (≈ 2 × layer_size). If
                               None, assumes 2 × vlm_layer_size_mb.

    Returns:
        Largest k such that k * layer + non_vlm + dfb ≤ budget. ≥ 0.
    """
    if dfb_buffer_overhead_mb is None:
        dfb_buffer_overhead_mb = 2 * vlm_layer_size_mb
    overhead_gb = non_vlm_overhead_gb + dfb_buffer_overhead_mb / 1024
    available_gb = vram_budget_gb - overhead_gb
    if available_gb <= 0:
        return 0
    return int(available_gb * 1024 / vlm_layer_size_mb)


def interleaved_placement(k: int, total_layers: int = 36) -> List[int]:
    """Paper Eq. 8 — interleaved residency placement.

    Rules (option A):
        - First layer (idx 0) always resident
        - Last layer (idx total_layers-1) always excluded
        - The remaining (k-1) layers placed at evenly spaced indices within
          [0, total_layers-1)

    Args:
        k:             Total resident layer count (≥ 1).
        total_layers:  Number of VLM layers (e.g., 36 for Alpamayo-R1).

    Returns:
        Sorted list of resident layer indices, length = k (or fewer if k > N).
    """
    if k <= 0:
        return []
    N = total_layers - 1  # candidate range [0, N) excluding last layer
    if k >= N:
        return list(range(N))
    return sorted(set(int(i * N / k) for i in range(k)))


def apply_conservative_margin(max_possible: int, margin: int = 2,
                              minimum: int = 1) -> int:
    """Apply safety margin: num_resident = max(minimum, max_possible - margin)."""
    return max(minimum, max_possible - margin)
