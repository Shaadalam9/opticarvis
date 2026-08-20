"""Profile the host machine + Alpamayo-R1, plan an interleaved residency
configuration that fits the VRAM budget, and save a config.json.

Mirrors the inference pipeline used in the paper's RF research code
(`research/RF/quick_run.py`): three DoubleBufHook instances (VLM, ViT, Expert)
share a pair of GPU buffer slots and a single prefetch CUDA stream. ViT runs
once per inference (auto_restart=False) while VLM and Expert auto-restart
across decode steps and diffusion steps respectively.

Usage:
    python scripts/profile.py --output config.json
    python scripts/profile.py --vram-budget 12.0 --output config.json
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from pathlib import Path

# expandable segments before importing torch (matches RF)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402

# Make the package importable when run as a script.
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
# Drop the script's own dir so transformers' lazy loader doesn't trip on it.
sys.path[:] = [p for p in sys.path if p != str(_HERE)]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from alpamayo_memopt import DoubleBufHook  # noqa: E402
from alpamayo_memopt import config as cfg  # noqa: E402
from alpamayo_memopt import predictor  # noqa: E402
from alpamayo_memopt import setup as rf  # noqa: E402

try:
    import psutil
except ImportError:
    psutil = None


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Profile system + Alpamayo-R1 and emit a residency config."
    )
    p.add_argument("--output", "-o", type=Path, default=Path("config.json"),
                   help="Output config JSON path.")
    p.add_argument("--vram-budget", type=float, default=None,
                   help="VRAM budget in GB. Default: total VRAM - 1 GB.")
    p.add_argument("--baseline-time", type=float, default=14.52,
                   help="Reference baseline time (s) for speedup reporting.")
    p.add_argument("--margin", type=int, default=2,
                   help="Conservative resident-count margin (default: 2).")
    p.add_argument("--max-clock", type=int, default=None,
                   help="If set, lock GPU graphics clock (sudo).")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────
# System detection
# ─────────────────────────────────────────────────────────────────────

def _cpu_dram_total_gb() -> float:
    if psutil is not None:
        return psutil.virtual_memory().total / (1024 ** 3)
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) / (1024 ** 2)
    raise RuntimeError("Cannot determine CPU DRAM total")


def _gpu_info() -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device not available")
    props = torch.cuda.get_device_properties(0)
    return {"name": props.name, "vram_total_gb": props.total_memory / (1024 ** 3)}


# ─────────────────────────────────────────────────────────────────────
# RF-style measurement
# ─────────────────────────────────────────────────────────────────────

def _measure(vlm_resident, data_cache, label=""):
    """Run one inference with given VLM resident layers using 3-hook DFB.

    Mirrors `research/RF/quick_run.py:measure()` line-for-line, but using
    DoubleBufHook (renamed-only port of ProperDoubleBufHook).
    """
    torch.cuda.empty_cache()
    gc.collect()

    model = rf.load_model()
    vlm_layers, vblocks, expert_layers = rf.setup_gpu_essentials(model)

    # Move VLM resident layers to GPU
    for i in vlm_resident:
        vlm_layers[i].to("cuda")
    vlm_offload = sorted(set(range(rf.N_VLM)) - set(vlm_resident))

    # Three hooks, shared buffers + prefetch stream (RF pattern).
    # VLM hook is timing-instrumented during the timed iteration so we can
    # extract the per-call forward time (used as C_EXE in the predictor).
    vlm_hook = DoubleBufHook(auto_restart=True, enable_timing=True)
    vis_hook = DoubleBufHook(auto_restart=False)   # ViT runs once per inference
    exp_hook = DoubleBufHook(auto_restart=True)

    if vlm_offload:
        vlm_hook.pin(vlm_layers, vlm_offload)
    vis_hook.pin(vblocks, list(range(len(vblocks))))
    exp_hook.pin(expert_layers, list(range(len(expert_layers))))

    mx = max(
        vlm_hook.max_elements() if vlm_offload else 0,
        vis_hook.max_elements(),
        exp_hook.max_elements(),
    )
    shared_bufs = [torch.empty(mx, dtype=torch.bfloat16, device="cuda") for _ in range(2)]
    shared_ps = torch.cuda.Stream()
    vlm_hook.set_bufs(shared_bufs, prefetch_stream=shared_ps)
    vis_hook.set_bufs(shared_bufs, prefetch_stream=shared_ps)
    exp_hook.set_bufs(shared_bufs, prefetch_stream=shared_ps)

    if vlm_offload:
        vlm_hook.register(vlm_layers, vlm_offload)
    vis_hook.register(vblocks, list(range(len(vblocks))))
    exp_hook.register(expert_layers, list(range(len(expert_layers))))

    model_inputs = rf.prepare_inputs(model, data_cache)

    def run_once():
        vlm_hook.reset(); vis_hook.reset(); exp_hook.reset()
        vis_hook.start()  # ViT only; VLM/Expert auto-start lazily
        torch.manual_seed(42); torch.cuda.manual_seed_all(42)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model.sample_trajectories_from_data_with_vlm_rollout(
                data=rf.deep_copy_inputs(model_inputs),
                top_p=1.0, temperature=0.0,
                num_traj_samples=1, max_generation_length=22, return_extra=True,
            )
        torch.cuda.synchronize()

    # Warmup, then time one iteration (matches RF's quick_run.py)
    run_once()
    vlm_hook.clear_timings()  # discard warmup timings
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    run_once()
    elapsed = time.perf_counter() - t0
    peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

    # Extract per-VLM-layer forward time from the timed iteration.
    # Decode-phase calls (= per-token, after the prefill batch) are the
    # representative C_EXE for the linear residency-benefit model.
    vlm_call_times = vlm_hook.get_timings_ms()
    if vlm_call_times:
        # Group calls by layer index, then per-layer first call = prefill,
        # rest = decode. Average decode call times.
        from collections import defaultdict
        by_layer = defaultdict(list)
        for idx, t in vlm_call_times:
            by_layer[idx].append(t)
        decode_times = [t for calls in by_layer.values() for t in calls[1:]]
        vlm_layer_exe_ms = sum(decode_times) / len(decode_times) if decode_times else 0.0
    else:
        vlm_layer_exe_ms = 0.0

    # Measure per-layer parameter sizes before teardown
    def _layer_size_mb(ly):
        return sum(p.numel() * p.element_size() for p in ly.parameters()) / (1024 ** 2)

    vlm_layer_size_mb = _layer_size_mb(vlm_layers[0])
    vit_layer_size_mb = _layer_size_mb(vblocks[0]) if vblocks else 0.0
    expert_layer_size_mb = _layer_size_mb(expert_layers[0]) if expert_layers else 0.0
    n_vit = len(vblocks)
    n_expert = len(expert_layers)

    # Cleanup
    vlm_hook.remove(); vis_hook.remove(); exp_hook.remove()
    del model, vlm_layers, vblocks, expert_layers, model_inputs
    torch.cuda.empty_cache(); gc.collect()

    return {
        "label": label,
        "elapsed_s": elapsed,
        "peak_vram_gb": peak_gb,
        "vlm_layer_size_mb": vlm_layer_size_mb,
        "vit_layer_size_mb": vit_layer_size_mb,
        "expert_layer_size_mb": expert_layer_size_mb,
        "n_vit": n_vit,
        "n_expert": n_expert,
        "vlm_layer_exe_ms": vlm_layer_exe_ms,
        "num_resident": len(vlm_resident),
        "resident_indices": list(vlm_resident),
    }


# ─────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────

def main() -> int:
    args = parse_args()

    print("=" * 60)
    print("Alpamayo Memory Optimizer — Profiler")
    print("=" * 60)

    cpu_dram_gb = _cpu_dram_total_gb()
    gpu_info = _gpu_info()
    vram_total_gb = gpu_info["vram_total_gb"]
    vram_budget_gb = args.vram_budget if args.vram_budget else vram_total_gb - 1.0

    print(f"\n[1] System")
    print(f"    GPU            : {gpu_info['name']}")
    print(f"    VRAM total     : {vram_total_gb:.2f} GB")
    print(f"    VRAM budget    : {vram_budget_gb:.2f} GB")
    print(f"    CPU DRAM total : {cpu_dram_gb:.2f} GB")

    if vram_budget_gb > vram_total_gb:
        print(f"    [!] --vram-budget exceeds total VRAM", file=sys.stderr)
        return 2

    if args.max_clock:
        print(f"\n[1.5] Locking GPU clock to {args.max_clock} MHz...")
        rf.set_max_clock(args.max_clock)

    # ── Step A: Sequential DL once (Nr=0) to get full-offload baseline ──
    print("\n[2] Loading benchmark sample + running Sequential Demand Layering (Nr=0)...")
    data_cache = rf.load_data()
    base = _measure([], data_cache, label="Nr=0")
    full_offload_time_s = base["elapsed_s"]
    vlm_layer_size_mb = base["vlm_layer_size_mb"]
    vlm_layer_exe_ms = base["vlm_layer_exe_ms"]

    print(f"    Full-offload time      : {full_offload_time_s:.3f} s")
    print(f"    Peak VRAM              : {base['peak_vram_gb']:.2f} GB")
    print(f"    VLM layer count × size : {rf.N_VLM} × {vlm_layer_size_mb:.2f} MB "
          f"= {rf.N_VLM * vlm_layer_size_mb / 1024:.2f} GB")
    print(f"    ViT block count × size : {base['n_vit']} × "
          f"{base['vit_layer_size_mb']:.2f} MB = "
          f"{base['n_vit'] * base['vit_layer_size_mb'] / 1024:.2f} GB")
    print(f"    Expert layer × size    : {base['n_expert']} × "
          f"{base['expert_layer_size_mb']:.2f} MB = "
          f"{base['n_expert'] * base['expert_layer_size_mb'] / 1024:.2f} GB")
    print(f"    VLM decode forward time: {vlm_layer_exe_ms:.3f} ms (per call)")

    # ── Step B: Compute residency plan ──
    # Conservatively: leftover_VRAM = budget - peak_at_Nr0; max_resident = leftover / layer
    overhead_gb = base["peak_vram_gb"]  # already includes DFB buffers + non-VLM
    available_gb = max(0.0, vram_budget_gb - overhead_gb)
    max_possible = int(available_gb * 1024 / vlm_layer_size_mb)
    num_resident = max(0, max_possible - args.margin)
    indices = rf.interleaved_placement(num_resident, N=rf.N_VLM - 1)

    print("\n[3] Residency planning")
    print(f"    Peak VRAM at Nr=0   : {base['peak_vram_gb']:.2f} GB")
    print(f"    Max possible       : {max_possible}")
    print(f"    Conservative (-{args.margin})  : {num_resident}")
    print(f"    Resident indices   : {indices}")

    # ── Step C: Predicted inference time ──
    # Per-layer DMA estimate from full_offload_time and N_VLM:
    # Decode-dominant assumption: full_offload ≈ (DECODE_TOKENS+1) * N_VLM * C_DMA
    # so C_DMA ≈ full_offload / ((DECODE_TOKENS+1) * N_VLM)
    vlm_layer_dma_ms = (
        full_offload_time_s * 1000 / ((rf.DECODE_TOKENS + 1) * rf.N_VLM)
    )
    predicted_s = predictor.predict_inference_time(
        num_resident=num_resident,
        full_offload_time_s=full_offload_time_s,
        vlm_layer_dma_ms=vlm_layer_dma_ms,
        vlm_layer_exe_ms=vlm_layer_exe_ms,
        num_decode_tokens=rf.DECODE_TOKENS,
    )
    speedup = args.baseline_time / predicted_s if predicted_s > 0 else float("inf")

    print("\n[4] Predicted performance")
    print(f"    Predicted time     : {predicted_s:.3f} s")
    print(f"    Speedup vs {args.baseline_time:.2f}s : {speedup:.2f}×")

    # ── Step D: Save config ──
    config = cfg.Config(
        system=cfg.SystemConfig(
            gpu_name=gpu_info["name"],
            vram_total_gb=vram_total_gb,
            vram_budget_gb=vram_budget_gb,
            cpu_dram_total_gb=cpu_dram_gb,
        ),
        model=cfg.ModelConfig(
            name="Alpamayo-R1-10B",
            weights_total_gb=0.0,  # not measured separately in this pass
            vlm_layers=rf.N_VLM,
            vlm_layer_size_mb=vlm_layer_size_mb,
        ),
        profiling=cfg.ProfilingConfig(
            vlm_layer_dma_ms=vlm_layer_dma_ms,
            vlm_layer_exe_ms=vlm_layer_exe_ms,
            vit_layer_dma_ms=0.0,
            vit_layer_exe_ms=0.0,
            non_vlm_overhead_gb=base["peak_vram_gb"],
        ),
        residency=cfg.ResidencyConfig(
            max_possible=max_possible,
            num_resident=num_resident,
            resident_indices=indices,
        ),
        predicted_performance=cfg.PredictedPerformance(
            inference_time_s=predicted_s,
            vit_resident=False,
            expert_resident=False,
        ),
    )
    cfg.save_config(config, args.output)
    print(f"\n[5] Config saved to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
