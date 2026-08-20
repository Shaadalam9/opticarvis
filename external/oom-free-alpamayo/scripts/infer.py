"""Run Alpamayo-R1 inference using a saved residency config.

Mirrors `research/RF/quick_run.py`: three DoubleBufHook instances (VLM, ViT,
Expert) share GPU buffer slots and a prefetch CUDA stream.

Usage:
    python scripts/infer.py --config config.json
    python scripts/infer.py --config config.json --num-iterations 5
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path[:] = [p for p in sys.path if p != str(_HERE)]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from alpamayo_memopt import DoubleBufHook, load_config  # noqa: E402
from alpamayo_memopt import setup as rf  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run Alpamayo-R1 inference with a saved residency config."
    )
    p.add_argument("--config", "-c", type=Path, default=Path("config.json"),
                   help="Path to config.json from scripts/profile.py "
                        "(default: ./config.json)")
    p.add_argument("--num-iterations", "-n", type=int, default=1,
                   help="Number of timed inference iterations.")
    p.add_argument("--warmup", type=int, default=1,
                   help="Number of warmup iterations (untimed).")
    p.add_argument("--max-clock", type=int, default=None,
                   help="If set, lock GPU graphics clock (sudo).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    print("=" * 60)
    print("Alpamayo Memory Optimizer — Inference")
    print("=" * 60)

    config = load_config(args.config)
    print(f"\n[1] Config: {args.config}")
    print(f"    GPU expected      : {config.system.gpu_name}")
    print(f"    Resident layers   : {config.residency.num_resident} of "
          f"{config.model.vlm_layers}")
    print(f"    Predicted time    : "
          f"{config.predicted_performance.inference_time_s:.3f} s")

    if args.max_clock:
        rf.set_max_clock(args.max_clock)

    # Load model and benchmark sample
    print("\n[2] Loading model + benchmark sample...")
    torch.cuda.empty_cache(); gc.collect()
    model = rf.load_model()
    vlm_layers, vblocks, expert_layers = rf.setup_gpu_essentials(model)

    if len(vlm_layers) != config.model.vlm_layers:
        print(f"    [!] VLM layer count mismatch: model has {len(vlm_layers)}, "
              f"config expects {config.model.vlm_layers}", file=sys.stderr)
        return 2

    data_cache = rf.load_data()
    model_inputs = rf.prepare_inputs(model, data_cache)

    # Move resident VLM layers to GPU
    vlm_resident = list(config.residency.resident_indices)
    for i in vlm_resident:
        vlm_layers[i].to("cuda")
    vlm_offload = sorted(set(range(len(vlm_layers))) - set(vlm_resident))

    # Three hooks, shared buffers + prefetch stream
    print("\n[3] Installing DoubleBufHook (VLM + ViT + Expert)...")
    vlm_hook = DoubleBufHook(auto_restart=True)
    vis_hook = DoubleBufHook(auto_restart=False)
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

    def run_once():
        vlm_hook.reset(); vis_hook.reset(); exp_hook.reset()
        vis_hook.start()
        torch.manual_seed(42); torch.cuda.manual_seed_all(42)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            model.sample_trajectories_from_data_with_vlm_rollout(
                data=rf.deep_copy_inputs(model_inputs),
                top_p=1.0, temperature=1.0,
                num_traj_samples=1, max_generation_length=22, return_extra=True,
            )
        torch.cuda.synchronize()

    # Warmup
    print(f"\n[4] Running: {args.warmup} warmup + {args.num_iterations} timed...")
    for _ in range(args.warmup):
        run_once()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    times = []
    for i in range(args.num_iterations):
        t0 = time.perf_counter()
        run_once()
        t1 = time.perf_counter()
        times.append(t1 - t0)
        print(f"    iter {i + 1}: {times[-1]:.3f} s")

    if times:
        mean_s = sum(times) / len(times)
        std_s = (sum((t - mean_s) ** 2 for t in times) / len(times)) ** 0.5
        peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print(f"\n    Mean              : {mean_s:.3f} s (std {std_s:.3f}, n={len(times)})")
        print(f"    Peak VRAM         : {peak_gb:.2f} GB")
        print(f"    Predicted (config): "
              f"{config.predicted_performance.inference_time_s:.3f} s")
        print(f"    Δ vs predicted    : "
              f"{(mean_s - config.predicted_performance.inference_time_s) * 1000:+.1f} ms")

    # Cleanup
    vlm_hook.remove(); vis_hook.remove(); exp_hook.remove()
    return 0


if __name__ == "__main__":
    sys.exit(main())
