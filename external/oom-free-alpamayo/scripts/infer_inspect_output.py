"""Run Alpamayo-R1 inference using a saved residency config and save output.

Usage:
    python scripts/infer_inspect_output.py --config config_5080_16gb.json
"""

from __future__ import annotations

import argparse
import gc
import json
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
from alpamayo_policy_io import add_object_summary, object_to_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Alpamayo-R1 inference and inspect returned output."
    )
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=Path("config.json"),
        help="Path to config file from scripts/profile.py.",
    )
    parser.add_argument(
        "--num-iterations",
        "-n",
        type=int,
        default=1,
        help="Number of timed inference iterations.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Number of warmup iterations.",
    )
    parser.add_argument(
        "--max-clock",
        type=int,
        default=None,
        help="If set, lock GPU graphics clock.",
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=Path("alpamayo_output_summary.txt"),
        help="Readable summary output file.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("alpamayo_inference_output.json"),
        help="JSON output file.",
    )
    return parser.parse_args()


def write_outputs(last_result, summary_path: Path, json_path: Path) -> None:
    summary_lines = []
    summary_lines.append("Alpamayo output summary")
    summary_lines.append("=======================")
    summary_lines.append("")

    add_object_summary(summary_lines, last_result, "result")

    for line in summary_lines:
        print(line)

    with open(summary_path, "w", encoding="utf-8") as output_file:
        output_file.write("\n".join(summary_lines))

    json_payload = {
        "result": object_to_json(last_result),
    }

    if isinstance(last_result, tuple):
        if len(last_result) >= 1:
            json_payload["pred_xyz"] = object_to_json(last_result[0])
        if len(last_result) >= 2:
            json_payload["pred_rot"] = object_to_json(last_result[1])
        if len(last_result) >= 3:
            json_payload["extra"] = object_to_json(last_result[2])

    with open(json_path, "w", encoding="utf-8") as output_file:
        json.dump(json_payload, output_file, indent=2)


def main() -> int:
    args = parse_args()

    print("=" * 60)
    print("Alpamayo Memory Optimizer - Inference")
    print("=" * 60)

    config = load_config(args.config)

    print(f"\n[1] Config: {args.config}")
    print(f"    GPU expected      : {config.system.gpu_name}")
    print(
        f"    Resident layers   : {config.residency.num_resident} of "
        f"{config.model.vlm_layers}"
    )
    print(
        f"    Predicted time    : "
        f"{config.predicted_performance.inference_time_s:.3f} s"
    )

    if args.max_clock:
        rf.set_max_clock(args.max_clock)

    print("\n[2] Loading model + benchmark sample...")
    torch.cuda.empty_cache()
    gc.collect()

    model = rf.load_model()
    vlm_layers, vblocks, expert_layers = rf.setup_gpu_essentials(model)

    if len(vlm_layers) != config.model.vlm_layers:
        print(
            f"    [!] VLM layer count mismatch: model has {len(vlm_layers)}, "
            f"config expects {config.model.vlm_layers}",
            file=sys.stderr,
        )
        return 2

    data_cache = rf.load_data()
    model_inputs = rf.prepare_inputs(model, data_cache)

    vlm_resident = list(config.residency.resident_indices)
    for index in vlm_resident:
        vlm_layers[index].to("cuda")

    vlm_offload = sorted(set(range(len(vlm_layers))) - set(vlm_resident))

    print("\n[3] Installing DoubleBufHook (VLM + ViT + Expert)...")

    vlm_hook = DoubleBufHook(auto_restart=True)
    vis_hook = DoubleBufHook(auto_restart=False)
    exp_hook = DoubleBufHook(auto_restart=True)

    if vlm_offload:
        vlm_hook.pin(vlm_layers, vlm_offload)

    vis_hook.pin(vblocks, list(range(len(vblocks))))
    exp_hook.pin(expert_layers, list(range(len(expert_layers))))

    max_elements = max(
        vlm_hook.max_elements() if vlm_offload else 0,
        vis_hook.max_elements(),
        exp_hook.max_elements(),
    )

    shared_bufs = [
        torch.empty(max_elements, dtype=torch.bfloat16, device="cuda")
        for _ in range(2)
    ]
    shared_stream = torch.cuda.Stream()

    vlm_hook.set_bufs(shared_bufs, prefetch_stream=shared_stream)
    vis_hook.set_bufs(shared_bufs, prefetch_stream=shared_stream)
    exp_hook.set_bufs(shared_bufs, prefetch_stream=shared_stream)

    if vlm_offload:
        vlm_hook.register(vlm_layers, vlm_offload)

    vis_hook.register(vblocks, list(range(len(vblocks))))
    exp_hook.register(expert_layers, list(range(len(expert_layers))))

    def run_once():
        vlm_hook.reset()
        vis_hook.reset()
        exp_hook.reset()

        vis_hook.start()

        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            result = model.sample_trajectories_from_data_with_vlm_rollout(
                data=rf.deep_copy_inputs(model_inputs),
                top_p=1.0,
                temperature=0.0,
                num_traj_samples=1,
                max_generation_length=22,
                return_extra=True,
            )

        torch.cuda.synchronize()
        return result

    print(f"\n[4] Running: {args.warmup} warmup + {args.num_iterations} timed...")

    for _ in range(args.warmup):
        run_once()

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    times = []
    last_result = None

    for iteration in range(args.num_iterations):
        start_time = time.perf_counter()
        last_result = run_once()
        end_time = time.perf_counter()

        times.append(end_time - start_time)
        print(f"    iter {iteration + 1}: {times[-1]:.3f} s")

    if times:
        mean_s = sum(times) / len(times)
        std_s = (sum((duration - mean_s) ** 2 for duration in times) / len(times)) ** 0.5
        peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

        print(f"\n    Mean              : {mean_s:.3f} s (std {std_s:.3f}, n={len(times)})")
        print(f"    Peak VRAM         : {peak_gb:.2f} GB")
        print(
            f"    Predicted (config): "
            f"{config.predicted_performance.inference_time_s:.3f} s"
        )
        print(
            f"    Delta vs predicted: "
            f"{(mean_s - config.predicted_performance.inference_time_s) * 1000:+.1f} ms"
        )

    print("\n[5] Saving output")
    write_outputs(last_result, args.output_summary, args.output_json)

    print(f"\n    Summary saved to  : {args.output_summary}")
    print(f"    JSON saved to     : {args.output_json}")

    vlm_hook.remove()
    vis_hook.remove()
    exp_hook.remove()

    return 0


if __name__ == "__main__":
    sys.exit(main())