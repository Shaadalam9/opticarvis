"""Run OOM-free Alpamayo-R1 inference on one or more CROWD video clips.

Single-clip mode (default): edit the CONFIG constants below and run
    python scripts/infer_crowd_clip.py

Batch mode: pass a manifest CSV so the multi-minute model load is paid once
for the whole set instead of once per clip:
    python scripts/infer_crowd_clip.py --clips clips.csv

Manifest columns: video_id, segment_start_time_s, clip_video
optional:         when_start_local_s, when_end_local_s

Per clip, three outputs are written to the output dir, named after the clip's
video_id: alpamayo_output_summary_<id>.txt, alpamayo_inference_output_<id>.json
and alpamayo_offline_when_<id>.csv (the offline WHEN CSV consumed by
opticarvis/policy_demo.py).
"""

from __future__ import annotations

import argparse
import csv
import datetime
import gc
import json
import os
import sys
import time

import cv2
import numpy as np
import torch

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path[:] = [p for p in sys.path if p != HERE]
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from alpamayo_memopt import DoubleBufHook, load_config  # noqa: E402
from alpamayo_memopt import setup as rf  # noqa: E402
from alpamayo_policy_io import (  # noqa: E402
    add_object_summary,
    build_policy_row,
    object_to_json,
    write_policy_csv,
)


# =============================================================================
# CONFIG — defaults for single-clip mode (overridden by --clips manifest)
# =============================================================================

CONFIG_FILE = "config_5080_16gb.json"

VIDEO_ID = "TuCsyBF3nHU"
SEGMENT_START_TIME_S = 4630.0
CLIP_VIDEO = "C:/Users/localadmin/Desktop/Shadab/alpamayo_outputs/crowd_clips/TuCsyBF3nHU_4630_30s.mp4"

OUTPUT_DIR = "C:/Users/localadmin/Desktop/Shadab/alpamayo_outputs"

IMAGE_HEIGHT = 1080
IMAGE_WIDTH = 1920
NUM_TIME_STEPS = 4
NUM_CAMERA_SLOTS = 4

# WHEN window written into the offline policy CSV, in seconds relative to the
# start of the clip video (which should match the candidate clip selected by
# opticarvis/policy_demo.py).
WHEN_START_LOCAL_S = 12.67
WHEN_END_LOCAL_S = 15.60

MODEL_SOURCE = "oom_free_alpamayo_r1_crowd_front_view_proxy"

SAMPLING = {
    "top_p": 1.0,
    "temperature": 1.0,
    "num_traj_samples": 1,
    "max_generation_length": 22,
    "seed": 42,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OOM-free Alpamayo-R1 inference on CROWD clips (single clip or manifest batch)."
    )
    parser.add_argument(
        "--clips",
        default=None,
        help="Manifest CSV (video_id, segment_start_time_s, clip_video[, when_start_local_s, when_end_local_s]). "
             "Omit to run the single clip from the CONFIG constants.",
    )
    parser.add_argument("--config", "-c", default=CONFIG_FILE, help="Residency config from scripts/profile.py.")
    parser.add_argument("--output-dir", "-o", default=OUTPUT_DIR, help="Directory for the per-clip outputs.")
    return parser.parse_args()


def resolve_config_path(config_arg: str) -> str:
    config_path = config_arg if os.path.isabs(config_arg) else os.path.join(REPO, config_arg)
    if not os.path.isfile(config_path):
        # Also honour a path relative to the current working directory.
        if os.path.isfile(config_arg):
            return os.path.abspath(config_arg)
        print(
            f"Config not found: {config_path}\n"
            "Run scripts/profile.py to generate one for this machine.",
            file=sys.stderr,
        )
        sys.exit(1)
    return config_path


def load_clip_list(args: argparse.Namespace) -> list:
    if args.clips is None:
        return [
            {
                "video_id": VIDEO_ID,
                "segment_start_time_s": float(SEGMENT_START_TIME_S),
                "clip_video": CLIP_VIDEO,
                "when_start_local_s": float(WHEN_START_LOCAL_S),
                "when_end_local_s": float(WHEN_END_LOCAL_S),
            }
        ]

    if not os.path.isfile(args.clips):
        print("Clip manifest not found:", args.clips, file=sys.stderr)
        sys.exit(1)

    clips = []
    with open(args.clips, "r", encoding="utf-8", newline="") as manifest:
        for line_number, row in enumerate(csv.DictReader(manifest), start=2):
            try:
                clips.append(
                    {
                        "video_id": str(row["video_id"]).strip(),
                        "segment_start_time_s": float(row["segment_start_time_s"]),
                        "clip_video": str(row["clip_video"]).strip(),
                        "when_start_local_s": float(row.get("when_start_local_s") or WHEN_START_LOCAL_S),
                        "when_end_local_s": float(row.get("when_end_local_s") or WHEN_END_LOCAL_S),
                    }
                )
            except (KeyError, ValueError) as exc:
                print(f"Skipping manifest line {line_number}: {exc}", file=sys.stderr)
    if not clips:
        print("Clip manifest has no usable rows:", args.clips, file=sys.stderr)
        sys.exit(1)
    return clips


def clip_output_paths(output_dir: str, video_id: str) -> dict:
    return {
        "summary": os.path.join(output_dir, f"alpamayo_output_summary_{video_id}.txt"),
        "json": os.path.join(output_dir, f"alpamayo_inference_output_{video_id}.json"),
        "csv": os.path.join(output_dir, f"alpamayo_offline_when_{video_id}.csv"),
    }


def read_clip_frames(video_path: str) -> torch.Tensor:
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Clip video not found: {video_path}")

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            raise IOError(f"Video has no readable frames: {video_path}")

        sample_positions = np.linspace(0, frame_count - 1, NUM_TIME_STEPS).astype(int)

        frames = []
        for frame_index in sample_positions:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame_bgr = capture.read()
            if not ok:
                raise IOError(f"Could not read frame {int(frame_index)} of {video_path}")

            frame_bgr = cv2.resize(frame_bgr, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_AREA)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_chw = np.transpose(frame_rgb, (2, 0, 1))
            frames.append(torch.from_numpy(frame_chw).to(torch.uint8))
    finally:
        capture.release()

    time_frames = torch.stack(frames, dim=0)
    return torch.stack([time_frames] * NUM_CAMERA_SLOTS, dim=0)


def replace_images_in_messages(messages, image_frames: torch.Tensor) -> None:
    flat_images = image_frames.reshape(
        NUM_CAMERA_SLOTS * NUM_TIME_STEPS,
        3,
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
    )

    image_index = 0
    for message in messages:
        for item in message.get("content", []):
            if isinstance(item, dict) and item.get("type") == "image":
                if image_index < flat_images.shape[0]:
                    item["image"] = flat_images[image_index]
                    image_index += 1

    print("Replaced image items in messages:", image_index)


def prepare_crowd_data_cache(data_cache: dict, clip: dict) -> None:
    """Inject the clip's frames into the benchmark data cache.

    Only data_cache["messages"] matters: rf.prepare_inputs reads messages plus
    the benchmark sample's ego_history_* tensors, so the ego history fed to the
    model intentionally remains the NVIDIA benchmark sample's (front-view proxy
    setup for CROWD clips).
    """
    image_frames = read_clip_frames(clip["clip_video"])
    replace_images_in_messages(data_cache["messages"], image_frames)


def write_outputs(last_result, clip: dict, paths: dict, run_meta: dict) -> None:
    summary_lines = []
    summary_lines.append("Alpamayo CROWD output summary")
    summary_lines.append("=============================")
    summary_lines.append("")
    summary_lines.append("video_id: " + clip["video_id"])
    summary_lines.append("segment_start_time_s: " + str(clip["segment_start_time_s"]))
    summary_lines.append("clip_video: " + clip["clip_video"])
    summary_lines.append("")

    add_object_summary(summary_lines, last_result, "result")

    with open(paths["summary"], "w", encoding="utf-8") as output_file:
        output_file.write("\n".join(summary_lines))

    for line in summary_lines:
        print(line)

    json_payload = {
        "video_id": clip["video_id"],
        "segment_start_time_s": clip["segment_start_time_s"],
        "clip_video": clip["clip_video"],
        "run_meta": run_meta,
        "result": object_to_json(last_result),
    }

    if isinstance(last_result, tuple):
        if len(last_result) >= 1:
            json_payload["pred_xyz"] = object_to_json(last_result[0])
        if len(last_result) >= 2:
            json_payload["pred_rot"] = object_to_json(last_result[1])
        if len(last_result) >= 3:
            json_payload["extra"] = object_to_json(last_result[2])

    with open(paths["json"], "w", encoding="utf-8") as output_file:
        json.dump(json_payload, output_file, indent=2)


def write_clip_policy_csv(clip: dict, paths: dict) -> dict:
    with open(paths["json"], "r", encoding="utf-8") as input_file:
        payload = json.load(input_file)

    row = build_policy_row(
        payload,
        video_id=clip["video_id"],
        segment_start_time_s=clip["segment_start_time_s"],
        when_start_local_s=clip["when_start_local_s"],
        when_end_local_s=clip["when_end_local_s"],
        model_source=MODEL_SOURCE,
    )
    write_policy_csv(paths["csv"], [row])

    print("")
    print("Policy CSV saved to:", paths["csv"])
    print("Reasoning:", row["reasoning_trace"])
    print("Meta action:", row["meta_action"])
    print("Reason:", row["explanation_reason"])
    return row


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("OOM free Alpamayo CROWD clip inference")
    print("=" * 60)

    config_path = resolve_config_path(args.config)
    config = load_config(config_path)
    clips = load_clip_list(args)

    print("")
    print("[1] Config:", config_path)
    print("    GPU expected:", config.system.gpu_name)
    print("    Resident layers:", str(config.residency.num_resident), "of", str(config.model.vlm_layers))
    print("    Predicted time:", str(round(config.predicted_performance.inference_time_s, 3)), "s per clip")
    print("    Clips to run:", len(clips))

    print("")
    print("[2] Loading model (this takes minutes; done once for all clips)")
    torch.cuda.empty_cache()
    gc.collect()

    stage_start = time.perf_counter()
    model = rf.load_model()
    vlm_layers, vblocks, expert_layers = rf.setup_gpu_essentials(model)
    print("    Model loaded in", round(time.perf_counter() - stage_start, 1), "s")

    if len(vlm_layers) != config.model.vlm_layers:
        print("VLM layer count mismatch", file=sys.stderr)
        print("Model layers:", len(vlm_layers), file=sys.stderr)
        print("Config layers:", config.model.vlm_layers, file=sys.stderr)
        sys.exit(2)

    print("")
    print("[3] Loading benchmark data cache")
    data_cache = rf.load_data()

    vlm_resident = list(config.residency.resident_indices)
    for index in vlm_resident:
        vlm_layers[index].to("cuda")

    vlm_offload = sorted(set(range(len(vlm_layers))) - set(vlm_resident))

    print("")
    print("[4] Installing DoubleBufHook")
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

    failures = 0
    for clip_index, clip in enumerate(clips, start=1):
        print("")
        print(f"[5] Clip {clip_index}/{len(clips)}: {clip['video_id']} @ {clip['segment_start_time_s']} s")
        paths = clip_output_paths(args.output_dir, clip["video_id"])
        try:
            prepare_crowd_data_cache(data_cache, clip)
            model_inputs = rf.prepare_inputs(model, data_cache)

            vlm_hook.reset()
            vis_hook.reset()
            exp_hook.reset()
            vis_hook.start()

            torch.manual_seed(SAMPLING["seed"])
            torch.cuda.manual_seed_all(SAMPLING["seed"])

            torch.cuda.reset_peak_memory_stats()
            print(f"    Running inference (expected ~{config.predicted_performance.inference_time_s:.1f} s)")
            start_time = time.perf_counter()

            with torch.autocast("cuda", dtype=torch.bfloat16):
                result = model.sample_trajectories_from_data_with_vlm_rollout(
                    data=rf.deep_copy_inputs(model_inputs),
                    top_p=SAMPLING["top_p"],
                    temperature=SAMPLING["temperature"],
                    num_traj_samples=SAMPLING["num_traj_samples"],
                    max_generation_length=SAMPLING["max_generation_length"],
                    return_extra=True,
                )

            torch.cuda.synchronize()
            end_time = time.perf_counter()

            inference_time_s = end_time - start_time
            peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

            print("    Inference time:", round(inference_time_s, 3), "s")
            print("    Peak VRAM (inference):", round(peak_gb, 2), "GB")

            run_meta = {
                "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
                "config_file": config_path,
                "model_id": rf.MODEL_ID,
                "num_resident_layers": config.residency.num_resident,
                "sampling": dict(SAMPLING),
                "num_time_steps": NUM_TIME_STEPS,
                "num_camera_slots": NUM_CAMERA_SLOTS,
                "image_size": [IMAGE_WIDTH, IMAGE_HEIGHT],
                "inference_time_s": round(inference_time_s, 3),
                "peak_vram_gb": round(peak_gb, 2),
            }

            print("")
            print("    Saving outputs")
            write_outputs(result, clip, paths, run_meta)
            write_clip_policy_csv(clip, paths)

            del result, model_inputs
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as exc:
            failures += 1
            print(f"    FAILED clip {clip['video_id']}: {exc}", file=sys.stderr)

    vlm_hook.remove()
    vis_hook.remove()
    exp_hook.remove()

    print("")
    if failures:
        print(f"Done with {failures} failed clip(s) of {len(clips)}.")
        sys.exit(1)
    print("Done.")


if __name__ == "__main__":
    main()
