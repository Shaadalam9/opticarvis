"""Adapter between OptiCarVis clip jobs and NVIDIA Alpamayo2 Super.

This file is intentionally kept separate from the current OOM free Alpamayo
runner. The current runner accepts a single 30 second MP4 clip. Alpamayo2 Super
needs its own input conversion and inference logic.

Expected contract:
    input:
        --jobs-jsonl
        --output-dir
        --model-id

    output:
        one OptiCarVis compatible Alpamayo JSON per job, saved to output-dir.

The JSON must keep the same interface consumed by workflow_runner.py.
"""

import argparse
import json
import os


def read_jsonl(path):
    rows = []

    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()

            if text:
                rows.append(json.loads(text))

    return rows


def job_output_path(job, output_dir):
    configured_path = str(job.get("alpamayo_json", "")).strip()

    if configured_path:
        return configured_path

    video_id = str(job.get("video_id"))
    start_time = int(round(float(job.get("segment_start_time_s", 0))))

    return os.path.join(
        output_dir,
        video_id + "_" + str(start_time) + "_alpamayo.json",
    )


def write_not_implemented_output(job, output_dir, model_id):
    output_path = job_output_path(job, output_dir)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    payload = {
        "video_id": job.get("video_id"),
        "segment_start_time_s": job.get("segment_start_time_s"),
        "clip_video": job.get("clip_video"),
        "backend": "alpamayo2_super",
        "model_id": model_id,
        "status": "adapter_not_implemented",
        "reasoning": "",
        "meta_action": "",
        "action": "",
        "error": (
            "Alpamayo2 Super adapter has been selected, but the real "
            "Alpamayo2 input conversion and inference implementation has "
            "not been added yet."
        ),
    }

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-id", required=True)
    args = parser.parse_args()

    jobs = read_jsonl(args.jobs_jsonl)
    os.makedirs(args.output_dir, exist_ok=True)

    print()
    print("Alpamayo2 Super adapter")
    print("=======================")
    print("jobs:", len(jobs))
    print("model_id:", args.model_id)
    print("output_dir:", args.output_dir)

    for job in jobs:
        output_path = write_not_implemented_output(
            job,
            args.output_dir,
            args.model_id,
        )

        print("Wrote placeholder:", output_path)

    raise RuntimeError(
        "Alpamayo2 Super adapter placeholder was called. "
        "Implement real Alpamayo2 Super inference here on the target system."
    )


if __name__ == "__main__":
    main()
