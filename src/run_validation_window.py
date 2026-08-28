"""Build and run one explicit OptiCarVis validation window.

Usage from the repository root:

    python src/run_validation_window.py 3ai7SUaPoHM 15 30

The first two values are the source video id and start time in seconds. Clip
length defaults to 30 seconds. Set OPTICARVIS_VALIDATION_BUILD_ONLY=1 to write
the isolated job file without starting the pipeline.
"""

import json
import os
import re
import subprocess
import sys


SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import build_candidate_index


def env_bool(name, default=False):
    value = os.environ.get(name)

    if value is None or str(value).strip() == "":
        return bool(default)

    return str(value).strip().lower() in ("1", "true", "yes", "y", "on")


def number_tag(value):
    number = float(value)

    if number.is_integer():
        return str(int(number))

    return ("%0.3f" % number).rstrip("0").rstrip(".").replace(".", "p")


def clean_cli_video_id(value):
    text = str(value).strip()

    if not re.fullmatch(r"[A-Za-z0-9_-]{6,}", text):
        raise ValueError("Invalid video id: %s" % text)

    return text


def validation_job(video_id, start_s, clip_length_s):
    mapping_csv = build_candidate_index.config_path(
        "mapping",
        "mapping.csv",
        environment_names=["OPTICARVIS_MAPPING_CSV"],
    )
    video_root = build_candidate_index.config_path(
        "videos",
        "videos",
        environment_names=["OPTICARVIS_VIDEOS_DIR"],
    )
    workflow_outputs = build_candidate_index.config_path(
        "workflow_outputs",
        "workflow_outputs",
        environment_names=["OPTICARVIS_WORKFLOW_OUTPUTS"],
    )
    alpamayo_outputs = build_candidate_index.config_path(
        "alpamayo_outputs",
        "alpamayo_outputs",
        environment_names=["OPTICARVIS_ALPAMAYO_OUTPUTS"],
    )
    records = build_candidate_index.video_records(
        mapping_csv,
        video_root,
        [video_id],
    )
    record = records[0] if records else {
        "city": "Unknown",
        "country": "Unknown",
        "continent": "Unknown",
        "video_id": video_id,
        "source_video": os.path.join(video_root, video_id + ".mp4"),
    }
    start_tag = number_tag(start_s)
    length_tag = number_tag(clip_length_s)
    job_tag = video_id + "_" + start_tag
    validation_dir = os.path.join(workflow_outputs, "validation_jobs")
    jobs_path = os.path.abspath(
        os.path.join(validation_dir, job_tag + "_jobs.jsonl")
    )
    master_path = os.path.abspath(
        os.path.join(validation_dir, job_tag + "_master_index.jsonl")
    )
    clip_root = os.path.join(alpamayo_outputs, "crowd_clips")
    alpamayo_json_root = os.path.join(alpamayo_outputs, "alpamayo_json")
    end_s = float(start_s) + float(clip_length_s)
    job = {
        "job_id": "validation_" + job_tag,
        "city_index": "validation_" + job_tag,
        "city": record["city"],
        "locality": record["city"],
        "country": record["country"],
        "continent": record["continent"],
        "video_id": video_id,
        "source_video": os.path.abspath(record["source_video"]),
        "segment_start_time_s": float(start_s),
        "segment_end_time_s": end_s,
        "clip_length_s": float(clip_length_s),
        "clip_video": os.path.abspath(
            os.path.join(clip_root, job_tag + "_" + length_tag + "s.mp4")
        ),
        "alpamayo_json": os.path.abspath(
            os.path.join(alpamayo_json_root, job_tag + "_alpamayo.json")
        ),
        "window_index": 0,
        "selection_method": "manual_validation",
        "selection_score": None,
        "selection_rank": None,
    }
    return job, jobs_path, master_path


def write_validation_job(job, jobs_path):
    os.makedirs(os.path.dirname(jobs_path), exist_ok=True)

    with open(jobs_path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(job, ensure_ascii=False) + "\n")

    return jobs_path


def main():
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("Usage:")
        print("python src/run_validation_window.py VIDEO_ID START_S [CLIP_LENGTH_S]")
        return 2

    try:
        video_id = clean_cli_video_id(sys.argv[1])
        start_s = float(sys.argv[2])
        clip_length_s = float(sys.argv[3]) if len(sys.argv) == 4 else 30.0
    except ValueError as error:
        print(str(error))
        return 2

    if start_s < 0:
        print("START_S must be zero or greater")
        return 2
    if clip_length_s <= 0:
        print("CLIP_LENGTH_S must be positive")
        return 2

    job, jobs_path, master_path = validation_job(
        video_id,
        start_s,
        clip_length_s,
    )
    write_validation_job(job, jobs_path)

    print("")
    print("Explicit validation window")
    print("==========================")
    print("video_id:", video_id)
    print("start_s:", start_s)
    print("clip_length_s:", clip_length_s)
    print("city:", job["city"])
    print("jobs_file:", jobs_path)

    if env_bool("OPTICARVIS_VALIDATION_BUILD_ONLY", False):
        print("build_only: true")
        return 0

    env = os.environ.copy()
    env["OPTICARVIS_CLIP_JOBS_JSONL"] = jobs_path
    env["OPTICARVIS_MASTER_CLIP_INDEX_JSONL"] = master_path
    command = [
        sys.executable,
        os.path.join(SRC_DIR, "batch_corrected_pipeline.py"),
        "1",
        "0",
    ]

    print("running:", " ".join(command))
    completed = subprocess.run(command, cwd=PROJECT_ROOT, env=env)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
