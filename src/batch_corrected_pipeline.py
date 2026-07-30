r"""Batch entry point for the 100 city OptiCarVis analysis.

Input:
    C:/Users/localadmin/Desktop/Shadab/workflow_outputs/clip_jobs.jsonl

Output:
    C:/Users/localadmin/Desktop/Shadab/workflow_outputs/master_clip_index.jsonl

Usage:
    python batch_corrected_pipeline.py
    python batch_corrected_pipeline.py 20
    python batch_corrected_pipeline.py 20 100

The second form runs only 20 jobs.
The third form starts at job index 100 and runs 20 jobs.

Important:
    This runner expects each job's Alpamayo JSON to exist before the main
    pipeline is run. If the Alpamayo JSON is missing, the job is recorded as
    missing_alpamayo_json and skipped.

    You can plug in your own Alpamayo command with:
        set OPTICARVIS_ALPAMAYO_COMMAND=...

    The command may use these placeholders:
        {clip_video}
        {alpamayo_json}
        {video_id}
        {segment_start_time_s}
        {job_id}
"""

import json
import os
import shlex
import subprocess
import sys
import time

from pipeline_common import (
    PROJECT_ROOT,
    WORKFLOW_OUTPUTS,
    ensure_dir,
    append_jsonl,
    ffmpeg_path,
)


CLIP_JOBS_JSONL = os.environ.get(
    "OPTICARVIS_CLIP_JOBS",
    WORKFLOW_OUTPUTS + "/clip_jobs.jsonl",
)

MASTER_INDEX_JSONL = os.environ.get(
    "OPTICARVIS_MASTER_CLIP_INDEX",
    WORKFLOW_OUTPUTS + "/master_clip_index.jsonl",
)

PIPELINE_SCRIPT = os.environ.get(
    "OPTICARVIS_SINGLE_PIPELINE",
    os.path.dirname(os.path.abspath(__file__)) + "/run_corrected_pipeline.py",
)

EXTRACT_CLIPS = os.environ.get("OPTICARVIS_EXTRACT_CLIPS", "1") == "1"
SKIP_EXISTING_STATE = os.environ.get("OPTICARVIS_SKIP_EXISTING_STATE", "0") == "1"
ALPAMAYO_COMMAND = os.environ.get("OPTICARVIS_ALPAMAYO_COMMAND", "")


def read_jobs(path):
    if not os.path.isfile(path):
        print("Missing clip jobs file:")
        print(path)
        print("")
        print("Run first:")
        print("python clip_job_builder.py")
        raise SystemExit(1)

    jobs = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                jobs.append(json.loads(line))
    return jobs


def state_json_for(job):
    return (
        PROJECT_ROOT
        + "/workflow_outputs/"
        + job["video_id"]
        + "_"
        + str(int(float(job["segment_start_time_s"])))
        + "_workflow_state.json"
    )


def extract_clip(job):
    if os.path.isfile(job["clip_video"]):
        return True, "clip_already_exists"

    if not EXTRACT_CLIPS:
        return False, "clip_missing_extraction_disabled"

    if not os.path.isfile(job["source_video"]):
        return False, "missing_source_video"

    ffmpeg = ffmpeg_path()

    if ffmpeg is None:
        return False, "missing_ffmpeg"

    ensure_dir(os.path.dirname(job["clip_video"]))

    command = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-ss",
        str(job["segment_start_time_s"]),
        "-i",
        job["source_video"],
        "-t",
        str(job["clip_length_s"]),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-an",
        job["clip_video"],
    ]

    completed = subprocess.run(command, capture_output=True, text=True)

    if completed.returncode != 0 or not os.path.isfile(job["clip_video"]):
        message = completed.stderr.strip() if completed.stderr else "ffmpeg_failed"
        return False, message[:500]

    return True, "clip_extracted"


def run_alpamayo_if_configured(job):
    if os.path.isfile(job["alpamayo_json"]):
        return True, "alpamayo_json_already_exists"

    if not ALPAMAYO_COMMAND:
        return False, "missing_alpamayo_json"

    ensure_dir(os.path.dirname(job["alpamayo_json"]))

    command_text = ALPAMAYO_COMMAND.format(
        clip_video=job["clip_video"],
        alpamayo_json=job["alpamayo_json"],
        video_id=job["video_id"],
        segment_start_time_s=job["segment_start_time_s"],
        job_id=job["job_id"],
    )

    command = shlex.split(command_text)
    completed = subprocess.run(command, capture_output=True, text=True)

    if completed.returncode != 0 or not os.path.isfile(job["alpamayo_json"]):
        message = completed.stderr.strip() if completed.stderr else "alpamayo_command_failed"
        return False, message[:500]

    return True, "alpamayo_json_created"


def job_environment(job):
    env = os.environ.copy()

    env["OPTICARVIS_PROJECT_ROOT"] = PROJECT_ROOT
    env["OPTICARVIS_JOB_ID"] = job["job_id"]
    env["OPTICARVIS_VIDEO_ID"] = job["video_id"]
    env["OPTICARVIS_SEGMENT_START_S"] = str(job["segment_start_time_s"])
    env["OPTICARVIS_CLIP_LENGTH_S"] = str(job["clip_length_s"])
    env["OPTICARVIS_SOURCE_VIDEO"] = job["source_video"]
    env["OPTICARVIS_CLIP_VIDEO"] = job["clip_video"]
    env["OPTICARVIS_ALPAMAYO_JSON"] = job["alpamayo_json"]
    env["OPTICARVIS_LOCALITY"] = job.get("locality", "")
    env["OPTICARVIS_COUNTRY"] = job.get("country", "")
    env["OPTICARVIS_CONTINENT"] = job.get("continent", "")

    return env


def load_state_summary(job):
    state_json = state_json_for(job)

    if not os.path.isfile(state_json):
        return {
            "state_json": state_json,
            "state_available": False,
        }

    with open(state_json, "r", encoding="utf-8") as handle:
        state = json.load(handle)

    explanation = state.get("explanation", {})
    decision = state.get("decision", {})
    outputs = state.get("outputs", {})

    return {
        "state_json": state_json,
        "state_available": True,
        "current_stage": state.get("current_stage", ""),
        "proper_time_to_explain": bool(explanation.get("needed", False)),
        "explanation_status": explanation.get("status", ""),
        "decision_reason": explanation.get("decision_reason", ""),
        "alpamayo_action": decision.get("alpamayo_action", ""),
        "scene_cause": decision.get("scene_cause", ""),
        "uncertainty_score": decision.get("uncertainty_score", None),
        "display_target": decision.get("display_plan", {}).get("display_target", ""),
        "gemma_gate_json": outputs.get("gemma_gate_json", ""),
        "segmentation_json": outputs.get("segmentation_json", ""),
        "depth_json": outputs.get("depth_json", ""),
        "mirage_effect_plan_json": outputs.get("mirage_effect_plan_json", ""),
        "rendered_video": (
            outputs.get("roadline_v3_final_preview_video")
            or outputs.get("clean_final_preview_video")
            or outputs.get("final_preview_video")
            or ""
        ),
    }


def append_master(job, status, message, elapsed_s):
    record = dict(job)
    record["batch_status"] = status
    record["batch_message"] = message
    record["elapsed_s"] = round(elapsed_s, 3)
    record.update(load_state_summary(job))
    append_jsonl(MASTER_INDEX_JSONL, record)


def run_one_job(job, index, total):
    print("")
    print("=" * 80)
    print("Job %d/%d: %s" % (index + 1, total, job["job_id"]))
    print("=" * 80)

    started = time.time()

    existing_state = state_json_for(job)

    if SKIP_EXISTING_STATE and os.path.isfile(existing_state):
        elapsed = time.time() - started
        append_master(job, "skipped_existing_state", existing_state, elapsed)
        print("Skipped existing state:", existing_state)
        return

    ok, message = extract_clip(job)
    if not ok:
        elapsed = time.time() - started
        append_master(job, "skipped_" + message, message, elapsed)
        print("Skipped:", message)
        return

    ok, message = run_alpamayo_if_configured(job)
    if not ok:
        elapsed = time.time() - started
        append_master(job, "skipped_missing_alpamayo_json", message, elapsed)
        print("Skipped:", message)
        print("Expected Alpamayo JSON:", job["alpamayo_json"])
        return

    env = job_environment(job)

    completed = subprocess.run(
        [sys.executable, PIPELINE_SCRIPT],
        cwd=os.path.dirname(PIPELINE_SCRIPT),
        env=env,
    )

    elapsed = time.time() - started

    if completed.returncode != 0:
        append_master(job, "failed_pipeline", "return_code_" + str(completed.returncode), elapsed)
        raise SystemExit(completed.returncode)

    append_master(job, "complete", "pipeline_complete", elapsed)


def main():
    max_jobs = int(sys.argv[1]) if len(sys.argv) > 1 else None
    start_index = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    jobs = read_jobs(CLIP_JOBS_JSONL)

    selected = jobs[start_index:]
    if max_jobs is not None:
        selected = selected[:max_jobs]

    ensure_dir(os.path.dirname(MASTER_INDEX_JSONL))

    print("")
    print("OptiCarVis 100 city batch pipeline")
    print("==================================")
    print("jobs_file:", CLIP_JOBS_JSONL)
    print("master_index:", MASTER_INDEX_JSONL)
    print("available_jobs:", len(jobs))
    print("start_index:", start_index)
    print("jobs_to_run:", len(selected))
    print("extract_clips:", EXTRACT_CLIPS)

    if ALPAMAYO_COMMAND:
        print("alpamayo_command: configured")
    else:
        print("alpamayo_command: not configured, existing Alpamayo JSONs required")

    for offset, job in enumerate(selected):
        run_one_job(job, start_index + offset, len(jobs))

    print("")
    print("Batch complete.")
    print("Master index:", MASTER_INDEX_JSONL)


if __name__ == "__main__":
    main()
