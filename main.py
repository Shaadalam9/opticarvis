"""Main entry point for the OptiCarVis batch pipeline.

Usage:
    python main.py
    python main.py --rebuild-jobs
    python main.py --only-build-jobs
    python main.py 10 0

Default behaviour:
    python main.py builds clip_jobs.jsonl if needed, then runs all pending
    generated clip jobs. Jobs with existing completed results are skipped.
"""

import json
import os
import subprocess
import sys

import common


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
WORKFLOW_OUTPUTS = os.path.join(PROJECT_ROOT, "workflow_outputs")

CLIP_JOBS_JSONL = os.path.join(WORKFLOW_OUTPUTS, "clip_jobs.jsonl")
PENDING_CLIP_JOBS_JSONL = os.path.join(WORKFLOW_OUTPUTS, "clip_jobs_pending.jsonl")

CLIP_JOB_BUILDER = os.path.join(SRC_DIR, "clip_job_builder.py")
BATCH_PIPELINE = os.path.join(SRC_DIR, "batch_corrected_pipeline.py")


def config_int(key, default):
    value = common.get_configs(key)

    if value is None:
        return default

    try:
        number = int(float(value))
    except ValueError:
        return default

    if number <= 0:
        return default

    return number


def ensure_workflow_outputs():
    os.makedirs(WORKFLOW_OUTPUTS, exist_ok=True)


def clip_jobs_missing():
    if not os.path.exists(CLIP_JOBS_JSONL):
        return True

    if os.path.getsize(CLIP_JOBS_JSONL) == 0:
        return True

    return False


def read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path):
    rows = []

    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()

            if text:
                rows.append(json.loads(text))

    return rows


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_script(script_path, args=None, env_extra=None):
    if args is None:
        args = []

    command = [sys.executable, script_path] + args

    env = os.environ.copy()

    if env_extra:
        env.update(env_extra)

    print()
    print("=" * 70)
    print("Running:", " ".join(command))
    print("=" * 70)

    subprocess.run(command, cwd=SRC_DIR, check=True, env=env)


def format_start_tag(value):
    number = float(value)

    if abs(number - round(number)) < 0.000001:
        return str(int(round(number)))

    text = ("%.3f" % number).rstrip("0").rstrip(".")
    text = text.replace("-", "m")
    text = text.replace(".", "p")

    return text


def job_tag(job):
    clip_video = str(job.get("clip_video", "")).strip()

    if clip_video:
        name = os.path.basename(clip_video)
        stem, extension = os.path.splitext(name)

        if extension.lower() == ".mp4":
            clip_length = int(round(float(job.get("clip_length_s", 30))))
            suffix = "_" + str(clip_length) + "s"

            if stem.endswith(suffix):
                return stem[: -len(suffix)]

    return (
        str(job.get("video_id"))
        + "_"
        + format_start_tag(job.get("segment_start_time_s", 0))
    )


def state_json_for_job(job):
    tag = job_tag(job)
    return os.path.join(WORKFLOW_OUTPUTS, tag + "_workflow_state.json")


def gate_json_for_job(job):
    tag = job_tag(job)
    return os.path.join(
        WORKFLOW_OUTPUTS,
        "gemma_reasoning",
        tag + "_gemma_gate.json",
    )


def final_render_exists_for_job(job):
    tag = job_tag(job)
    final_dir = os.path.join(WORKFLOW_OUTPUTS, "final_renders")

    if not os.path.isdir(final_dir):
        return False

    for name in os.listdir(final_dir):
        lower_name = name.lower()

        if not lower_name.endswith(".mp4"):
            continue

        if not name.startswith(tag):
            continue

        path = os.path.join(final_dir, name)

        if os.path.getsize(path) > 0:
            return True

    return False


def output_video_exists(state):
    outputs = state.get("outputs", {})

    candidate_keys = [
        "roadline_v3_final_preview_video",
        "roadline_v3_final_preview_video_vehicles",
        "clean_final_preview_video",
        "final_preview_video",
    ]

    for key in candidate_keys:
        path = outputs.get(key)

        if path and os.path.exists(path) and os.path.getsize(path) > 0:
            return True

    return False


def processed_status(job):
    state_json = state_json_for_job(job)

    if final_render_exists_for_job(job):
        return "final_render_exists"

    if os.path.exists(state_json):
        state = read_json(state_json)

        current_stage = str(state.get("current_stage", "")).strip()
        explanation = state.get("explanation", {})

        if current_stage == "gemma_gate_no":
            return "gemma_no_state_exists"

        if explanation.get("needed") is False:
            return "gemma_no_state_exists"

        if output_video_exists(state):
            return "final_render_exists"

    gate_json = gate_json_for_job(job)

    if os.path.exists(gate_json):
        gate = read_json(gate_json)

        if gate.get("proper_time_to_explain") is False:
            return "gemma_no_gate_exists"

        if gate.get("proper_time_to_explain") is True and final_render_exists_for_job(job):
            return "final_render_exists"

    return ""


def build_pending_jobs(all_jobs):
    pending_jobs = []
    skipped_counts = {}

    for job in all_jobs:
        status = processed_status(job)

        if status:
            skipped_counts[status] = skipped_counts.get(status, 0) + 1
        else:
            pending_jobs.append(job)

    return pending_jobs, skipped_counts


def print_skip_summary(total_jobs, pending_jobs, skipped_counts):
    skipped_total = total_jobs - len(pending_jobs)

    print()
    print("Pending job summary")
    print("===================")
    print("total_jobs:", total_jobs)
    print("already_processed:", skipped_total)
    print("pending_jobs:", len(pending_jobs))

    for key in sorted(skipped_counts):
        print(key + ":", skipped_counts[key])


def run_all_pending_jobs():
    all_jobs = read_jsonl(CLIP_JOBS_JSONL)
    pending_jobs, skipped_counts = build_pending_jobs(all_jobs)

    print_skip_summary(len(all_jobs), pending_jobs, skipped_counts)

    if not pending_jobs:
        print()
        print("No pending jobs. Everything already has completed results.")
        return

    write_jsonl(PENDING_CLIP_JOBS_JSONL, pending_jobs)

    env_extra = {
        "OPTICARVIS_CLIP_JOBS": PENDING_CLIP_JOBS_JSONL,
    }

    chunk_size = config_int("BATCH_CHUNK_SIZE", 5)

    print()
    print("Running pending jobs in chunks")
    print("==============================")
    print("chunk_size:", chunk_size)

    start = 0

    while start < len(pending_jobs):
        end = min(start + chunk_size, len(pending_jobs))
        current_count = end - start

        print()
        print("Pending chunk:", str(start + 1) + "-" + str(end), "of", len(pending_jobs))

        run_script(
            BATCH_PIPELINE,
            [str(current_count), str(start)],
            env_extra=env_extra,
        )

        start = end


def parse_args(argv):
    rebuild_jobs = False
    only_build_jobs = False
    positional = []

    for value in argv:
        if value == "--rebuild-jobs":
            rebuild_jobs = True
        elif value == "--only-build-jobs":
            only_build_jobs = True
        else:
            positional.append(value)

    return rebuild_jobs, only_build_jobs, positional


def main():
    rebuild_jobs, only_build_jobs, positional = parse_args(sys.argv[1:])

    ensure_workflow_outputs()

    if rebuild_jobs or clip_jobs_missing():
        print("clip_jobs.jsonl is missing, empty, or rebuild was requested.")
        run_script(CLIP_JOB_BUILDER)
    else:
        print("clip_jobs.jsonl exists:", CLIP_JOBS_JSONL)

    if only_build_jobs:
        print("Only build jobs requested. Stopping here.")
        return

    if positional:
        if len(positional) == 1:
            jobs_to_run = positional[0]
            start_index = "0"
        else:
            jobs_to_run = positional[0]
            start_index = positional[1]

        run_script(BATCH_PIPELINE, [jobs_to_run, start_index])
        return

    run_all_pending_jobs()


if __name__ == "__main__":
    main()
