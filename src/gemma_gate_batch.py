r"""Run context extraction and the Gemma 4 gate for many jobs in one process.

The per-job pipeline pays the full Gemma load (~9.6 GB, one to two minutes) for
every clip, because each stage runs as a fresh subprocess. Measured on the DGX
Spark, a clip the gate declines spends nearly all of its 91-154 s in that load.
Over a 100 city batch -- and several windows per city once the batch scans for
a window the gate approves -- the reloads dwarf the decisions.

This driver loads Gemma once and walks the job list:

    <python> src/gemma_gate_batch.py --jobs-jsonl <jobs.jsonl>

Per job it runs the same two stages the per-job pipeline would (workflow_runner,
then gemma_reasoning_module), by setting the job's environment and reloading
those modules -- their paths and job identity are fixed at import time, which is
exactly why the model cache lives in gemma_model_cache, a module this driver
never reloads.

Output is the same per-job state JSON the per-job pipeline writes; the batch
runner then reads state["explanation"]["needed"] and only spawns the full
pipeline (with OPTICARVIS_GATE_PRECOMPUTED=1) for jobs the gate approved.

Exit code 0 when every job got a decision, 1 otherwise.
"""

import argparse
import importlib
import json
import os
import sys


SRC_DIR = os.path.dirname(os.path.abspath(__file__))

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


# Job fields mapped into the environment the stage modules read. Mirrors
# batch_corrected_pipeline.job_environment; kept as data so a drift is one line.
JOB_ENV_KEYS = (
    ("OPTICARVIS_JOB_ID", "job_id"),
    ("OPTICARVIS_VIDEO_ID", "video_id"),
    ("OPTICARVIS_SEGMENT_START_S", "segment_start_time_s"),
    ("OPTICARVIS_CLIP_LENGTH_S", "clip_length_s"),
    ("OPTICARVIS_SOURCE_VIDEO", "source_video"),
    ("OPTICARVIS_CLIP_VIDEO", "clip_video"),
    ("OPTICARVIS_ALPAMAYO_JSON", "alpamayo_json"),
    ("OPTICARVIS_LOCALITY", "locality"),
    ("OPTICARVIS_COUNTRY", "country"),
    ("OPTICARVIS_CONTINENT", "continent"),
)


def read_jobs(path):
    jobs = []

    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()

            if text:
                jobs.append(json.loads(text))

    return jobs


def apply_job_environment(job):
    for env_name, key in JOB_ENV_KEYS:
        value = job.get(key)

        if value is None:
            os.environ.pop(env_name, None)
        else:
            os.environ[env_name] = str(value)


def reload_stage_modules():
    """Fresh stage modules under the current job's environment.

    pipeline_common must be reloaded first: the stage modules read its
    constants with `from pipeline_common import ...` at their own import, so
    reloading them binds whatever pipeline_common currently holds.
    """
    import pipeline_common

    importlib.reload(pipeline_common)

    import workflow_runner
    import gemma_reasoning_module

    return importlib.reload(workflow_runner), importlib.reload(gemma_reasoning_module)


def gate_one_job(job):
    apply_job_environment(job)
    workflow_runner, gemma_reasoning_module = reload_stage_modules()

    workflow_runner.main()
    gemma_reasoning_module.main()


def main():
    parser = argparse.ArgumentParser(
        description="Decide the Gemma 4 gate for a list of clip jobs.",
    )
    parser.add_argument("--jobs-jsonl", required=True)
    args = parser.parse_args()

    jobs = read_jobs(args.jobs_jsonl)

    print("")
    print("Gemma 4 gate batch")
    print("==================")
    print("jobs:", len(jobs))

    decided = failed = 0
    gate_unavailable = False

    for index, job in enumerate(jobs, start=1):
        print("")
        print("[%d/%d] %s" % (index, len(jobs), job.get("job_id", "?")))

        try:
            gate_one_job(job)
            decided += 1
        # SystemExit as well: the stage modules exit on their own missing-input
        # paths (a state file, a clip), and SystemExit does not inherit from
        # Exception -- an uncaught one would kill this driver and leave every
        # remaining job undecided.
        except (Exception, SystemExit) as error:
            print("    GATE FAILED:", type(error).__name__, str(error)[:400])
            failed += 1

            # With OPTICARVIS_REQUIRE_GEMMA_GATE set, a model that cannot LOAD
            # raises rather than falling back -- and it will raise identically
            # for every remaining job, a load attempt each time. Stop retrying;
            # the batch runner records the jobs without a decision as failed.
            # Match the specific message from call_gemma4_gate's guard, not the
            # env var name: a per-clip inference error can mention the variable
            # without meaning the model is unavailable.
            if str(error).startswith("Gemma 4 gate could not run"):
                gate_unavailable = True
                remaining = len(jobs) - index
                if remaining:
                    print("    Gate unavailable; skipping the remaining %d job(s)."
                          % remaining)
                break

    print("")
    print("decided:", decided, "| failed:", failed + (len(jobs) - decided - failed))

    if gate_unavailable or failed:
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
