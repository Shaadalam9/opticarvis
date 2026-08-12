r"""Run one OptiCarVis clip job.

This is the single job entry point. It is job aware through pipeline_common.py.
The batch runner calls this file with environment variables for each
city, video, and start time job.

Order:
1. Alpamayo context extraction.
2. Gemma4 gate decides whether this is a proper time to explain.
3. If Gemma4 says no, stop.
4. If Gemma4 says yes, run segmentation, depth, MIRAGE planning, and render.
"""

import json
import os
import subprocess
import sys

from pipeline_common import (
    STATE_JSON,
    VIDEO_ID,
    SEGMENT_START_TIME_S,
    CLIP_LENGTH_S,
    JOB_ID,
    LOCALITY,
    COUNTRY,
    CONTINENT,
)


SRC_DIR = os.path.dirname(os.path.abspath(__file__))


def run_step(script_name):
    script_path = os.path.join(SRC_DIR, script_name)

    if not os.path.isfile(script_path):
        print("Missing script:", script_path)
        raise SystemExit(1)

    print("")
    print("=" * 70)
    print("Running:", script_name)
    print("=" * 70)

    subprocess.run(
        [sys.executable, script_path],
        cwd=SRC_DIR,
        check=True,
    )


def load_state():
    if not os.path.isfile(STATE_JSON):
        print("Missing workflow state:", STATE_JSON)
        raise SystemExit(1)

    with open(STATE_JSON, "r", encoding="utf-8") as handle:
        return json.load(handle)


def print_job_header():
    print("")
    print("Corrected OptiCarVis workflow")
    print("=============================")
    print("job_id:", JOB_ID)
    print("video_id:", VIDEO_ID)
    print("segment_start_time_s:", SEGMENT_START_TIME_S)
    print("clip_length_s:", CLIP_LENGTH_S)

    if LOCALITY or COUNTRY or CONTINENT:
        print("city:", LOCALITY)
        print("country:", COUNTRY)
        print("continent:", CONTINENT)

    print("")
    print("Alpamayo provides context.")
    print("Gemma4 decides whether this is a proper time to explain.")
    print("MIRAGE runs only if Gemma4 says yes.")


def print_final_video(outputs):
    keys = [
        "roadline_v3_final_preview_video",
        "roadline_v3_final_preview_video_vehicles",
        "clean_final_preview_video",
        "final_preview_video",
    ]

    for key in keys:
        value = outputs.get(key)
        if value:
            print("Final video:", value)
            return

    print("Final video: not found in workflow state outputs")


def main():
    print_job_header()

    run_step("workflow_runner.py")
    run_step("gemma_reasoning_module.py")

    state = load_state()
    explanation = state.get("explanation", {})
    explanation_needed = bool(explanation.get("needed", False))

    if not explanation_needed:
        print("")
        print("Gemma4 gate said NO.")
        print("No segmentation, depth, MIRAGE or rendering will run.")
        print("Reason:", explanation.get("decision_reason", ""))
        print("State JSON:", STATE_JSON)
        return

    print("")
    print("Gemma4 gate said YES.")
    print("Continuing to visual grounding and MIRAGE display.")

    run_step("semantic_segmentation_module.py")
    run_step("depth_estimation_module.py")
    run_step("mirage_effect_planner.py")
    run_step("final_preview_renderer.py")

    state = load_state()
    outputs = state.get("outputs", {})

    print("")
    print("Pipeline complete.")
    print("State JSON:", STATE_JSON)
    print_final_video(outputs)


if __name__ == "__main__":
    main()