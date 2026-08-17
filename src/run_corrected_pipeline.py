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

    # The batch runner can decide the gate for a whole round in one process
    # (gemma_gate_batch.py), which loads Gemma once instead of once per clip.
    # In that mode the state already holds the decision, and re-running stage 1
    # here would overwrite it -- workflow_runner rebuilds the state file.
    gate_precomputed = os.environ.get("OPTICARVIS_GATE_PRECOMPUTED", "0") == "1"

    if gate_precomputed:
        print("")
        print("Gate precomputed by gemma_gate_batch.py; skipping stages 1-2.")
    else:
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

    # Reconstruct the ego's future path so the ribbon can bend into real turns
    # (OPTICARVIS_VO_TRAJECTORY; the renderer picks the track up from
    # ego_trajectory.py's default output path). Failure is expected on some
    # scenes -- VO cannot recover motion in dense stop-and-go traffic -- and the
    # documented fallback is a straight in-lane ribbon, so a failed track must
    # not fail the job.
    if os.environ.get("OPTICARVIS_VO_TRAJECTORY", "0") == "1":
        vo_script = os.path.join(SRC_DIR, "ego_trajectory.py")
        completed = subprocess.run([sys.executable, vo_script], cwd=SRC_DIR)

        if completed.returncode != 0:
            print("ego_trajectory failed (code %d); ribbon stays straight in the ego lane."
                  % completed.returncode)
        elif os.environ.get("OPTICARVIS_FUTURE_ANCHOR", "1") == "1":
            # Trace the driven path onto the actual street pixels of every
            # frame (homography chains to the future frames). Needs the VO
            # track's ego poses, hence inside this branch. A failure only
            # costs the anchor precision: the renderer falls back to the
            # direct flat-ground projection of the same VO path.
            anchor_script = os.path.join(SRC_DIR, "future_anchor.py")
            completed = subprocess.run([sys.executable, anchor_script], cwd=SRC_DIR)

            if completed.returncode != 0:
                print("future_anchor failed (code %d); renderer falls back to "
                      "the direct flat-ground path." % completed.returncode)

    run_step("final_preview_renderer.py")

    state = load_state()
    outputs = state.get("outputs", {})

    print("")
    print("Pipeline complete.")
    print("State JSON:", STATE_JSON)
    print_final_video(outputs)


if __name__ == "__main__":
    main()
