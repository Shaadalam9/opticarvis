r"""Run the corrected Alpamayo to Gemma4 gate pipeline.

Order:
1. Alpamayo context extraction.
2. Gemma4 gate decides whether this is a proper time to explain.
3. If Gemma4 says no, stop.
4. If Gemma4 says yes, run segmentation, depth, MIRAGE planning and render.
"""

import json
import os
import subprocess
import sys

PROJECT_ROOT = "C:/Users/localadmin/Desktop/Shadab"
OPTICARVIS_ROOT = PROJECT_ROOT + "/opticarvis"
VIDEO_ID = "TuCsyBF3nHU"
SEGMENT_START_TIME_S = 4630.0
STATE_JSON = PROJECT_ROOT + "/workflow_outputs/" + VIDEO_ID + "_" + str(int(SEGMENT_START_TIME_S)) + "_workflow_state.json"


def run_step(script_name):
    script_path = OPTICARVIS_ROOT + "/" + script_name
    if not os.path.isfile(script_path):
        print("Missing script:", script_path)
        raise SystemExit(1)

    print("\n" + "=" * 70)
    print("Running:", script_name)
    print("=" * 70)
    subprocess.run([sys.executable, script_path], cwd=OPTICARVIS_ROOT, check=True)


def load_state():
    if not os.path.isfile(STATE_JSON):
        print("Missing workflow state:", STATE_JSON)
        raise SystemExit(1)
    with open(STATE_JSON, "r", encoding="utf-8") as handle:
        return json.load(handle)


def main():
    print("\nCorrected OptiCarVis workflow")
    print("=============================")
    print("Alpamayo provides context.")
    print("Gemma4 decides whether this is a proper time to explain.")
    print("MIRAGE runs only if Gemma4 says yes.")

    run_step("workflow_runner.py")
    run_step("gemma_reasoning_module.py")

    state = load_state()
    explanation = state.get("explanation", {})
    explanation_needed = bool(explanation.get("needed", False))

    if not explanation_needed:
        print("\nGemma4 gate said NO.")
        print("No segmentation, depth, MIRAGE or rendering will run.")
        print("Reason:", explanation.get("decision_reason", ""))
        return

    print("\nGemma4 gate said YES.")
    print("Continuing to visual grounding and MIRAGE display.")

    run_step("semantic_segmentation_module.py")
    run_step("depth_estimation_module.py")
    run_step("mirage_effect_planner.py")
    run_step("final_preview_renderer.py")

    state = load_state()
    outputs = state.get("outputs", {})

    print("\nPipeline complete.")
    print("State JSON:", STATE_JSON)
    if outputs.get("clean_final_preview_video"):
        print("Final video:", outputs.get("clean_final_preview_video"))
    elif outputs.get("final_preview_video"):
        print("Final video:", outputs.get("final_preview_video"))


if __name__ == "__main__":
    main()