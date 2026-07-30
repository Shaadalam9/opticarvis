r"""Stage 1: extract Alpamayo context only.

Correct workflow:
Alpamayo provides trajectory, reasoning, action and uncertainty context.
Gemma4 later decides whether this is a proper time to explain.
"""

import json
import math
import os

from pipeline_common import (
    PROJECT_ROOT,
    VIDEO_ID,
    SEGMENT_START_TIME_S,
    CLIP_LENGTH_S,
    ALPAMAYO_JSON,
    STATE_JSON,
    current_job_summary,
    read_json,
    write_json,
    clamp,
)

OUTPUT_ROOT = PROJECT_ROOT + "/workflow_outputs"
ALPAMAYO_CONTEXT_JSON = OUTPUT_ROOT + "/alpamayo_traces/" + VIDEO_ID + "_" + str(int(SEGMENT_START_TIME_S)) + "_alpamayo_context.json"

OUTPUT_DIRS = [
    OUTPUT_ROOT,
    OUTPUT_ROOT + "/alpamayo_traces",
    OUTPUT_ROOT + "/gemma_reasoning",
    OUTPUT_ROOT + "/segmentation",
    OUTPUT_ROOT + "/depth",
    OUTPUT_ROOT + "/mirage",
    OUTPUT_ROOT + "/final_renders",
]


def make_dirs():
    for output_dir in OUTPUT_DIRS:
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)


def flatten(value):
    if isinstance(value, list):
        output = []
        for item in value:
            output.extend(flatten(item))
        return output
    return [value]


def clean_text(value):
    if value is None:
        return ""
    text = str(value).strip()
    while text.startswith("[") and text.endswith("]"):
        text = text[1:-1].strip()
    while len(text) >= 2 and text[0] in ["'", '"'] and text[-1] == text[0]:
        text = text[1:-1].strip()
    return text


def get_extra_text(payload, key):
    extra = payload.get("extra", {})
    values = extra.get("values", {})
    field = values.get(key, "")
    if isinstance(field, dict):
        if "value" in field:
            field = field.get("value")
        elif "values" in field:
            field = field.get("values")
        elif "items" in field:
            field = field.get("items")
    for item in flatten(field):
        text = clean_text(item)
        if text:
            return text
    return ""


def get_pred_xyz(payload):
    pred_xyz = payload.get("pred_xyz", {})
    values = pred_xyz.get("values", [])
    flat = flatten(values)
    points = []
    current = []
    for item in flat:
        if isinstance(item, int) or isinstance(item, float):
            current.append(float(item))
            if len(current) == 3:
                points.append(current)
                current = []
    return points


def mean(values):
    if not values:
        return 0.0
    return sum(values) / len(values)


def trajectory_metrics(points):
    if len(points) < 2:
        return {
            "trajectory_point_count": len(points),
            "start_speed_mps": 0.0,
            "end_speed_mps": 0.0,
            "slowdown_score": 0.0,
            "lateral_deviation_m": 0.0,
            "lateral_deviation_score": 0.0,
            "trajectory_change_score": 0.0,
        }

    speeds = []
    dt = 0.1
    for i in range(1, len(points)):
        x0, y0, z0 = points[i - 1]
        x1, y1, z1 = points[i]
        distance = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2 + (z1 - z0) ** 2)
        speeds.append(distance / dt)

    start_speed = mean(speeds[: min(8, len(speeds))])
    end_speed = mean(speeds[-min(8, len(speeds)):])

    if start_speed > 0.1:
        slowdown_score = clamp((start_speed - end_speed) / start_speed, 0.0, 1.0)
    else:
        slowdown_score = 0.0

    lateral_deviation = max([abs(p[1]) for p in points])
    lateral_deviation_score = clamp(lateral_deviation / 1.5, 0.0, 1.0)
    trajectory_change_score = max(slowdown_score, lateral_deviation_score)

    return {
        "trajectory_point_count": len(points),
        "start_speed_mps": round(start_speed, 3),
        "end_speed_mps": round(end_speed, 3),
        "slowdown_score": round(slowdown_score, 3),
        "lateral_deviation_m": round(lateral_deviation, 3),
        "lateral_deviation_score": round(lateral_deviation_score, 3),
        "trajectory_change_score": round(trajectory_change_score, 3),
    }


def infer_action(reasoning):
    text = reasoning.lower()
    if "stop" in text or "traffic light" in text or "red light" in text:
        return "stop"
    if "yield" in text or "pedestrian" in text or "crosswalk" in text:
        return "slow_or_yield"
    if "slow" in text or "brake" in text:
        return "slow_or_yield"
    if "nudge" in text or "clearance" in text:
        return "nudge"
    if "turn" in text:
        return "turn"
    return "continue"


def infer_scene_cause(reasoning):
    text = reasoning.lower()
    if "pedestrian" in text or "crosswalk" in text:
        return "pedestrian_crosswalk_interaction"
    if "traffic light" in text or "red light" in text:
        return "traffic_light_interaction"
    if "vehicle" in text or "traffic" in text:
        return "vehicle_or_traffic_interaction"
    if "cone" in text or "construction" in text or "clearance" in text:
        return "construction_or_clearance_interaction"
    if "turn" in text:
        return "turning_manoeuvre"
    return "unclassified_driving_context"


def uncertainty_score(reasoning, action, metrics):
    text = reasoning.lower()
    score = 0.2
    if action in ["slow_or_yield", "nudge", "turn"]:
        score = max(score, 0.5)
    if "pedestrian" in text or "crosswalk" in text:
        score = max(score, 0.65)
    if "ambiguous" in text or "uncertain" in text or "maybe" in text or "possibly" in text:
        score = max(score, 0.8)
    if metrics["trajectory_change_score"] >= 0.5:
        score = max(score, 0.55)
    return round(score, 3)


def is_candidate(action, scene_cause, uncertainty, metrics):
    if action in ["stop", "slow_or_yield", "nudge", "turn"]:
        return True
    if scene_cause != "unclassified_driving_context":
        return True
    if uncertainty >= 0.5:
        return True
    if metrics["trajectory_change_score"] >= 0.4:
        return True
    return False


def build_state(payload):
    reasoning = get_extra_text(payload, "cot")
    meta_action = get_extra_text(payload, "meta_action")
    answer = get_extra_text(payload, "answer")
    points = get_pred_xyz(payload)
    metrics = trajectory_metrics(points)
    action = infer_action(reasoning)
    scene_cause = infer_scene_cause(reasoning)
    uncertainty = uncertainty_score(reasoning, action, metrics)
    candidate_context = is_candidate(action, scene_cause, uncertainty, metrics)

    if candidate_context:
        gemma_status = "pending"
        explanation_status = "pending_gemma_gate"
    else:
        gemma_status = "skipped_no_candidate_context"
        explanation_status = "not_needed_no_candidate_context"

    alpamayo_context = {
        "video_id": VIDEO_ID,
        "segment_start_time_s": SEGMENT_START_TIME_S,
        "clip_length_s": CLIP_LENGTH_S,
        "job": current_job_summary(),
        "workflow_stage": "alpamayo_context_extraction",
        "alpamayo_action": action,
        "scene_cause": scene_cause,
        "alpamayo_reasoning_trace": reasoning,
        "alpamayo_meta_action_raw": meta_action,
        "alpamayo_answer_raw": answer,
        "trajectory_metrics": metrics,
        "uncertainty_score": uncertainty,
        "candidate_context": candidate_context,
        "gemma_gate_required": candidate_context,
        "proper_time_to_explain": None,
        "explanation_needed": None,
        "explanation_status": explanation_status,
        "trajectory_points_xyz": points,
        "important_note": "Alpamayo is context only. Gemma4 decides whether this is a proper time to explain.",
    }

    state = {
        "video_id": VIDEO_ID,
        "segment_start_time_s": SEGMENT_START_TIME_S,
        "clip_length_s": CLIP_LENGTH_S,
        "job": current_job_summary(),
        "current_stage": "alpamayo_context_extraction_complete",
        "pipeline_version": "alpamayo_context_then_gemma_gate_v2",
        "inputs": current_job_summary(),
        "outputs": {
            "alpamayo_context_json": ALPAMAYO_CONTEXT_JSON,
            "state_json": STATE_JSON,
        },
        "alpamayo_context": alpamayo_context,
        "gemma_gate": {
            "required": candidate_context,
            "status": gemma_status,
            "proper_time_to_explain": None,
            "model_called": False,
        },
        "explanation": {
            "needed": None,
            "status": explanation_status,
            "decided_by": "pending_gemma4" if candidate_context else "alpamayo_no_candidate_context",
        },
        "next_modules": {
            "gemma": {"needed": candidate_context, "status": gemma_status},
            "semantic_segmentation": {"needed": None, "status": "blocked_until_gemma_yes"},
            "depth_estimation": {"needed": None, "status": "blocked_until_gemma_yes"},
            "mirage": {"needed": None, "status": "blocked_until_gemma_yes"},
            "final_render": {"needed": None, "status": "blocked_until_gemma_yes"},
        },
    }

    return alpamayo_context, state


def main():
    make_dirs()
    payload = read_json(ALPAMAYO_JSON, "Alpamayo JSON")
    context, state = build_state(payload)
    write_json(ALPAMAYO_CONTEXT_JSON, context)
    write_json(STATE_JSON, state)

    print("\nStage 1: Alpamayo context extraction")
    print("====================================")
    print("alpamayo_action:", context["alpamayo_action"])
    print("scene_cause:", context["scene_cause"])
    print("reasoning:", context["alpamayo_reasoning_trace"])
    print("uncertainty_score:", context["uncertainty_score"])
    print("candidate_context:", context["candidate_context"])
    print("gemma_gate_required:", context["gemma_gate_required"])
    print("explanation_status:", context["explanation_status"])
    print("\nAlpamayo did not decide explanation_needed. Gemma4 must decide next.")
    print("\nContext JSON:", ALPAMAYO_CONTEXT_JSON)
    print("State JSON:", STATE_JSON)


if __name__ == "__main__":
    main()
