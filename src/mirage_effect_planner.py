r"""Stage 5 MIRAGE style effect planner for the OptiCarVis workflow.

Purpose:
    Read the workflow state after Alpamayo, Gemma, segmentation, and depth.
    Choose a mediated reality visual effect plan.
    Save the plan as JSON.
    Update the workflow state for the final renderer.

Current implementation:
    This is a planner, not the full MIRAGE renderer.
    It chooses the display target, effect family, intensity, label, and visual layers.

MIRAGE style categories used here:
    augmented reality  -> add highlight, contour, label, future path
    diminished reality -> reduce background salience
    modified reality   -> alter visual emphasis of the causal region

Run from the single OptiCarVis uv venv:
    cd C:\Users\localadmin\Desktop\Shadab\opticarvis
    C:\Users\localadmin\Desktop\Shadab\opticarvis\.venv\Scripts\python.exe mirage_effect_planner.py
"""

import json
import os

from pipeline_common import (
    PROJECT_ROOT,
    VIDEO_ID,
    SEGMENT_START_TIME_S,
    read_json,
    write_json,
    clamp,
)

STATE_JSON = (
    PROJECT_ROOT
    + "/workflow_outputs/"
    + VIDEO_ID
    + "_"
    + str(int(SEGMENT_START_TIME_S))
    + "_workflow_state.json"
)

GEMMA_JSON = (
    PROJECT_ROOT
    + "/workflow_outputs/gemma_reasoning/"
    + VIDEO_ID
    + "_"
    + str(int(SEGMENT_START_TIME_S))
    + "_gemma_reasoning.json"
)

SEGMENTATION_JSON = (
    PROJECT_ROOT
    + "/workflow_outputs/segmentation/"
    + VIDEO_ID
    + "_"
    + str(int(SEGMENT_START_TIME_S))
    + "_segmentation.json"
)

DEPTH_JSON = (
    PROJECT_ROOT
    + "/workflow_outputs/depth/"
    + VIDEO_ID
    + "_"
    + str(int(SEGMENT_START_TIME_S))
    + "_depth.json"
)

OUTPUT_DIR = PROJECT_ROOT + "/workflow_outputs/mirage"

OUTPUT_JSON = (
    OUTPUT_DIR
    + "/"
    + VIDEO_ID
    + "_"
    + str(int(SEGMENT_START_TIME_S))
    + "_effect_plan.json"
)


def create_dirs():
    if not os.path.isdir(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)


def read_json_optional(input_file):
    if not os.path.isfile(input_file):
        return None

    with open(input_file, "r", encoding="utf-8") as handle:
        return json.load(handle)


def get_decision(workflow_state):
    return workflow_state.get("decision", {})


def get_display_plan(workflow_state):
    decision = get_decision(workflow_state)
    return decision.get("display_plan", {})


def get_depth_summary(depth_payload):
    aggregate = depth_payload.get("aggregate", {})

    return {
        "spatial_relevance": aggregate.get("spatial_relevance", "unknown"),
        "depth_supports_explanation": aggregate.get("depth_supports_explanation", False),
        "max_target_proximity_score": aggregate.get("max_target_proximity_score", 0.0),
        "closest_target_object": aggregate.get("closest_target_object"),
    }


def get_segmentation_summary(segmentation_payload):
    targets = segmentation_payload.get("segmentation_targets", {})

    return {
        "target_status": targets.get("target_status", "unknown"),
        "requested_display_target": targets.get("requested_display_target", ""),
        "target_classes": targets.get("target_classes", []),
        "detected_target_classes": targets.get("detected_target_classes", []),
        "object_counts": segmentation_payload.get("object_counts", {}),
    }


def get_gemma_summary(gemma_payload):
    if gemma_payload is None:
        return {
            "available": False,
            "semantic_reason": "",
            "recommended_display_text": "",
            "confidence": 0.0,
        }

    return {
        "available": True,
        "semantic_reason": gemma_payload.get("semantic_reason", ""),
        "recommended_display_text": gemma_payload.get("recommended_display_text", ""),
        "confidence": gemma_payload.get("confidence", 0.0),
        "causal_objects": gemma_payload.get("causal_objects", []),
        "causal_regions": gemma_payload.get("causal_regions", []),
    }


def choose_effect_family(action, uncertainty_score, spatial_relevance):
    if action == "continue" and uncertainty_score < 0.65:
        return "none"

    if uncertainty_score >= 0.75:
        return "augmented_plus_diminished"

    if action in ["stop", "slow_or_yield", "nudge", "turn"]:
        if spatial_relevance in ["high", "medium"]:
            return "augmented_reality"
        return "modified_reality"

    return "augmented_reality"


def choose_intensity(action, uncertainty_score, spatial_relevance):
    if action == "continue" and uncertainty_score < 0.65:
        return "none"

    score = uncertainty_score

    if spatial_relevance == "high":
        score += 0.20
    elif spatial_relevance == "medium":
        score += 0.10

    if action in ["stop", "slow_or_yield"]:
        score += 0.10

    score = clamp(score, 0.0, 1.0)

    if score >= 0.85:
        return "high"

    if score >= 0.55:
        return "medium"

    return "low"


def choose_visual_layers(display_target, effect_family, intensity, segmentation_summary, depth_summary):
    if effect_family == "none":
        return []

    layers = []

    if display_target == "pedestrians_and_crosswalk":
        layers.append(
            {
                "layer_name": "pedestrian_instance_highlight",
                "source": "semantic_segmentation",
                "target_classes": ["person"],
                "effect": "soft_contour_and_mask_glow",
                "priority": 1,
            }
        )
        layers.append(
            {
                "layer_name": "crosswalk_attention_region",
                "source": "future_road_segmentation",
                "target_classes": ["crosswalk", "road_marking"],
                "effect": "subtle_region_overlay",
                "priority": 2,
                "status": "planned_not_available_yet",
            }
        )

    elif display_target == "traffic_light_and_stop_line":
        layers.append(
            {
                "layer_name": "traffic_light_highlight",
                "source": "semantic_segmentation",
                "target_classes": ["traffic_light"],
                "effect": "soft_contour_and_label_anchor",
                "priority": 1,
            }
        )
        layers.append(
            {
                "layer_name": "stop_line_region",
                "source": "future_road_segmentation",
                "target_classes": ["stop_line", "lane_marking"],
                "effect": "subtle_region_overlay",
                "priority": 2,
                "status": "planned_not_available_yet",
            }
        )

    elif display_target == "relevant_vehicle_or_traffic_region":
        layers.append(
            {
                "layer_name": "near_vehicle_highlight",
                "source": "semantic_segmentation",
                "target_classes": ["car", "bus", "truck", "motorcycle", "bicycle"],
                "effect": "soft_contour_and_mask_glow",
                "priority": 1,
            }
        )

    elif display_target == "construction_or_clearance_region":
        layers.append(
            {
                "layer_name": "clearance_region_highlight",
                "source": "semantic_segmentation",
                "target_classes": ["person", "car", "bus", "truck", "motorcycle", "bicycle"],
                "effect": "soft_contour_and_free_space_overlay",
                "priority": 1,
            }
        )

    else:
        layers.append(
            {
                "layer_name": "ego_future_path",
                "source": "alpamayo_trajectory",
                "target_classes": ["future_path"],
                "effect": "thin_future_path_overlay",
                "priority": 1,
            }
        )

    if intensity == "high":
        layers.append(
            {
                "layer_name": "background_salience_reduction",
                "source": "mirage_diminished_reality",
                "target_classes": ["non_causal_background"],
                "effect": "slight_desaturation_or_blur",
                "priority": 3,
            }
        )

    layers.append(
        {
            "layer_name": "explanation_label",
            "source": "alpamayo_gemma_reasoning",
            "target_classes": ["text_label"],
            "effect": "short_label_near_causal_region",
            "priority": 4,
        }
    )

    if depth_summary["depth_supports_explanation"]:
        layers.append(
            {
                "layer_name": "depth_priority_gate",
                "source": "depth_estimation",
                "target_classes": segmentation_summary["detected_target_classes"],
                "effect": "render_only_spatially_relevant_targets",
                "priority": 0,
            }
        )

    layers = sorted(layers, key=lambda item: item["priority"])

    return layers


def choose_label_text(display_plan, gemma_summary, action, display_target):
    gemma_text = gemma_summary.get("recommended_display_text", "")

    if gemma_text:
        return gemma_text

    label_text = display_plan.get("label_text", "")

    if label_text:
        return label_text

    if display_target == "pedestrians_and_crosswalk":
        return "Yielding for pedestrians in the crosswalk"

    if display_target == "traffic_light_and_stop_line":
        return "Stopping for the traffic light"

    if action == "slow_or_yield":
        return "Yielding for the road user ahead"

    if action == "stop":
        return "Stopping for the current traffic situation"

    return "Vehicle behaviour adjusted for the scene"


def build_effect_plan(workflow_state, gemma_payload, segmentation_payload, depth_payload):
    decision = get_decision(workflow_state)
    display_plan = get_display_plan(workflow_state)

    action = decision.get("alpamayo_action", "unknown")
    uncertainty_score = float(decision.get("uncertainty_score", 0.0))
    reasoning = decision.get("alpamayo_reasoning_trace", "")
    scene_cause = decision.get("scene_cause", "unknown")
    display_target = display_plan.get("display_target", "ego_future_path")

    depth_summary = get_depth_summary(depth_payload)
    segmentation_summary = get_segmentation_summary(segmentation_payload)
    gemma_summary = get_gemma_summary(gemma_payload)

    effect_family = choose_effect_family(
        action,
        uncertainty_score,
        depth_summary["spatial_relevance"],
    )

    intensity = choose_intensity(
        action,
        uncertainty_score,
        depth_summary["spatial_relevance"],
    )

    visual_layers = choose_visual_layers(
        display_target,
        effect_family,
        intensity,
        segmentation_summary,
        depth_summary,
    )

    label_text = choose_label_text(
        display_plan,
        gemma_summary,
        action,
        display_target,
    )

    if effect_family == "none":
        explanation_policy = "do_not_render"
    elif intensity == "high":
        explanation_policy = "render_strong_contextual_explanation"
    else:
        explanation_policy = "render_subtle_contextual_explanation"

    return {
        "video_id": VIDEO_ID,
        "segment_start_time_s": SEGMENT_START_TIME_S,
        "workflow_stage": "mirage_effect_planning",
        "explanation_policy": explanation_policy,
        "mirage_effect_family": effect_family,
        "display_target": display_target,
        "display_intensity": intensity,
        "label_text": label_text,
        "alpamayo_action": action,
        "scene_cause": scene_cause,
        "alpamayo_reasoning_trace": reasoning,
        "uncertainty_score": uncertainty_score,
        "segmentation_summary": segmentation_summary,
        "depth_summary": depth_summary,
        "gemma_summary": gemma_summary,
        "visual_layers": visual_layers,
        "renderer_contract": {
            "input_video": PROJECT_ROOT
            + "/alpamayo_outputs/crowd_clips/"
            + VIDEO_ID
            + "_"
            + str(int(SEGMENT_START_TIME_S))
            + "_30s.mp4",
            "output_video": PROJECT_ROOT
            + "/workflow_outputs/final_renders/"
            + VIDEO_ID
            + "_"
            + str(int(SEGMENT_START_TIME_S))
            + "_mirage_preview_roadline_v3.mp4",
            "effect_plan_json": OUTPUT_JSON,
        },
        "notes": [
            "This is a MIRAGE style effect plan, not the full MIRAGE renderer.",
            "The final renderer should use segmentation masks, depth priority, and this effect plan.",
            "Crosswalk and stop line masks require a specialised road scene segmentation model in a later stage.",
        ],
    }


def update_workflow_state(workflow_state, effect_plan):
    workflow_state["current_stage"] = "mirage_effect_planning_complete"

    workflow_state["outputs"]["mirage_effect_plan_json"] = OUTPUT_JSON

    if "mirage" in workflow_state["next_modules"]:
        workflow_state["next_modules"]["mirage"]["status"] = "complete"

    workflow_state["mirage"] = {
        "status": "complete",
        "output_json": OUTPUT_JSON,
        "explanation_policy": effect_plan["explanation_policy"],
        "mirage_effect_family": effect_plan["mirage_effect_family"],
        "display_target": effect_plan["display_target"],
        "display_intensity": effect_plan["display_intensity"],
        "label_text": effect_plan["label_text"],
        "visual_layer_count": len(effect_plan["visual_layers"]),
    }

    write_json(STATE_JSON, workflow_state)


def print_summary(effect_plan):
    print("")
    print("MIRAGE effect planner")
    print("=====================")
    print("explanation_policy:", effect_plan["explanation_policy"])
    print("mirage_effect_family:", effect_plan["mirage_effect_family"])
    print("display_target:", effect_plan["display_target"])
    print("display_intensity:", effect_plan["display_intensity"])
    print("label_text:", effect_plan["label_text"])
    print("visual_layer_count:", len(effect_plan["visual_layers"]))
    print("")
    print("Visual layers:")
    for layer in effect_plan["visual_layers"]:
        print("  -", layer["layer_name"], "|", layer["effect"])
    print("")
    print("Output JSON:", OUTPUT_JSON)
    print("Updated state:", STATE_JSON)


def main():
    create_dirs()

    workflow_state = read_json(STATE_JSON)
    segmentation_payload = read_json(SEGMENTATION_JSON)
    depth_payload = read_json(DEPTH_JSON)
    gemma_payload = read_json_optional(GEMMA_JSON)

    effect_plan = build_effect_plan(
        workflow_state,
        gemma_payload,
        segmentation_payload,
        depth_payload,
    )

    write_json(OUTPUT_JSON, effect_plan)
    update_workflow_state(workflow_state, effect_plan)

    print_summary(effect_plan)


if __name__ == "__main__":
    main()
