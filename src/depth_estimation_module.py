r"""Stage 4 depth estimation and spatial relevance module for the OptiCarVis workflow.

Purpose:
    Read semantic segmentation outputs.
    Estimate spatial relevance of detected objects.
    Save depth and proximity reasoning as JSON.
    Update the workflow state for the MIRAGE visual effect planner.

Current implementation:
    Uses a lightweight monocular depth proxy from segmentation boxes:
        larger object height -> likely closer
        lower image position -> likely closer to ego path
    This keeps the workflow running without downloading a depth model.

Later upgrade:
    Replace the proxy with a true monocular depth model such as Depth Anything,
    Metric3D, ZoeDepth, or another selected model.

Run from the repo root, in the project venv:
    python src/depth_estimation_module.py
"""

import os

import cv2
import numpy as np

from pipeline_common import (
    VIDEO_ID,
    SEGMENT_START_TIME_S,
    STATE_JSON,
    read_json,
    write_json,
    clamp,
    ensure_dir,
    segment_tag,
    workflow_path,
)

# Use the real monocular depth model (Depth Anything V2 via scene_models) instead
# of the bbox proxy. Falls back to the proxy automatically if it cannot load.
USE_REAL_DEPTH = True
_real_depth_ok = None

SEGMENTATION_JSON = workflow_path("segmentation", segment_tag() + "_segmentation.json")

OUTPUT_DIR = workflow_path("depth")

OUTPUT_JSON = workflow_path("depth", segment_tag() + "_depth.json")


def create_dirs():
    ensure_dir(OUTPUT_DIR)


def get_requested_target_classes(segmentation_payload):
    targets = segmentation_payload.get("segmentation_targets", {})
    target_classes = targets.get("target_classes", [])

    if not target_classes:
        return []

    return target_classes


def get_decision_display_target(workflow_state):
    decision = workflow_state.get("decision", {})
    display_plan = decision.get("display_plan", {})
    return display_plan.get("display_target", "")


def real_depth_available():
    """True if the Depth Anything model loads; cached, with graceful fallback."""
    global _real_depth_ok
    if _real_depth_ok is None:
        if not USE_REAL_DEPTH:
            _real_depth_ok = False
            return False
        try:
            import scene_models

            scene_models.load_depth_model()
            _real_depth_ok = True
        except Exception as error:
            print("Real depth unavailable (%s); using bbox proxy." % error)
            _real_depth_ok = False
    return _real_depth_ok


def frame_depth_map(frame_payload):
    """Normalised inverse-depth map (0..1, higher = nearer) for a sampled frame."""
    frame_file = frame_payload.get("frame_file")
    if not frame_file or not os.path.isfile(frame_file):
        return None
    image = cv2.imread(frame_file)
    if image is None:
        return None
    import scene_models

    depth = scene_models.depth_map(image)
    return cv2.normalize(depth, None, 0.0, 1.0, cv2.NORM_MINMAX)


def compute_object_depth_real(obj, depth_norm):
    """Per-object proximity from the real depth map (median inverse depth in box)."""
    box = obj.get("box", {})
    height, width = depth_norm.shape

    x1 = int(clamp(round(float(box.get("x1", 0.0))), 0, width - 1))
    y1 = int(clamp(round(float(box.get("y1", 0.0))), 0, height - 1))
    x2 = int(clamp(round(float(box.get("x2", 0.0))), 1, width))
    y2 = int(clamp(round(float(box.get("y2", 0.0))), 1, height))

    if x2 <= x1 or y2 <= y1:
        return None

    region = depth_norm[y1:y2, x1:x2]
    if region.size == 0:
        return None

    proximity_score = round(float(np.median(region)), 3)

    if proximity_score >= 0.70:
        proximity_level = "near"
    elif proximity_score >= 0.40:
        proximity_level = "medium"
    else:
        proximity_level = "far"

    return {
        "class_name": obj.get("class_name", ""),
        "confidence": obj.get("confidence", 0.0),
        "box": box,
        "mask_area_ratio": obj.get("mask_area_ratio", 0.0),
        # Kept for schema compatibility with the proxy (not used by real depth).
        "size_score": None,
        "lower_image_score": None,
        "area_score": None,
        "proximity_score": proximity_score,
        "proximity_level": proximity_level,
        "pseudo_depth_rank": round(1.0 - proximity_score, 3),
        "depth_method": "depth_anything_v2",
    }


def compute_depth_proxy_for_object(obj):
    box = obj.get("box", {})

    y_center_norm = float(box.get("y_center_norm", 0.0))
    width_norm = float(box.get("width_norm", 0.0))
    height_norm = float(box.get("height_norm", 0.0))

    object_area_norm = width_norm * height_norm

    size_score = clamp(height_norm / 0.45, 0.0, 1.0)
    lower_image_score = clamp((y_center_norm - 0.25) / 0.75, 0.0, 1.0)
    area_score = clamp(object_area_norm / 0.10, 0.0, 1.0)

    proximity_score = (
        0.55 * size_score
        + 0.30 * lower_image_score
        + 0.15 * area_score
    )

    proximity_score = round(clamp(proximity_score, 0.0, 1.0), 3)

    if proximity_score >= 0.70:
        proximity_level = "near"
    elif proximity_score >= 0.40:
        proximity_level = "medium"
    else:
        proximity_level = "far"

    pseudo_depth_rank = round(1.0 - proximity_score, 3)

    return {
        "class_name": obj.get("class_name", ""),
        "confidence": obj.get("confidence", 0.0),
        "box": box,
        "mask_area_ratio": obj.get("mask_area_ratio", 0.0),
        "size_score": round(size_score, 3),
        "lower_image_score": round(lower_image_score, 3),
        "area_score": round(area_score, 3),
        "proximity_score": proximity_score,
        "proximity_level": proximity_level,
        "pseudo_depth_rank": pseudo_depth_rank,
        "depth_method": "bbox_spatial_proxy",
    }


def analyse_frame_depth(frame_payload, target_classes):
    objects = frame_payload.get("objects", [])

    depth_norm = None
    if real_depth_available():
        depth_norm = frame_depth_map(frame_payload)

    analysed_objects = []

    for obj in objects:
        class_name = obj.get("class_name", "")

        is_target_class = class_name in target_classes

        depth_obj = None
        if depth_norm is not None:
            depth_obj = compute_object_depth_real(obj, depth_norm)
        if depth_obj is None:
            depth_obj = compute_depth_proxy_for_object(obj)

        depth_obj["is_target_class"] = is_target_class
        depth_obj["frame_index"] = frame_payload.get("frame_index")
        depth_obj["frame_time_s"] = frame_payload.get("frame_time_s")

        analysed_objects.append(depth_obj)

    analysed_objects = sorted(
        analysed_objects,
        key=lambda item: item["proximity_score"],
        reverse=True,
    )

    target_objects = [obj for obj in analysed_objects if obj["is_target_class"]]

    if target_objects:
        closest_target = target_objects[0]
    else:
        closest_target = None

    return {
        "frame_index": frame_payload.get("frame_index"),
        "frame_time_s": frame_payload.get("frame_time_s"),
        "objects": analysed_objects,
        "closest_target_object": closest_target,
    }


def aggregate_depth_results(frame_depth_results, target_classes):
    all_target_objects = []

    for frame_result in frame_depth_results:
        for obj in frame_result.get("objects", []):
            if obj.get("is_target_class"):
                all_target_objects.append(obj)

    all_target_objects = sorted(
        all_target_objects,
        key=lambda item: item["proximity_score"],
        reverse=True,
    )

    if all_target_objects:
        top_target = all_target_objects[0]
        max_target_proximity = top_target["proximity_score"]
    else:
        top_target = None
        max_target_proximity = 0.0

    if max_target_proximity >= 0.70:
        spatial_relevance = "high"
    elif max_target_proximity >= 0.40:
        spatial_relevance = "medium"
    elif all_target_objects:
        spatial_relevance = "low"
    else:
        spatial_relevance = "not_detected"

    if spatial_relevance in ["high", "medium"]:
        depth_supports_explanation = True
    else:
        depth_supports_explanation = False

    return {
        "target_classes": target_classes,
        "target_object_count": len(all_target_objects),
        "max_target_proximity_score": round(max_target_proximity, 3),
        "spatial_relevance": spatial_relevance,
        "depth_supports_explanation": depth_supports_explanation,
        "closest_target_object": top_target,
    }


def update_workflow_state(workflow_state, depth_payload):
    workflow_state["current_stage"] = "depth_estimation_complete"

    workflow_state["outputs"]["depth_json"] = OUTPUT_JSON

    if "depth_estimation" in workflow_state["next_modules"]:
        workflow_state["next_modules"]["depth_estimation"]["status"] = "complete"

    if "mirage" in workflow_state["next_modules"]:
        workflow_state["next_modules"]["mirage"]["status"] = "pending"

    workflow_state["depth"] = {
        "status": "complete",
        "output_json": OUTPUT_JSON,
        "depth_method": depth_payload["depth_method"],
        "spatial_relevance": depth_payload["aggregate"]["spatial_relevance"],
        "depth_supports_explanation": depth_payload["aggregate"]["depth_supports_explanation"],
        "max_target_proximity_score": depth_payload["aggregate"]["max_target_proximity_score"],
    }

    write_json(STATE_JSON, workflow_state)


def print_summary(payload):
    aggregate = payload["aggregate"]

    print("")
    print("Depth estimation module")
    print("=======================")
    print("depth_method:", payload["depth_method"])
    print("requested_display_target:", payload["requested_display_target"])
    print("target_classes:", aggregate["target_classes"])
    print("target_object_count:", aggregate["target_object_count"])
    print("max_target_proximity_score:", aggregate["max_target_proximity_score"])
    print("spatial_relevance:", aggregate["spatial_relevance"])
    print("depth_supports_explanation:", aggregate["depth_supports_explanation"])
    print("")
    print("Output JSON:", OUTPUT_JSON)
    print("Updated state:", STATE_JSON)


def main():
    create_dirs()

    workflow_state = read_json(STATE_JSON)
    segmentation_payload = read_json(SEGMENTATION_JSON)

    target_classes = get_requested_target_classes(segmentation_payload)
    requested_display_target = get_decision_display_target(workflow_state)

    frame_depth_results = []

    for frame_payload in segmentation_payload.get("frames", []):
        frame_depth = analyse_frame_depth(frame_payload, target_classes)
        frame_depth_results.append(frame_depth)

    aggregate = aggregate_depth_results(frame_depth_results, target_classes)

    payload = {
        "video_id": VIDEO_ID,
        "segment_start_time_s": SEGMENT_START_TIME_S,
        "workflow_stage": "depth_estimation",
        "depth_method": "depth_anything_v2" if real_depth_available() else "bbox_spatial_proxy",
        "requested_display_target": requested_display_target,
        "segmentation_json": SEGMENTATION_JSON,
        "frames": frame_depth_results,
        "aggregate": aggregate,
        "notes": [
            "Per-object relative depth from Depth Anything V2 (median inverse depth in the box).",
            "Higher proximity_score means nearer. Falls back to the bbox spatial proxy if the model is unavailable.",
        ],
    }

    write_json(OUTPUT_JSON, payload)
    update_workflow_state(workflow_state, payload)

    print_summary(payload)


if __name__ == "__main__":
    main()
