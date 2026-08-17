r"""Stage 3 semantic segmentation module for the OptiCarVis workflow.

Purpose:
    Read the workflow state produced by Stage 1 and Stage 2.
    Run semantic segmentation on representative frames from the CROWD clip.
    Save segmentation JSON for later depth and MIRAGE modules.
    Update the workflow state with segmentation output paths.

Current implementation:
    Uses Ultralytics YOLO segmentation when available.
    It focuses on causal classes useful for the current workflow:
        person, bicycle, car, motorcycle, bus, truck, traffic light.
    For the current Alpamayo decision, the main target is:
        pedestrians_and_crosswalk

Run from the repo root, in the project venv:
    python src/semantic_segmentation_module.py

ultralytics is a declared dependency, so `uv sync --frozen` installs it.
"""

import os

import cv2
import numpy as np

from pipeline_common import (
    VIDEO_ID,
    SEGMENT_START_TIME_S,
    CLIP_VIDEO,
    STATE_JSON,
    YOLO_SEG_MODEL,
    read_json,
    write_json,
    ensure_dir,
    segment_tag,
    workflow_path,
)

OUTPUT_DIR = workflow_path("segmentation")

OUTPUT_JSON = workflow_path("segmentation", segment_tag() + "_segmentation.json")

OUTPUT_PREVIEW_DIR = workflow_path(
    "segmentation",
    segment_tag() + "_preview_frames",
)

# Swap via OPTICARVIS_YOLO_SEG_MODEL; see pipeline_common "Models".
MODEL_NAME = YOLO_SEG_MODEL
IMAGE_SIZE = 1280
CONFIDENCE_THRESHOLD = 0.25

SAMPLE_FRAME_POSITIONS = [0.25, 0.50, 0.75]

COCO_NAMES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    9: "traffic_light",
}

CLASSES_TO_KEEP = sorted(list(COCO_NAMES.keys()))


def create_dirs():
    ensure_dir(OUTPUT_DIR)
    ensure_dir(OUTPUT_PREVIEW_DIR)


def get_clip_metadata(video_file):
    if not os.path.isfile(video_file):
        print("Missing clip video:")
        print(video_file)
        raise SystemExit(1)

    capture = cv2.VideoCapture(video_file)

    if not capture.isOpened():
        print("Could not open clip video:")
        print(video_file)
        raise SystemExit(1)

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    capture.release()

    return {
        "frame_count": frame_count,
        "fps": fps,
        "width": width,
        "height": height,
    }


def extract_sample_frames(video_file, metadata):
    capture = cv2.VideoCapture(video_file)

    frames = []

    for position in SAMPLE_FRAME_POSITIONS:
        frame_index = int((metadata["frame_count"] - 1) * position)
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

        ok, frame = capture.read()

        if not ok:
            print("Could not read frame:", frame_index)
            raise SystemExit(1)

        frame_file = os.path.join(
            OUTPUT_PREVIEW_DIR,
            "frame_" + str(frame_index).zfill(6) + ".jpg",
        )

        cv2.imwrite(frame_file, frame)

        frames.append(
            {
                "frame_index": frame_index,
                "frame_time_s": round(frame_index / metadata["fps"], 3),
                "frame_file": frame_file,
            }
        )

    capture.release()

    return frames


def xyxy_to_dict(xyxy, width, height):
    x1 = float(xyxy[0])
    y1 = float(xyxy[1])
    x2 = float(xyxy[2])
    y2 = float(xyxy[3])

    box_width = max(0.0, x2 - x1)
    box_height = max(0.0, y2 - y1)

    return {
        "x1": round(x1, 3),
        "y1": round(y1, 3),
        "x2": round(x2, 3),
        "y2": round(y2, 3),
        "x_center_norm": round(((x1 + x2) / 2.0) / width, 6),
        "y_center_norm": round(((y1 + y2) / 2.0) / height, 6),
        "width_norm": round(box_width / width, 6),
        "height_norm": round(box_height / height, 6),
    }


def mask_area_ratio(mask_data, width, height):
    if mask_data is None:
        return 0.0

    mask_array = mask_data.astype(np.uint8)
    return round(float(mask_array.sum()) / float(width * height), 6)


def run_yolo_segmentation(sample_frames, metadata):
    from ultralytics import YOLO

    model = YOLO(MODEL_NAME)

    frame_results = []
    object_counts = {}

    for frame_info in sample_frames:
        result_list = model.predict(
            source=frame_info["frame_file"],
            imgsz=IMAGE_SIZE,
            conf=CONFIDENCE_THRESHOLD,
            classes=CLASSES_TO_KEEP,
            verbose=False,
        )

        result = result_list[0]

        frame_objects = []

        if result.boxes is not None:
            boxes_xyxy = result.boxes.xyxy.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()

            masks = None
            if result.masks is not None:
                masks = result.masks.data.cpu().numpy()

            for index in range(len(boxes_xyxy)):
                class_id = int(class_ids[index])
                class_name = COCO_NAMES.get(class_id, str(class_id))

                mask_ratio = 0.0
                if masks is not None and index < len(masks):
                    mask_ratio = mask_area_ratio(
                        masks[index],
                        metadata["width"],
                        metadata["height"],
                    )

                frame_objects.append(
                    {
                        "class_id": class_id,
                        "class_name": class_name,
                        "confidence": round(float(confidences[index]), 4),
                        "box": xyxy_to_dict(
                            boxes_xyxy[index],
                            metadata["width"],
                            metadata["height"],
                        ),
                        "mask_area_ratio": mask_ratio,
                    }
                )

                object_counts[class_name] = object_counts.get(class_name, 0) + 1

        frame_results.append(
            {
                "frame_index": frame_info["frame_index"],
                "frame_time_s": frame_info["frame_time_s"],
                "frame_file": frame_info["frame_file"],
                "objects": frame_objects,
            }
        )

    return frame_results, object_counts


def infer_segmentation_targets(workflow_state, object_counts):
    decision = workflow_state.get("decision", {})
    display_plan = decision.get("display_plan", {})
    display_target = display_plan.get("display_target", "")

    target_classes = []

    if display_target == "pedestrians_and_crosswalk":
        target_classes = ["person"]
    elif display_target == "traffic_light_and_stop_line":
        target_classes = ["traffic_light"]
    elif display_target == "relevant_vehicle_or_traffic_region":
        target_classes = ["car", "bus", "truck", "motorcycle", "bicycle"]
    elif display_target == "construction_or_clearance_region":
        target_classes = ["person", "car", "bus", "truck", "motorcycle", "bicycle"]
    else:
        target_classes = ["person", "car", "bus", "truck", "traffic_light"]

    detected_target_classes = []

    for class_name in target_classes:
        if object_counts.get(class_name, 0) > 0:
            detected_target_classes.append(class_name)

    if detected_target_classes:
        target_status = "target_detected"
    else:
        target_status = "target_not_detected_in_sampled_frames"

    return {
        "requested_display_target": display_target,
        "target_classes": target_classes,
        "detected_target_classes": detected_target_classes,
        "target_status": target_status,
    }


def update_workflow_state(workflow_state, segmentation_payload):
    workflow_state["current_stage"] = "semantic_segmentation_complete"

    workflow_state["outputs"]["segmentation_json"] = OUTPUT_JSON

    if "semantic_segmentation" in workflow_state["next_modules"]:
        workflow_state["next_modules"]["semantic_segmentation"]["status"] = "complete"

    if "depth_estimation" in workflow_state["next_modules"]:
        workflow_state["next_modules"]["depth_estimation"]["status"] = "pending"

    if "mirage" in workflow_state["next_modules"]:
        workflow_state["next_modules"]["mirage"]["status"] = "pending"

    workflow_state["segmentation"] = {
        "status": "complete",
        "output_json": OUTPUT_JSON,
        "target_status": segmentation_payload["segmentation_targets"]["target_status"],
        "detected_target_classes": segmentation_payload["segmentation_targets"]["detected_target_classes"],
    }

    write_json(STATE_JSON, workflow_state)


def print_summary(payload):
    print("")
    print("Semantic segmentation module")
    print("============================")
    print("model:", payload["model"])
    print("clip_video:", payload["clip_video"])
    print("sampled_frames:", len(payload["frames"]))
    print("object_counts:", payload["object_counts"])
    print("requested_display_target:", payload["segmentation_targets"]["requested_display_target"])
    print("target_classes:", payload["segmentation_targets"]["target_classes"])
    print("detected_target_classes:", payload["segmentation_targets"]["detected_target_classes"])
    print("target_status:", payload["segmentation_targets"]["target_status"])
    print("")
    print("Output JSON:", OUTPUT_JSON)
    print("Updated state:", STATE_JSON)


def main():
    create_dirs()

    workflow_state = read_json(STATE_JSON)
    metadata = get_clip_metadata(CLIP_VIDEO)
    sample_frames = extract_sample_frames(CLIP_VIDEO, metadata)

    frame_results, object_counts = run_yolo_segmentation(sample_frames, metadata)
    segmentation_targets = infer_segmentation_targets(workflow_state, object_counts)

    payload = {
        "video_id": VIDEO_ID,
        "segment_start_time_s": SEGMENT_START_TIME_S,
        "workflow_stage": "semantic_segmentation",
        "model": MODEL_NAME,
        "clip_video": CLIP_VIDEO,
        "clip_metadata": metadata,
        "frames": frame_results,
        "object_counts": object_counts,
        "segmentation_targets": segmentation_targets,
        "notes": [
            "This stage grounds the Alpamayo and Gemma reasoning in visual objects or regions.",
            "For crosswalk cases, YOLO segmentation detects people but does not segment the crosswalk itself.",
            "A later specialised road scene segmentation model can add lane, road, "
            "pavement, stop line, and crosswalk masks.",
        ],
    }

    write_json(OUTPUT_JSON, payload)
    update_workflow_state(workflow_state, payload)

    print_summary(payload)


if __name__ == "__main__":
    main()
