r"""Simulated live Alpamayo planner stream (per-timestep output).

The real Alpamayo planner runs continuously and emits per-timestep reasoning
about why the vehicle is acting. That live model is not available here, so this
derives a faithful stand-in from the scene itself: at each window it checks
whether a road user is actually in the ego lane directly ahead (via YOLO) and
emits the kind of action / reasoning a running planner would produce. The Gemma
gate then re-checks this stream to decide whether to explain.

Run:
    python alpamayo_stream.py <clip.mp4> <stream_out.json> [stride_s]
"""

import sys

import cv2
import numpy as np
from ultralytics import YOLO

import final_preview_renderer as R
from pipeline_common import write_json


STRIDE_S = 4.0


def ego_lane_polygon(width, height):
    """Trapezoid approximating the vehicle's own lane ahead, to the vanishing point."""
    top_y = R.HORIZON_V + 55.0
    return np.array([
        [0.30 * width, height],
        [0.70 * width, height],
        [R.VANISH_U + 55.0, top_y],
        [R.VANISH_U - 55.0, top_y],
    ], dtype=np.int32)


def in_path(box, polygon):
    x1, y1, x2, y2 = box
    foot = (float((x1 + x2) / 2.0), float(y2))
    return cv2.pointPolygonTest(polygon, foot, False) >= 0


def planner_output(in_path_classes):
    """Compose an Alpamayo-style planner output from what is in the ego lane."""
    if not in_path_classes:
        return {
            "alpamayo_action": "continue",
            "alpamayo_reasoning_trace": "Proceeding along the lane; the road directly ahead is clear of road users.",
            "scene_cause": "clear_driving",
            "uncertainty_score": 0.12,
        }
    if "person" in in_path_classes:
        kind = "pedestrian"
    elif any(c in ("bike", "moto") for c in in_path_classes):
        kind = "cyclist"
    else:
        kind = "vehicle"
    return {
        "alpamayo_action": "slow_or_yield",
        "alpamayo_reasoning_trace": (
            "Yielding for a %s that has moved into the vehicle's own driving lane "
            "directly ahead." % kind
        ),
        "scene_cause": "road_user_in_path",
        "uncertainty_score": 0.7,
    }


def stream_windows(clip_video, stride_s=STRIDE_S):
    capture = cv2.VideoCapture(clip_video)
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps else 0.0

    model = YOLO(R.MODEL_NAME)
    polygon = None
    windows = []
    t = stride_s / 2.0
    while t < duration:
        frame_index = int(round(t * fps))
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = capture.read()
        if not ok:
            break
        if polygon is None:
            polygon = ego_lane_polygon(frame.shape[1], frame.shape[0])

        result = model.predict(
            frame, imgsz=R.IMAGE_SIZE, conf=R.CONFIDENCE_THRESHOLD,
            classes=R.OCCLUDER_CLASS_IDS, verbose=False,
        )[0]

        in_path_classes = []
        if result.boxes is not None and result.boxes.xyxy is not None:
            xyxy = result.boxes.xyxy.cpu().numpy()
            cls = result.boxes.cls.cpu().numpy()
            for i in range(len(xyxy)):
                box = [int(v) for v in xyxy[i]]
                if in_path(box, polygon):
                    in_path_classes.append(R.class_display_name(int(cls[i])))

        output = planner_output(in_path_classes)
        window = {
            "t_center": round(t, 2),
            "t_start": round(t - stride_s / 2.0, 2),
            "t_end": round(t + stride_s / 2.0, 2),
            "frame_index": frame_index,
            "road_users_in_path": in_path_classes,
        }
        window.update(output)
        windows.append(window)
        print("t=%6.1fs -> %-11s | in_path=%s" % (t, output["alpamayo_action"], in_path_classes))
        t += stride_s

    capture.release()
    return {
        "clip_video": clip_video,
        "fps": fps,
        "frame_count": frame_count,
        "stride_s": stride_s,
        "windows": windows,
    }


def main():
    clip = sys.argv[1]
    out = sys.argv[2]
    stride = float(sys.argv[3]) if len(sys.argv) > 3 else STRIDE_S
    stream = stream_windows(clip, stride_s=stride)
    write_json(out, stream)
    n = len(stream["windows"])
    yielding = sum(w["alpamayo_action"] == "slow_or_yield" for w in stream["windows"])
    print("\nwindows: %d | yielding: %d | clear: %d" % (n, yielding, n - yielding))
    print("saved:", out)


if __name__ == "__main__":
    main()
