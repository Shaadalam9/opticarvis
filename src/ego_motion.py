r"""Ego motion track for the look ahead ribbon.

Estimates the vehicle yaw over the clip from image motion. A turn pans the
whole scene horizontally, so the per frame global horizontal shift from phase
correlation on downscaled grayscale frames gives a simple yaw proxy.

The cumulative horizontal pan is saved. The renderer can look a few seconds
ahead in it to bend the ribbon into an upcoming turn when
OPTICARVIS_EGO_LOOKAHEAD=1.

Run with explicit paths:
    python ego_motion.py <clip.mp4> <ego_track_out.json>

Run for the current pipeline_common job:
    python ego_motion.py
"""

import os
import sys

import cv2
import numpy as np

from pipeline_common import (
    CLIP_VIDEO,
    ensure_dir,
    normalise_path,
    segment_tag,
    workflow_path,
    write_json,
)


PROC_W = 320
PROC_H = 180


def gray_frame(frame):
    small = cv2.resize(frame, (PROC_W, PROC_H))
    return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)


def ego_yaw_track(clip_video):
    if not os.path.isfile(clip_video):
        print("Missing clip video:")
        print(clip_video)
        raise SystemExit(1)

    capture = cv2.VideoCapture(clip_video)

    if not capture.isOpened():
        print("Could not open clip video:")
        print(clip_video)
        raise SystemExit(1)

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    ok, frame = capture.read()

    if not ok:
        capture.release()
        return {
            "clip_video": clip_video,
            "fps": fps,
            "frame_count": frame_count,
            "cum_pan_px": [0.0],
        }

    width = int(frame.shape[1])
    scale = float(width) / float(PROC_W)

    previous = gray_frame(frame)
    cumulative = 0.0
    series = [0.0]
    deltas = []

    window = cv2.createHanningWindow((PROC_W, PROC_H), cv2.CV_32F)

    while True:
        ok, frame = capture.read()

        if not ok:
            break

        current = gray_frame(frame)
        shift, response = cv2.phaseCorrelate(previous, current, window)
        dx = float(shift[0])

        delta = dx * scale
        cumulative += delta

        deltas.append(
            {
                "dx_px": round(delta, 3),
                "phase_response": round(float(response), 6),
            }
        )
        series.append(round(cumulative, 3))

        previous = current

    capture.release()

    return {
        "clip_video": clip_video,
        "fps": fps,
        "frame_count": frame_count,
        "processed_frame_count": len(series),
        "proc_width": PROC_W,
        "proc_height": PROC_H,
        "reported_width_scale": round(scale, 6),
        "cum_pan_px": series,
        "frame_deltas": deltas,
    }


def default_output_json():
    return workflow_path("ego_motion", segment_tag() + "_ego_motion.json")


def parse_args(argv):
    if len(argv) == 1:
        return {
            "clip": CLIP_VIDEO,
            "output": default_output_json(),
        }

    if len(argv) == 3:
        return {
            "clip": normalise_path(argv[1]),
            "output": normalise_path(argv[2]),
        }

    print(__doc__)
    raise SystemExit(2)


def main():
    args = parse_args(sys.argv)

    ensure_dir(os.path.dirname(args["output"]))

    track = ego_yaw_track(args["clip"])
    write_json(args["output"], track)

    series = track["cum_pan_px"]
    span = (max(series) - min(series)) if series else 0.0

    print("")
    print("Ego motion track complete")
    print("=========================")
    print("clip:", args["clip"])
    print("frames:", track["frame_count"])
    print("processed_frames:", track["processed_frame_count"])
    print("fps: %.1f" % track["fps"])
    print("cumulative_pan_span_px: %.0f" % span)
    print("saved:", args["output"])


if __name__ == "__main__":
    main()