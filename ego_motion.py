r"""Ego-motion track (pass 1 for the look-ahead ribbon).

Estimates the vehicle's yaw over the clip from image motion: a turn pans the
whole scene horizontally, so the per-frame global horizontal shift (via phase
correlation on downscaled grayscale frames, robust to moving objects) is a good
yaw proxy. The cumulative horizontal pan is saved; the renderer looks a few
seconds ahead in it to bend the ribbon into an upcoming turn.

Run:
    python ego_motion.py <clip.mp4> <ego_track_out.json>
"""

import sys

import cv2
import numpy as np

from pipeline_common import write_json


PROC_W = 320
PROC_H = 180


def _gray(frame):
    small = cv2.resize(frame, (PROC_W, PROC_H))
    return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)


def ego_yaw_track(clip_video):
    capture = cv2.VideoCapture(clip_video)
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    scale = 1280.0 / PROC_W   # report pan in full-resolution pixels

    ok, frame = capture.read()
    if not ok:
        capture.release()
        return {"clip_video": clip_video, "fps": fps, "frame_count": frame_count, "cum_pan_px": [0.0]}

    prev = _gray(frame)
    cumulative = 0.0
    series = [0.0]
    window = cv2.createHanningWindow((PROC_W, PROC_H), cv2.CV_32F)

    while True:
        ok, frame = capture.read()
        if not ok:
            break
        current = _gray(frame)
        (dx, _dy), _response = cv2.phaseCorrelate(prev, current, window)
        cumulative += dx * scale
        series.append(cumulative)
        prev = current

    capture.release()
    return {
        "clip_video": clip_video,
        "fps": fps,
        "frame_count": frame_count,
        "cum_pan_px": series,
    }


def main():
    clip = sys.argv[1]
    out = sys.argv[2]
    track = ego_yaw_track(clip)
    write_json(out, track)
    series = track["cum_pan_px"]
    span = (max(series) - min(series)) if series else 0.0
    print("frames: %d | fps: %.1f | cumulative-pan span: %.0f px" % (
        track["frame_count"], track["fps"], span))
    print("saved:", out)


if __name__ == "__main__":
    main()
