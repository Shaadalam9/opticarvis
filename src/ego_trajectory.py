r"""Future frame visual odometry for the OptiCarVis ribbon.

This reconstructs the ego vehicle path from a pre recorded dashcam clip, so the
future path ribbon can follow a real turn even where there is no plan and no lane
lines, for example a 90 degree turn through an intersection.

The video is pre recorded, so the car actual future path is visible in the next
few seconds of frames. This module recovers it with lightweight planar visual
odometry using the same flat ground pinhole calibration as the renderer. Motion
is estimated from ground plane features whose distance is known from the
calibration:

1. Forward distance per frame
   A road point at image row v is at distance d = f * H / (v - horizon). Its
   vertical motion over one frame maps to a new distance, and the difference is
   how far the car advanced.

2. Yaw per frame
   A ground point horizontal motion has a translational part and a rotational
   part. The translational part is predictable once the forward step is known.
   Subtracting it leaves the yaw component.

Sign convention:
    Internally, +yaw and +Y are left. The renderer expects right positive lateral
    values, so future_trajectories negates the lateral output before saving.

Run with explicit paths:
    python ego_trajectory.py <clip.mp4> <out_trajectory.json> [lookahead_m]

Run for the current pipeline_common job:
    python ego_trajectory.py
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


# Flat ground pinhole calibration. Keep these aligned with
# final_preview_renderer.py defaults unless you intentionally retune both.
FOCAL_PX = 1000.0
CAM_HEIGHT_M = 1.30
HORIZON_V = 448.0
VANISH_U = 636.0

# +1 means a positive rotational image shift is treated as a left turn.
YAW_SIGN = 1.0

GROUND_BAND = (520, 690)
GROUND_MIN_ROWS = 60.0
MAX_STEP_M = 2.0
STEP_DECAY = 0.90

LOOKAHEAD_M = 30.0
LOOKAHEAD_MAX_S = 20.0
DEFAULT_STEP_M = 0.0
MIN_FORWARD_M = 0.2
STRIDE_M = 1.0

_LK = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)


def gray_frame(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def features(gray, y0, y1, n=400):
    mask = np.zeros_like(gray)
    mask[y0:y1, :] = 255

    return cv2.goodFeaturesToTrack(gray, n, 0.01, 7, mask=mask)


def estimate_motion(prev_gray, cur_gray, prev_step):
    """Return d_yaw_rad, d_forward_m, ok between two consecutive frames."""
    points = features(prev_gray, *GROUND_BAND)

    if points is None or len(points) <= 8:
        return 0.0, prev_step * STEP_DECAY, False

    next_points, status, _error = cv2.calcOpticalFlowPyrLK(
        prev_gray,
        cur_gray,
        points,
        None,
        **_LK,
    )

    good = status.ravel() == 1

    if good.sum() <= 6:
        return 0.0, prev_step * STEP_DECAY, False

    u0 = points[good][:, 0, 0].astype(np.float64)
    v0 = points[good][:, 0, 1].astype(np.float64)
    u1 = next_points[good][:, 0, 0].astype(np.float64)
    v1 = next_points[good][:, 0, 1].astype(np.float64)

    usable = (v0 > HORIZON_V + GROUND_MIN_ROWS) & (v1 > HORIZON_V + GROUND_MIN_ROWS)

    if usable.sum() <= 6:
        return 0.0, prev_step * STEP_DECAY, False

    u0 = u0[usable]
    v0 = v0[usable]
    u1 = u1[usable]
    v1 = v1[usable]

    depth0 = FOCAL_PX * CAM_HEIGHT_M / (v0 - HORIZON_V)
    depth1 = FOCAL_PX * CAM_HEIGHT_M / (v1 - HORIZON_V)

    steps = depth0 - depth1
    plausible = np.abs(steps) < MAX_STEP_M

    if plausible.sum() <= 6:
        return 0.0, prev_step * STEP_DECAY, False

    d_forward = float(np.median(steps[plausible]))

    if d_forward < 0.0:
        d_forward = 0.0

    translation = (u0 - VANISH_U) * d_forward / np.maximum(depth0, 1e-3)
    rotation = (u1 - u0) - translation
    d_yaw = YAW_SIGN * float(np.median(rotation)) / FOCAL_PX

    return d_yaw, d_forward, True


def reconstruct_path(clip_video):
    """Integrate per frame motion into a top down ego path."""
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

    ok, previous_frame = capture.read()

    if not ok:
        capture.release()
        return fps, np.zeros(1), np.zeros(1), np.zeros(1)

    prev_gray = gray_frame(previous_frame)
    yaws = [0.0]
    steps = [0.0]
    valid_flags = [False]
    prev_step = DEFAULT_STEP_M

    while True:
        ok, current_frame = capture.read()

        if not ok:
            break

        cur_gray = gray_frame(current_frame)
        d_yaw, d_forward, valid = estimate_motion(prev_gray, cur_gray, prev_step)

        yaws.append(d_yaw)
        steps.append(d_forward)
        valid_flags.append(bool(valid))

        prev_step = d_forward
        prev_gray = cur_gray

    capture.release()

    psi = np.cumsum(np.asarray(yaws))
    step = np.asarray(steps)
    x = np.cumsum(step * np.cos(psi))
    y = np.cumsum(step * np.sin(psi))

    return fps, x, y, psi, valid_flags


def future_trajectories(
    x,
    y,
    psi,
    fps,
    lookahead_m=LOOKAHEAD_M,
    max_s=LOOKAHEAD_MAX_S,
    stride_m=STRIDE_M,
):
    """Return per frame future path in that frame local coordinates."""
    n = len(x)
    frame_cap = int(round(max_s * fps))
    output = []

    for i in range(n):
        j_end = min(i + frame_cap, n - 1)
        cos_i = np.cos(-psi[i])
        sin_i = np.sin(-psi[i])

        points = []
        last_fwd = None
        last_lat = None

        for j in range(i, j_end + 1):
            dx = x[j] - x[i]
            dy = y[j] - y[i]

            fwd = cos_i * dx - sin_i * dy
            lat_left = sin_i * dx + cos_i * dy
            lat = -lat_left

            if fwd < MIN_FORWARD_M:
                continue

            if last_fwd is not None:
                if fwd <= last_fwd + 1e-3:
                    continue

                arc_step = float(np.hypot(fwd - last_fwd, lat - last_lat))

                if arc_step < stride_m:
                    continue

            last_fwd = fwd
            last_lat = lat

            points.append([round(float(fwd), 3), round(float(lat), 3)])

            if last_fwd >= lookahead_m:
                break

        output.append(points)

    return output


def default_output_json():
    return workflow_path("ego_trajectory", segment_tag() + "_ego_trajectory.json")


def parse_args(argv):
    if len(argv) == 1:
        return {
            "clip": CLIP_VIDEO,
            "output": default_output_json(),
            "lookahead_m": LOOKAHEAD_M,
        }

    if len(argv) in [3, 4]:
        return {
            "clip": normalise_path(argv[1]),
            "output": normalise_path(argv[2]),
            "lookahead_m": float(argv[3]) if len(argv) == 4 else LOOKAHEAD_M,
        }

    print(__doc__)
    raise SystemExit(2)


def main():
    args = parse_args(sys.argv)

    fps, x, y, psi, valid_flags = reconstruct_path(args["clip"])

    if not (fps and fps > 1.0):
        print("Refusing to continue: video reports fps=%r." % fps)
        raise SystemExit(1)

    trajectories = future_trajectories(
        x,
        y,
        psi,
        fps,
        lookahead_m=args["lookahead_m"],
    )

    payload = {
        "clip_video": args["clip"],
        "fps": fps,
        "frame_count": len(x),
        "lookahead_m": args["lookahead_m"],
        "lookahead_max_s": LOOKAHEAD_MAX_S,
        "focal_px": FOCAL_PX,
        "cam_height_m": CAM_HEIGHT_M,
        "horizon_v": HORIZON_V,
        "vanish_u": VANISH_U,
        "lateral_convention": "right_positive",
        "valid_motion_frames": int(sum(valid_flags)),
        "future_trajectories": trajectories,
    }

    ensure_dir(os.path.dirname(args["output"]))
    write_json(args["output"], payload)

    if len(x) > 1:
        speed_kmh = float(np.median(np.hypot(np.diff(x), np.diff(y)))) * fps * 3.6
    else:
        speed_kmh = 0.0

    non_empty = sum(1 for path in trajectories if path)

    print("")
    print("Ego trajectory complete")
    print("=======================")
    print("clip:", args["clip"])
    print("frames:", len(x))
    print("median_speed_kmh: %.1f" % speed_kmh)
    print("total_heading_deg: %+.1f (+ = left)" % np.degrees(psi[-1]))
    print("valid_motion_frames:", int(sum(valid_flags)))
    print("non_empty_future_paths: %d/%d" % (non_empty, len(trajectories)))
    print("saved:", args["output"])


if __name__ == "__main__":
    main()