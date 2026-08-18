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
# The calibration below is absolute pixels at this reference resolution; every
# frame is resized to it in gray_frame(). Keep in step with
# final_preview_renderer.CALIB_REF_WIDTH / CALIB_REF_HEIGHT.
CALIB_REF_WIDTH = 1280
CALIB_REF_HEIGHT = 720

FOCAL_PX = 1000.0
CAM_HEIGHT_M = 1.30
HORIZON_V = 448.0
VANISH_U = 636.0

# +1 means a positive rotational image shift is treated as a left turn.
YAW_SIGN = 1.0

# Metric forward window, not fixed rows. Hardcoded rows silently encode the dev
# rig's horizon (row -> distance is d = f*H/(v - horizon)); on a rig whose
# horizon sits elsewhere the same rows sample a different slab, and on a clip
# with a visible bonnet they sampled the bonnet itself -- a plane rigid with the
# camera. The odometry then fitted the bonnet and reported a car doing 23-43
# km/h as stationary (measured on Brazzaville, D1FkMQWpMoM). Resolved per clip
# in ground_band(), after the calibration override is applied.
GROUND_BAND_NEAR_M = 6.0
GROUND_BAND_FAR_M = 30.0
GROUND_BAND_FALLBACK = (520, 690)
GROUND_BAND = GROUND_BAND_FALLBACK
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
    """Greyscale at the resolution the flat-ground calibration assumes.

    FOCAL_PX, HORIZON_V, VANISH_U and GROUND_BAND are absolute pixels at
    CALIB_REF_WIDTH x CALIB_REF_HEIGHT. Most clips in this project are
    3840x2160, where rows 520-690 land in the sky: every tracked feature then
    fails the (v > HORIZON_V + GROUND_MIN_ROWS) test or returns a nonsense
    depth, the median forward step collapses to ~0, and the clip reads as a
    STATIONARY vehicle rather than as an error. Resizing here keeps the
    calibration exact and makes the metric output resolution independent --
    depth is f*H/(v - horizon), so working in calibration pixels throughout
    yields the same metres at any input size.

    This is an exact no-op for clips already at the reference resolution.
    """
    if (frame.shape[1], frame.shape[0]) != (CALIB_REF_WIDTH, CALIB_REF_HEIGHT):
        frame = cv2.resize(
            frame,
            (CALIB_REF_WIDTH, CALIB_REF_HEIGHT),
            interpolation=cv2.INTER_AREA,
        )

    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def ground_band():
    """Rows covering GROUND_BAND_NEAR_M..GROUND_BAND_FAR_M for this rig.

    Kept in step with future_anchor.ground_band_rows: the anchors and the
    arclength tags they carry must agree about which plane is the ground.
    """
    reach = FOCAL_PX * CAM_HEIGHT_M
    top = int(round(HORIZON_V + reach / GROUND_BAND_FAR_M))
    bottom = int(round(HORIZON_V + reach / GROUND_BAND_NEAR_M))
    top = max(top, int(round(HORIZON_V)) + 4)
    bottom = min(bottom, CALIB_REF_HEIGHT - 6)

    if bottom - top < 40:
        return GROUND_BAND_FALLBACK

    return (top, bottom)


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


# Heading validation against scene phase correlation (ENGINEERING.md 3a: the
# scene sliding right means the camera yaws left). The VO yaw estimate depends
# on the flat-ground calibration matching the camera; on a rig it does not
# match, the parallax subtraction leaves a one-sided residual and the heading
# acquires a steady drift -- real turns still read correctly, but the future
# path curls sideways on straights, and the ribbon bends where the car is not
# going. Measured on Fvt6rD9tt1c_22: +225 deg accumulated over 30 s of mostly
# straight road, +70 deg in one window the scene pans the other way.
VALIDATION_WINDOW_S = 4.0
VALIDATION_PAN_STRONG_PX = 60.0   # at the 320x180 correlation scale
VALIDATION_PAN_QUIET_PX = 30.0
VALIDATION_DRIFT_DEG = 10.0       # heading accumulated in a pan-quiet window
VALIDATION_TURN_DEG = 5.0


def scene_pan_profile(clip_path):
    """Per-frame horizontal scene shift via phase correlation, upper half only
    (skyline and buildings: mostly rotational, little ground parallax)."""
    capture = cv2.VideoCapture(clip_path)
    previous = None
    pans = []

    try:
        while True:
            ok, frame = capture.read()

            if not ok:
                break

            gray = cv2.resize(
                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (320, 180)
            ).astype(np.float32)[:90, :]

            if previous is not None:
                (dx, _dy), _response = cv2.phaseCorrelate(previous, gray)
                pans.append(float(dx))

            previous = gray
    finally:
        capture.release()

    return np.asarray(pans, dtype=np.float64)


def validate_heading_against_pan(psi, pans, fps):
    """Cross-check the integrated VO heading against the scene pan, per window.

    Returns (valid, reasons). Conservative on purpose: a rejected track means a
    straight in-lane ribbon, and a ribbon that bends where the car is not going
    is worse than one that does not bend.
    """
    window = max(1, int(round(VALIDATION_WINDOW_S * fps)))
    psi_deg = np.degrees(np.asarray(psi, dtype=np.float64))
    reasons = []
    drift_windows = 0

    for start in range(0, len(pans) - window + 1, window):
        pan_sum = float(pans[start:start + window].sum())
        end = min(start + window, len(psi_deg) - 1)
        heading_delta = float(psi_deg[end] - psi_deg[start])

        # Sign convention: scene right (+pan) = yaw left = +psi (vehicle frame).
        if abs(pan_sum) >= VALIDATION_PAN_STRONG_PX and abs(heading_delta) >= VALIDATION_TURN_DEG:
            if np.sign(pan_sum) != np.sign(heading_delta):
                reasons.append(
                    "t=%.1f-%.1fs: scene pans %+.0f px but VO heading moves %+.1f deg"
                    % (start / fps, (start + window) / fps, pan_sum, heading_delta)
                )
        elif abs(pan_sum) <= VALIDATION_PAN_QUIET_PX and abs(heading_delta) >= VALIDATION_DRIFT_DEG:
            drift_windows += 1
            reasons.append(
                "t=%.1f-%.1fs: scene is quiet (%+.0f px) but VO heading drifts %+.1f deg"
                % (start / fps, (start + window) / fps, pan_sum, heading_delta)
            )

    contradictions = len(reasons) - drift_windows
    valid = contradictions == 0 and drift_windows <= 1

    return valid, reasons


def apply_calibration_overrides():
    """Adopt the per-clip camera calibration (auto_calibrate.py), if present.

    The yaw estimate subtracts translational parallax predicted around
    VANISH_U; with the default constants on a rig they do not match, the
    residual acquires a steady drift (+225 deg over 30 s measured) that the
    heading validation then rejects. The calibration file is shared with the
    renderer and the Alpamayo2 wrapper, so all three consumers agree.
    """
    global VANISH_U, HORIZON_V, FOCAL_PX, CAM_HEIGHT_M, GROUND_BAND

    path = workflow_path("calibration", segment_tag() + "_camera_calibration.json")

    if not os.path.isfile(path):
        return

    import json

    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    VANISH_U = float(data.get("VANISH_U", VANISH_U))
    HORIZON_V = float(data.get("HORIZON_V", HORIZON_V))
    FOCAL_PX = float(data.get("CAM_FOCAL_PX", FOCAL_PX))
    CAM_HEIGHT_M = float(data.get("CAM_HEIGHT_M", CAM_HEIGHT_M))
    print("Loaded camera calibration: %s (VANISH_U %.1f, HORIZON_V %.1f)"
          % (path, VANISH_U, HORIZON_V))
    GROUND_BAND = ground_band()
    print("Ground band rows: %d-%d (%.0f-%.0f m ahead)"
          % (GROUND_BAND[0], GROUND_BAND[1], GROUND_BAND_FAR_M, GROUND_BAND_NEAR_M))


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

    apply_calibration_overrides()

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

    pans = scene_pan_profile(args["clip"])
    heading_valid, reasons = validate_heading_against_pan(psi, pans, fps)

    if not heading_valid:
        print("")
        print("VO heading REJECTED by the scene-pan cross-check "
              "(ENGINEERING.md 3a); the ribbon will stay straight in the ego "
              "lane rather than bend where the car is not going:")

        for reason in reasons:
            print("  " + reason)

        trajectories = [[] for _ in trajectories]

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
        # Absolute ego pose per frame (right-positive, like everything else in
        # this file): the renderer anchors the planner's trajectory to the
        # world where it was planned and advances it by the car's ACTUAL
        # motion. Only trustworthy when heading_validation.valid -- an
        # anchoring built on a biased heading re-detaches the ribbon.
        "ego_pose_right_positive": [
            [round(float(px), 3), round(-float(py), 3), round(-float(ppsi), 5)]
            for px, py, ppsi in zip(x, y, psi)
        ],
        "heading_validation": {
            "valid": bool(heading_valid),
            "method": "scene_pan_cross_check",
            "reasons": reasons,
        },
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
