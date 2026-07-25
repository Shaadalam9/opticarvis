r"""Future-frame visual odometry: reconstruct the ego's driving path from a given
(pre-recorded) dashcam video, so the future-path ribbon can follow a real turn
even where there is no plan and no lane lines (e.g. a 90 degree turn through an
intersection).

The video is pre-recorded, so the car's actual future path IS in the next few
seconds of frames. We recover it with a lightweight *planar* visual odometry that
uses the same flat-ground pinhole calibration as the renderer. Both motion
components are read from ground-plane features, whose distance is known from the
calibration (d = f*H/(v - horizon)):

  * forward distance per frame - a road point at image row v is at distance d;
    its vertical motion over one frame maps to a new distance, and the difference
    (d0 - d1) is how far the car advanced.
  * yaw (heading change) per frame - a ground point's horizontal motion has two
    parts: a *translational* part that pushes it away from the vanishing column
    as the car advances ((u - VANISH_U) * step / d, predictable once the forward
    step is known) and a *rotational* part (f * dpsi, identical for every point).
    Subtracting the predicted translation before taking the median isolates yaw.
    Reading yaw from raw horizontal flow instead (the naive approach) is
    dominated by translational parallax and fabricates turns on straight roads.

Integrating (yaw, forward) gives a top-down ego path. For each frame we then
express the next LOOKAHEAD_S seconds of that path in the frame's own coordinates
(x_forward, y_lateral) - exactly what project_ground_point expects.

SIGN CONVENTION (do not "simplify" this): internally the path is built in the
usual vehicle frame where +yaw and +Y are LEFT. The renderer's
project_ground_point is RIGHT-positive (LATERAL_SIGN = +1 puts +y_lat at
u > VANISH_U). future_trajectories therefore negates the lateral value on the way
out, so the emitted y_lateral is RIGHT-positive and matches the renderer. Getting
this wrong mirrors every turn - the ribbon points where the car is not driving.

Run:
    python ego_trajectory.py <clip.mp4> <out_trajectory.json> [lookahead_s]
"""

import sys

import cv2
import numpy as np

from pipeline_common import write_json


# Flat-ground pinhole (matches final_preview_renderer's calibration defaults).
FOCAL_PX = 1000.0
CAM_HEIGHT_M = 1.30
HORIZON_V = 448.0
VANISH_U = 636.0

# Yaw sign: +1 means a positive rotational image shift (scene sliding right) is a
# LEFT turn, i.e. +yaw = left, consistent with the +Y = left world frame below.
YAW_SIGN = 1.0

# Ground-flow band (rows). Only rows well below the horizon are used: their depth
# is well conditioned (v - HORIZON_V is not near zero), which both the forward
# step and the translation subtraction depend on.
GROUND_BAND = (520, 690)
GROUND_MIN_ROWS = 60.0     # ignore features closer than this many rows to the horizon
MAX_STEP_M = 2.0           # reject implausible per-frame forward steps
STEP_DECAY = 0.90          # decay (not latch) the forward step when flow is unusable

LOOKAHEAD_S = 4.0          # how far ahead to expose per-frame future path
DEFAULT_STEP_M = 0.0       # start from rest; the first valid flow sets the speed
MIN_FORWARD_M = 0.2        # drop path points at/behind the camera
STRIDE_M = 1.0             # thin path samples by ARC length, not forward distance

_LK = dict(winSize=(21, 21), maxLevel=3,
           criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))


def _gray(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def _features(gray, y0, y1, n=400):
    mask = np.zeros_like(gray)
    mask[y0:y1, :] = 255
    return cv2.goodFeaturesToTrack(gray, n, 0.01, 7, mask=mask)


def estimate_motion(prev_gray, cur_gray, prev_step):
    """Return (d_yaw_rad, d_forward_m, ok) between two consecutive frames.

    ok is False when ground flow was unusable, so the caller can decay rather
    than trust the returned forward step.
    """
    points = _features(prev_gray, *GROUND_BAND)
    if points is None or len(points) <= 8:
        return 0.0, prev_step * STEP_DECAY, False

    nxt, status, _err = cv2.calcOpticalFlowPyrLK(prev_gray, cur_gray, points, None, **_LK)
    good = status.ravel() == 1
    if good.sum() <= 6:
        return 0.0, prev_step * STEP_DECAY, False

    u0 = points[good][:, 0, 0].astype(np.float64)
    v0 = points[good][:, 0, 1].astype(np.float64)
    u1 = nxt[good][:, 0, 0].astype(np.float64)
    v1 = nxt[good][:, 0, 1].astype(np.float64)

    usable = (v0 > HORIZON_V + GROUND_MIN_ROWS) & (v1 > HORIZON_V + GROUND_MIN_ROWS)
    if usable.sum() <= 6:
        return 0.0, prev_step * STEP_DECAY, False

    u0 = u0[usable]
    v0 = v0[usable]
    u1 = u1[usable]
    v1 = v1[usable]

    depth0 = FOCAL_PX * CAM_HEIGHT_M / (v0 - HORIZON_V)
    depth1 = FOCAL_PX * CAM_HEIGHT_M / (v1 - HORIZON_V)

    # Forward step: symmetric (a stopped car must be able to read ~0, and a
    # one-sided gate biases every low-speed estimate upward).
    steps = depth0 - depth1
    plausible = np.abs(steps) < MAX_STEP_M
    if plausible.sum() <= 6:
        return 0.0, prev_step * STEP_DECAY, False
    d_forward = float(np.median(steps[plausible]))
    if d_forward < 0.0:
        d_forward = 0.0

    # Yaw: remove the predictable translational expansion, then the residual
    # horizontal motion is pure rotation (identical for every point).
    translation = (u0 - VANISH_U) * d_forward / np.maximum(depth0, 1e-3)
    rotation = (u1 - u0) - translation
    d_yaw = YAW_SIGN * float(np.median(rotation)) / FOCAL_PX

    return d_yaw, d_forward, True


def reconstruct_path(clip_video):
    """Integrate per-frame motion into a top-down ego path.

    Returns fps and arrays X, Y, PSI (metres, metres, radians) in the vehicle
    frame: the path starts at the origin heading +X, with +Y and +PSI to the LEFT.
    """
    capture = cv2.VideoCapture(clip_video)
    fps = float(capture.get(cv2.CAP_PROP_FPS))

    ok, prev = capture.read()
    if not ok:
        capture.release()
        return fps, np.zeros(1), np.zeros(1), np.zeros(1)

    prev_gray = _gray(prev)
    yaws = [0.0]
    steps = [0.0]
    prev_step = DEFAULT_STEP_M
    while True:
        ok, cur = capture.read()
        if not ok:
            break
        cur_gray = _gray(cur)
        d_yaw, d_forward, _valid = estimate_motion(prev_gray, cur_gray, prev_step)
        yaws.append(d_yaw)
        steps.append(d_forward)
        prev_step = d_forward
        prev_gray = cur_gray
    capture.release()

    psi = np.cumsum(np.asarray(yaws))
    step = np.asarray(steps)
    x = np.cumsum(step * np.cos(psi))
    y = np.cumsum(step * np.sin(psi))
    return fps, x, y, psi


def future_trajectories(x, y, psi, fps, lookahead_s=LOOKAHEAD_S, stride_m=STRIDE_M):
    """Per-frame future path in that frame's local coords.

    For frame i, transforms the global path points ahead of it (up to
    lookahead_s) into frame i's frame: x_forward ahead of the car, y_lateral to
    the side. y_lateral is emitted RIGHT-positive to match
    project_ground_point / LATERAL_SIGN (see the module docstring).

    Samples are thinned by ARC length so a tight turn keeps the samples that
    describe its curvature (thinning on forward distance alone discards the
    post-apex arc, and splicing the endpoint back on draws a straight chord
    across the missing curve). x_forward is kept strictly increasing so
    downstream interpolation over image rows is well defined.
    """
    n = len(x)
    la = int(round(lookahead_s * fps))
    out = []
    for i in range(n):
        j_end = min(i + la, n - 1)
        cos_i = np.cos(-psi[i])
        sin_i = np.sin(-psi[i])
        pts = []
        last_fwd = None
        last_lat = None
        for j in range(i, j_end + 1):
            dx = x[j] - x[i]
            dy = y[j] - y[i]
            fwd = cos_i * dx - sin_i * dy
            lat_left = sin_i * dx + cos_i * dy
            lat = -lat_left                      # -> RIGHT-positive for the renderer
            if fwd < MIN_FORWARD_M:
                continue
            if last_fwd is not None:
                if fwd <= last_fwd + 1e-3:
                    continue                     # keep x_forward strictly increasing
                if float(np.hypot(fwd - last_fwd, lat - last_lat)) < stride_m:
                    continue                     # thin by arc length
            last_fwd = fwd
            last_lat = lat
            pts.append([round(float(fwd), 3), round(float(lat), 3)])
        out.append(pts)
    return out


def main():
    clip = sys.argv[1]
    out = sys.argv[2]
    lookahead = float(sys.argv[3]) if len(sys.argv) > 3 else LOOKAHEAD_S

    fps, x, y, psi = reconstruct_path(clip)
    if not (fps and fps > 1.0):
        print("Refusing to continue: video reports fps=%r (cannot build a time horizon)." % fps)
        raise SystemExit(1)

    traj = future_trajectories(x, y, psi, fps, lookahead_s=lookahead)
    write_json(out, {
        "clip_video": clip,
        "fps": fps,
        "frame_count": len(x),
        "lookahead_s": lookahead,
        "focal_px": FOCAL_PX,
        "cam_height_m": CAM_HEIGHT_M,
        "horizon_v": HORIZON_V,
        "vanish_u": VANISH_U,
        "lateral_convention": "right_positive",
        "future_trajectories": traj,
    })
    speed_kmh = float(np.median(np.hypot(np.diff(x), np.diff(y)))) * fps * 3.6
    non_empty = sum(1 for p in traj if p)
    print("frames: %d | median speed: %.1f km/h | total heading: %+.1f deg (+ = left)" % (
        len(x), speed_kmh, np.degrees(psi[-1])))
    print("non-empty future paths: %d/%d" % (non_empty, len(traj)))
    print("saved:", out)


if __name__ == "__main__":
    main()
