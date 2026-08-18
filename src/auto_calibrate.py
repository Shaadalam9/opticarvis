r"""Estimate a clip's vanishing point and write its camera calibration file.

The flat-ground defaults (VANISH_U 636, HORIZON_V 448) belong to the rig the
project was developed on. On another dashcam they are wrong in a way nothing
crashes on: measured on Fvt6rD9tt1c_22, the true vanishing point sits ~113 px
left of the default, which biased the VO yaw +225 deg over a 30 s clip, planted
the projected ribbon beside the lane, and -- worst -- skewed the ego history
fed to the Alpamayo planner, which then predicted a harder turn than the scene
warranted. One constant, three symptoms. Across 100 cities of different rigs it
varies clip by clip, so it is estimated per clip:

    .venv/bin/python src/auto_calibrate.py [clip.mp4]

Writes workflow_outputs/calibration/<tag>_camera_calibration.json -- the file
final_preview_renderer.apply_calibration_overrides() already reads -- with
VANISH_U/HORIZON_V at the 1280x720 calibration reference. ego_trajectory.py and
the Alpamayo2 wrapper read the same file, so all three consumers agree.

Method: forward motion makes the scene's optical flow radiate from the
vanishing point, so each flow vector's line passes near it. Flow is sampled
only at instants the scene-pan cross-check certifies as straight (yaw shifts
the radiant point sideways), and the per-instant least-squares intersections
are combined by median. If too few straight instants exist, or the estimate
lands outside sane bounds, NO file is written and every consumer keeps the
defaults -- a wrong calibration is worse than the default one.
"""

import os
import sys

import cv2
import numpy as np

SRC_DIR = os.path.dirname(os.path.abspath(__file__))

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from pipeline_common import (  # noqa: E402
    CLIP_VIDEO,
    ensure_dir,
    normalise_path,
    segment_tag,
    workflow_path,
    write_json,
)

CALIB_REF_WIDTH = 1280
CALIB_REF_HEIGHT = 720

# Scene-pan threshold certifying an instant as straight, at the 320x90
# correlation scale (matches ego_trajectory's validation windows).
PAN_QUIET_PX = 0.35

# Estimate quality gates. Below MIN_SAMPLES the clip simply was not straight
# for long enough to calibrate from; outside the bounds the estimator latched
# onto something that is not the road's vanishing point.
MIN_SAMPLES = 25
VANISH_U_BOUNDS = (CALIB_REF_WIDTH * 0.20, CALIB_REF_WIDTH * 0.80)
HORIZON_V_BOUNDS = (CALIB_REF_HEIGHT * 0.40, CALIB_REF_HEIGHT * 0.80)
MAX_IQR_U = 280.0


# Consensus gates for the per-instant intersection. Independently moving
# vehicles produce flow lines that do not pass through the ego's focus of
# expansion, so a plain least-squares fit over every line is dragged off the
# true vanishing point -- and consistently enough that the outer median over
# hundreds of instants cannot recover it (measured on a dense-traffic Nairobi
# clip: 20 px high, 40 px right, with a 95% CI of only 4.6 px around the wrong
# answer). Fitting only the largest consistent set of lines removes the
# moving-object votes instead of averaging them in.
VP_RANSAC_ITERATIONS = 200
VP_INLIER_PX = 3.0
VP_MIN_CONSENSUS = 8


def _intersect_lines(normals, rhs):
    vp, *_ = np.linalg.lstsq(normals, rhs, rcond=None)

    return vp


def consensus_vanishing_point(normals, rhs, rng):
    """Vanishing point from the largest set of mutually consistent flow lines.

    Returns None when no pair gathers VP_MIN_CONSENSUS support, which lets the
    caller fall back rather than trust a two-line accident.
    """
    count = len(normals)

    if count < VP_MIN_CONSENSUS:
        return None

    best_inliers = None
    best_score = 0

    for _ in range(VP_RANSAC_ITERATIONS):
        pair = rng.choice(count, size=2, replace=False)
        matrix = normals[pair]

        if abs(float(np.linalg.det(matrix))) < 1e-6:
            continue

        try:
            candidate = np.linalg.solve(matrix, rhs[pair])
        except np.linalg.LinAlgError:
            continue

        # distance from the candidate to each flow line (unit normals)
        residual = np.abs(normals @ candidate - rhs)
        inliers = residual < VP_INLIER_PX
        score = int(inliers.sum())

        if score > best_score:
            best_score = score
            best_inliers = inliers

    if best_inliers is None or best_score < VP_MIN_CONSENSUS:
        return None

    return _intersect_lines(normals[best_inliers], rhs[best_inliers])


def flow_vanishing_point(gray_a, gray_b, rng=None):
    """Intersection of one instant's flow lines, or None.

    The intersection is taken over the largest consistent subset of lines; a
    plain fit over all of them believes every moving vehicle in the scene.
    """
    points = cv2.goodFeaturesToTrack(gray_a, 600, 0.01, 8)

    if points is None or len(points) < 40:
        return None

    nxt, status, _err = cv2.calcOpticalFlowPyrLK(
        gray_a, gray_b, points, None, winSize=(21, 21), maxLevel=3)
    good = status.ravel() == 1
    p0 = points[good][:, 0, :]
    flow = nxt[good][:, 0, :] - p0
    magnitude = np.hypot(flow[:, 0], flow[:, 1])
    keep = (magnitude > 1.2) & (magnitude < 40)
    p0, flow = p0[keep], flow[keep]

    if len(p0) < 30:
        return None

    direction = flow / np.hypot(flow[:, 0], flow[:, 1])[:, None]
    normals = np.stack([-direction[:, 1], direction[:, 0]], axis=1)
    rhs = (normals * p0).sum(axis=1)

    if rng is None:
        rng = np.random.default_rng(0)

    vp = consensus_vanishing_point(normals, rhs, rng)

    if vp is None:
        vp = _intersect_lines(normals, rhs)

    return vp


def scene_pan(gray_a, gray_b):
    a = cv2.resize(gray_a, (320, 180)).astype(np.float32)[:90, :]
    b = cv2.resize(gray_b, (320, 180)).astype(np.float32)[:90, :]
    (dx, _dy), _response = cv2.phaseCorrelate(a, b)

    return dx


def estimate_vanishing_point(clip_path):
    """Return (result_dict, reasons). result_dict is None when unusable."""
    capture = cv2.VideoCapture(clip_path)

    if not capture.isOpened():
        return None, ["could not open clip"]

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    scale = CALIB_REF_WIDTH / float(width) if width else 1.0

    samples = []
    previous = None
    index = 0

    try:
        while True:
            ok, frame = capture.read()

            if not ok:
                break

            if index % 2 == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                if previous is not None and abs(scene_pan(previous, gray)) < PAN_QUIET_PX:
                    vp = flow_vanishing_point(previous, gray)

                    if vp is not None:
                        u, v = vp[0] * scale, vp[1] * scale

                        if 100 < u < CALIB_REF_WIDTH - 100 and 100 < v < CALIB_REF_HEIGHT - 60:
                            samples.append((u, v))

                previous = gray

            index += 1
    finally:
        capture.release()

    if len(samples) < MIN_SAMPLES:
        return None, ["only %d straight-instant samples (need %d) -- not enough "
                      "straight driving to calibrate from" % (len(samples), MIN_SAMPLES)]

    us = np.array([s[0] for s in samples])
    vs = np.array([s[1] for s in samples])
    vanish_u = float(np.median(us))
    horizon_v = float(np.median(vs))
    iqr_u = float(np.percentile(us, 75) - np.percentile(us, 25))

    reasons = []

    if not (VANISH_U_BOUNDS[0] <= vanish_u <= VANISH_U_BOUNDS[1]):
        reasons.append("VANISH_U %.0f outside sane bounds %s" % (vanish_u, VANISH_U_BOUNDS))

    if not (HORIZON_V_BOUNDS[0] <= horizon_v <= HORIZON_V_BOUNDS[1]):
        reasons.append("HORIZON_V %.0f outside sane bounds %s" % (horizon_v, HORIZON_V_BOUNDS))

    if iqr_u > MAX_IQR_U:
        reasons.append("VANISH_U spread too wide (IQR %.0f px > %.0f) -- estimate "
                       "unstable on this clip" % (iqr_u, MAX_IQR_U))

    if reasons:
        return None, reasons

    return {
        "VANISH_U": round(vanish_u, 1),
        "HORIZON_V": round(horizon_v, 1),
        "CALIB_REF_WIDTH": CALIB_REF_WIDTH,
        "CALIB_REF_HEIGHT": CALIB_REF_HEIGHT,
        "estimator": {
            "method": "flow_convergence_on_straight_instants",
            "samples": len(samples),
            "iqr_u_px": round(iqr_u, 1),
        },
    }, []


def main():
    clip = normalise_path(sys.argv[1]) if len(sys.argv) > 1 else CLIP_VIDEO
    output = workflow_path("calibration", segment_tag() + "_camera_calibration.json")

    print("")
    print("Auto camera calibration")
    print("=======================")
    print("clip:", clip)

    result, reasons = estimate_vanishing_point(clip)

    if result is None:
        print("No calibration written; consumers keep the defaults "
              "(a wrong calibration is worse than the default one):")

        for reason in reasons:
            print("  " + reason)

        # Not an error: the conservative outcome is the defaults.
        return 0

    ensure_dir(os.path.dirname(output))
    write_json(output, result)

    print("VANISH_U : %.1f (default 636.0)" % result["VANISH_U"])
    print("HORIZON_V: %.1f (default 448.0)" % result["HORIZON_V"])
    print("samples  : %d | IQR_u %.0f px" % (
        result["estimator"]["samples"], result["estimator"]["iqr_u_px"]))
    print("saved:", output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
