r"""Guard the future-anchor chain math and its renderer consumption.

Everything below the renderer is closed-form testable: under the flat-ground
pinhole model, a known ego step induces an EXACT ground-plane homography
between consecutive frames, so the chain's back-projection of the future
reference pixel can be checked sub-pixel against direct projection of the
future ego ground point. The renderer-side tests guard the arc-parameterised
geometry (the representation a 96-degree fold-back turn needs), the sidecar
validation ladder, and the planner-offset sign convention (the mirror bug the
load_planner_track comment warns about).

CPU only. Standalone:

    .venv/bin/python tests/test_future_anchor.py

or under pytest.
"""

import math
import os
import sys

import cv2
import numpy as np

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")

if SRC not in sys.path:
    sys.path.insert(0, SRC)

import final_preview_renderer as R      # noqa: E402
import future_anchor as FA              # noqa: E402

FOCAL = 1000.0
CAM_H = 1.3
HORIZON = 448.0
VANISH = 636.0


def project(point):
    """Flat-model projection of an ego-frame ground point (x_fwd, y_right)."""
    x, y = point
    return (VANISH + FOCAL * y / x, HORIZON + FOCAL * CAM_H / x)


def to_frame(point_world, pose):
    """World ground point -> ego frame of `pose` (x, y, psi)."""
    px, py, psi = pose
    dx = point_world[0] - px
    dy = point_world[1] - py
    cos_p, sin_p = math.cos(psi), math.sin(psi)
    return (cos_p * dx + sin_p * dy, -sin_p * dx + cos_p * dy)


def ground_homography(pose_a, pose_b):
    """EXACT homography frame a -> frame b induced by the ego step, from four
    ground-point correspondences (a plane-to-plane map is fully determined)."""
    spread = [(6.0, -3.0), (6.0, 3.0), (25.0, -4.0), (25.0, 4.0)]
    src = []
    dst = []

    for world in spread:
        in_a = to_frame(world, pose_a)
        in_b = to_frame(world, pose_b)
        src.append(project(in_a))
        dst.append(project(in_b))

    return cv2.getPerspectiveTransform(
        np.array(src, dtype=np.float32), np.array(dst, dtype=np.float32))


def drive(poses_spec):
    """Integrate (step_m, dyaw_rad) increments into world poses from origin."""
    poses = [(0.0, 0.0, 0.0)]

    for step, dyaw in poses_spec:
        x, y, psi = poses[-1]
        psi2 = psi + dyaw
        poses.append((x + step * math.cos(psi2), y + step * math.sin(psi2), psi2))

    return poses


def ref_pixel():
    return FA.reference_pixel(FOCAL, CAM_H, HORIZON, VANISH, ref_ahead_m=4.5)


def expected_anchor(pose_now, pose_future):
    """Where frame-now sees the ground spot 4.5 m ahead of the future ego."""
    fx, fy, fpsi = pose_future
    world = (fx + 4.5 * math.cos(fpsi), fy + 4.5 * math.sin(fpsi))
    return project(to_frame(world, pose_now))


def test_back_projection_matches_direct_projection():
    """Pure translation, pure yaw, and a fast-turn step must all round-trip."""
    for step, dyaw in [(1.2, 0.0), (0.0, math.radians(2.0)), (0.4, math.radians(-0.8))]:
        poses = drive([(step, dyaw)])
        matrix = ground_homography(poses[0], poses[1])
        got = FA.back_project(matrix, ref_pixel())
        want = expected_anchor(poses[0], poses[1])
        err = math.hypot(got[0] - want[0], got[1] - want[1])
        assert err < 0.1, (
            "step %.2f dyaw %.2fdeg: anchor off by %.3f px"
            % (step, math.degrees(dyaw), err))


def test_chain_composition_matches_long_hop():
    """A 24 deg/s left turn at 5 m/s, 30 fps: chained 1-hops == direct hop,
    and the mixed keyframe chain of compose_chain matches both."""
    spec = [(5.0 / 30.0, math.radians(-24.0 / 30.0))] * 24
    poses = drive(spec)

    hops_1 = [ground_homography(poses[i], poses[i + 1]) for i in range(24)]
    hops_k = {m: ground_homography(poses[m * 8], poses[m * 8 + 8]) for m in range(3)}

    direct = ground_homography(poses[0], poses[24])
    chained = FA.compose_chain(hops_1, {}, 0, 24)
    mixed = FA.compose_chain(hops_1, hops_k, 0, 24)

    probe = ref_pixel()

    for name, matrix in [("1-hop chain", chained), ("keyframe chain", mixed)]:
        got = FA.back_project(matrix, probe)
        want = FA.back_project(direct, probe)
        err = math.hypot(got[0] - want[0], got[1] - want[1])
        assert err < 0.5, "%s deviates %.3f px from the direct homography" % (name, err)


def test_chain_refuses_missing_hops():
    hops = [np.eye(3), None, np.eye(3)]
    assert FA.compose_chain(hops, {}, 0, 3) is None


def test_arclength_ignores_heading_bias():
    """s_m must come from position deltas only: a linear heading drift added to
    the poses may not change the arclength."""
    poses = [[i * 0.4, 0.0, 0.0] for i in range(50)]
    drifted = [[x, y, psi + 0.02 * i] for i, (x, y, psi) in enumerate(poses)]
    a = FA.pose_arclength(poses)
    b = FA.pose_arclength(drifted)
    assert np.allclose(a, b), "heading drift leaked into the arclength"


def _set_calibration():
    R.CAM_FOCAL_PX = FOCAL
    R.CAM_HEIGHT_M = CAM_H
    R.HORIZON_V = HORIZON
    R.VANISH_U = VANISH


def test_arc_geometry_expresses_a_folded_turn():
    """A 96-degree arc folds back in forward distance; the arc-parameterised
    builder must still produce a sane band where y(x) cannot."""
    _set_calibration()
    radius = 18.0
    path = [(radius * math.sin(a), -radius * (1.0 - math.cos(a)))
            for a in np.linspace(0.05, math.radians(96.0), 60)]

    geometry = R.build_arc_ribbon_geometry(path)
    assert geometry is not None, "arc builder refused a plain constant-radius turn"
    assert geometry.get("ordered") is True
    assert geometry["near_v"] > geometry["far_v"]
    assert len(geometry["centre"]) >= 3

    # rails must straddle the centre, not collapse onto it
    gaps = np.hypot(*(geometry["left"] - geometry["right"]).T)
    assert float(np.min(gaps)) > 1.0, "rails collapsed on the folded turn"


def test_arc_geometry_truncates_at_measured_end():
    """No extrapolation: a 7 m path must not paint a 17 m band."""
    _set_calibration()
    path = [(x, 0.0) for x in np.arange(4.5, 7.01, 0.25)]
    geometry = R.build_arc_ribbon_geometry(path)
    assert geometry is not None
    v_end = HORIZON + FOCAL * CAM_H / 7.0
    assert geometry["far_v"] >= v_end - 2.0, (
        "band extends beyond the measured path end (far_v %.1f < %.1f)"
        % (geometry["far_v"], v_end))


def _anchor_payload(frames=100, calib=None):
    return {
        "type": "future_anchor_track",
        "version": 1,
        "frame_count": frames,
        "calib": calib or {"focal_px": FOCAL, "cam_height_m": CAM_H,
                           "horizon_v": HORIZON, "vanish_u": VANISH},
        "lateral_convention": "right_positive",
        "anchors": [[[600.0, 700.0, 4.5], [610.0, 650.0, 8.0], [620.0, 600.0, 12.0]]
                    for _ in range(frames)],
    }


def test_sidecar_validation_ladder():
    """Version, frame-count and calibration mismatches must each degrade to
    None (the renderer then falls down the ladder), never raise."""
    _set_calibration()
    R._RESOLUTION_SCALE_APPLIED = 1.0
    saved_flag = R.USE_FUTURE_ANCHOR
    R.USE_FUTURE_ANCHOR = True

    try:
        assert R.parse_future_anchors(_anchor_payload(), 100) is not None

        bad_version = _anchor_payload()
        bad_version["version"] = 7
        assert R.parse_future_anchors(bad_version, 100) is None

        assert R.parse_future_anchors(_anchor_payload(frames=100), 400) is None

        skewed = _anchor_payload(calib={"focal_px": FOCAL, "cam_height_m": CAM_H,
                                        "horizon_v": HORIZON + 40.0, "vanish_u": VANISH})
        assert R.parse_future_anchors(skewed, 100) is None

        R.USE_FUTURE_ANCHOR = False
        assert R.parse_future_anchors(_anchor_payload(), 100) is None
    finally:
        R.USE_FUTURE_ANCHOR = saved_flag


def test_sidecar_scaling_touches_pixels_not_metres():
    _set_calibration()
    saved_flag = R.USE_FUTURE_ANCHOR
    R.USE_FUTURE_ANCHOR = True

    try:
        R._RESOLUTION_SCALE_APPLIED = 1.0
        base = R.parse_future_anchors(_anchor_payload(), 100)

        R._RESOLUTION_SCALE_APPLIED = 3.0
        R.HORIZON_V = HORIZON * 3.0
        R.VANISH_U = VANISH * 3.0
        scaled = R.parse_future_anchors(_anchor_payload(), 100)

        assert np.allclose(scaled[0][:, :2], base[0][:, :2] * 3.0)
        assert np.allclose(scaled[0][:, 2], base[0][:, 2]), "s_m is metric; scaling it is a bug"
    finally:
        R.USE_FUTURE_ANCHOR = saved_flag
        R._RESOLUTION_SCALE_APPLIED = 1.0
        _set_calibration()


def _straight_track(frames=200, step=0.4):
    return {
        "heading_validation": {"valid": True},
        "ego_pose_right_positive": [[i * step, 0.0, 0.0] for i in range(frames)],
    }


def test_planner_offset_zero_when_plan_matches_drive():
    _set_calibration()
    track = {"points": [(x, 0.0) for x in np.arange(0.5, 20.0, 0.5)],
             "yaws": [0.0] * 39, "t0_s": 1.0, "step_s": 0.1}
    offset = R.planner_offset_function(track, _straight_track(), 30.0)
    assert offset is not None

    for s in (14.0, 20.0, 26.0):
        assert abs(offset(s)) < 1e-6, "identical plan and drive must give zero offset"


def test_planner_offset_sign_right_positive():
    """A plan displaced to the RIGHT of the driven path must give POSITIVE
    offsets: right-positive is the one lateral convention in the renderer."""
    _set_calibration()
    track = {"points": [(x, 1.5) for x in np.arange(0.5, 20.0, 0.5)],
             "yaws": [0.0] * 39, "t0_s": 1.0, "step_s": 0.1}
    offset = R.planner_offset_function(track, _straight_track(), 30.0)
    assert offset is not None

    value = offset(20.0)
    assert value > 1.0, "rightward plan read as %.2f (expected ~ +1.5)" % value

    # And a leftward plan (as an FLU left curve arrives after the
    # PLANNER_LATERAL_SIGN=-1 flip: negative lateral) must read negative.
    track_left = {"points": [(x, -1.5) for x in np.arange(0.5, 20.0, 0.5)],
                  "yaws": [0.0] * 39, "t0_s": 1.0, "step_s": 0.1}
    offset_left = R.planner_offset_function(track_left, _straight_track(), 30.0)
    assert offset_left(20.0) < -1.0, "leftward plan lost its sign"


def test_anchor_geometry_takes_precedence():
    """resolve_frame_overlay must draw the anchor geometry when given one."""
    _set_calibration()
    path = [(x, 0.0) for x in np.arange(4.5, 18.0, 0.4)]
    geometry = R.build_arc_ribbon_geometry(path)
    assert geometry is not None

    overlay, near_v, far_v = R.resolve_frame_overlay(
        None, 720, 1280, None, None, {}, anchor_geometry=geometry)

    assert overlay is not None
    assert near_v == geometry["near_v"] and far_v == geometry["far_v"]
    assert R.LAST_RIBBON_GEOMETRY is geometry


def test_ground_band_follows_the_rig_not_the_dev_horizon():
    """The band must name a stretch of ROAD, not a stretch of image.

    Hardcoded rows silently encode one rig's horizon (row -> distance is
    d = f*H/(v - horizon)). On a camera whose horizon sits 125 px higher those
    same rows pointed at the ego vehicle's bonnet, RANSAC fitted the bonnet --
    rigid with the camera, so every hop composed to identity -- and a car doing
    39 km/h was measured as stationary. Same metres, whatever the rig.
    """
    dev = FA.ground_band_rows(
        {"horizon_v": 448.0, "focal_px": FOCAL, "cam_height_m": CAM_H, "vanish_u": VANISH})
    other = FA.ground_band_rows(
        {"horizon_v": 322.9, "focal_px": FOCAL, "cam_height_m": CAM_H, "vanish_u": VANISH})

    assert other[0] < dev[0] and other[1] < dev[1], (
        "a higher horizon must move the band UP the image; got dev=%s other=%s"
        % (dev, other))

    # both bands must denote the same forward window in metres
    for horizon, band in ((448.0, dev), (322.9, other)):
        far = FOCAL * CAM_H / (band[0] - horizon)
        near = FOCAL * CAM_H / (band[1] - horizon)
        assert abs(far - FA.GROUND_BAND_FAR_M) < 3.0, "far edge %.1f m" % far
        assert abs(near - FA.GROUND_BAND_NEAR_M) < 1.0, "near edge %.1f m" % near

    # no calibration -> the historical constant, so a dev-rig clip is unchanged
    assert FA.ground_band_rows(None) == FA.GROUND_BAND_FALLBACK


def test_hood_rows_are_detected_and_excluded():
    """A rigid bottom strip (the bonnet) must be found and cut from the band."""
    rng = np.random.default_rng(4)
    texture = rng.integers(0, 255, size=(720, 1280), dtype=np.uint8)
    hood = rng.integers(0, 255, size=(180, 1280), dtype=np.uint8)

    grays = []

    for i in range(8):
        frame = np.roll(texture, i * 9, axis=0).copy()   # road streams downward
        frame[540:, :] = hood                            # bonnet never moves
        grays.append(frame)

    hood_top = FA.detect_hood_row(grays, (470, 714))

    assert hood_top is not None, "a static bottom strip must be detected"
    assert 480 <= hood_top <= 560, "bonnet top found at row %s, expected ~540" % hood_top


def test_anchor_span_counts_drawable_frames_only():
    """A plane-locked chain emits many anchors that span no ground at all.

    Counting those as successes is exactly how the bonnet lock disguised itself
    as a stationary vehicle, so the yield metric uses the renderer's threshold.
    """
    frozen = [[523.0 + 0.01 * i, 745.0, 4.5 + 0.001 * i] for i in range(200)]
    moving = [[523.0, 745.0 - 4.0 * i, 4.5 + 0.4 * i] for i in range(40)]

    assert FA.anchor_span_m(frozen) < FA.MIN_DRAWABLE_SPAN_M
    assert FA.anchor_span_m(moving) >= FA.MIN_DRAWABLE_SPAN_M
    assert FA.anchor_span_m([]) == 0.0


if __name__ == "__main__":
    failures = 0

    for name, test in sorted(globals().items()):
        if name.startswith("test_") and callable(test):
            try:
                test()
                print("PASS  %s" % name)
            except AssertionError as error:
                failures += 1
                print("FAIL  %s\n      %s" % (name, error))

    raise SystemExit(1 if failures else 0)
