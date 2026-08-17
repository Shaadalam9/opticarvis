r"""Guard the per-frame advance of the planner ribbon.

Drawn static, the plan's already-driven part stays painted on the road and the
curve sits at the wrong distance -- too early before the planned moment's
position, too late after. advance_planner_trajectory re-expresses the
remaining waypoints in the pose the plan has reached; these tests pin the
frame transfer on plans with known geometry.

CPU only. Standalone:

    .venv/bin/python tests/test_planner_advance.py

or under pytest.
"""

import math
import os
import sys

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")

if SRC not in sys.path:
    sys.path.insert(0, SRC)

import final_preview_renderer as R  # noqa: E402

STEP = 0.1


def straight_track(heading_flu_rad, speed=8.0, t0=2.0):
    points_flu = [
        (speed * STEP * i * math.cos(heading_flu_rad),
         speed * STEP * i * math.sin(heading_flu_rad))
        for i in range(64)
    ]

    return {
        "points": [(x, R.PLANNER_LATERAL_SIGN * y) for x, y in points_flu],
        "yaws": [R.PLANNER_LATERAL_SIGN * heading_flu_rad] * 64,
        "t0_s": t0,
        "step_s": STEP,
    }


def arc_track(radius=30.0, sweep_deg=90.0):
    points = []
    yaws = []

    for i in range(64):
        angle = (i / 63.0) * math.radians(sweep_deg)
        points.append((radius * math.sin(angle),
                       R.PLANNER_LATERAL_SIGN * radius * (1.0 - math.cos(angle))))
        yaws.append(R.PLANNER_LATERAL_SIGN * angle)

    return {"points": points, "yaws": yaws, "t0_s": 0.0, "step_s": STEP}


def test_straight_plan_stays_dead_ahead():
    """Advancing along a straight plan must never invent lateral offset."""
    track = straight_track(math.radians(30))

    for clip_time in (2.0, 3.0, 5.0, 7.5):
        remaining = R.advance_planner_trajectory(track, clip_time)

        assert remaining is not None, "plan exhausted too early at t=%.1f" % clip_time
        worst = max(abs(y) for _x, y in remaining)
        assert worst < 1e-6, (
            "straight plan grew %.4f m of lateral after advancing" % worst)


def test_remaining_count_shrinks_with_time():
    track = straight_track(0.0)
    counts = [len(R.advance_planner_trajectory(track, t)) for t in (2.0, 4.0, 6.0)]
    assert counts[0] > counts[1] > counts[2], (
        "the plan must be consumed as the clip advances, got %r" % counts)


def test_arc_keeps_bending_the_same_way():
    """Halfway through a left arc, the rest must still bend left from the new
    pose, and its first point must sit nearly dead ahead."""
    track = arc_track()
    remaining = R.advance_planner_trajectory(track, 3.15)

    assert remaining is not None
    assert abs(remaining[0][1]) < 0.05, (
        "first remaining point off-axis: %r" % (remaining[0],))
    assert remaining[-1][1] * R.PLANNER_LATERAL_SIGN > 5.0, (
        "the remaining arc lost its bend after the frame transfer")


def test_exhausted_plan_returns_none():
    """Past the horizon the caller must fall back, not redraw a stale plan."""
    track = straight_track(0.0, t0=2.0)
    assert R.advance_planner_trajectory(track, 2.0 + 6.3) is None


def test_before_t0_clamps_to_plan_start():
    track = straight_track(0.0, t0=5.0)
    early = R.advance_planner_trajectory(track, 1.0)
    at_t0 = R.advance_planner_trajectory(track, 5.0)
    assert early is not None and len(early) == len(at_t0), (
        "before the planned moment the full plan should be shown")


def test_world_anchoring_pins_the_plan_to_the_street():
    """With actual ego poses, the planned curve stays at its world spot and
    approaches exactly as fast as the car actually drives -- even when the
    driver diverges from the plan.
    """
    fps = 30.0
    points = []

    for i in range(64):
        x = 8.0 * STEP * i
        y = 0.0 if x < 10 else -(x - 10) * 0.5
        points.append((x, y))

    track = {"points": points, "yaws": [0.0] * 64, "t0_s": 2.0, "step_s": STEP}
    # The actual car drives straight the whole time, diverging from the plan.
    poses = [[8.0 * (i / fps), 0.0, 0.0] for i in range(300)]

    for clip_time in (2.0, 2.5, 3.0):
        remaining = R.anchor_planner_trajectory(track, clip_time, poses, fps)
        kink_x = next(x for x, y in remaining if y < -0.01)
        driven = 8.0 * (clip_time - 2.0)
        assert abs(kink_x - (10.0 - driven)) < 1.0, (
            "kink drifted off its world spot: t=%.1f kink=%.2f driven=%.1f"
            % (clip_time, kink_x, driven))

    # Once the car has passed the kink, only the curve beyond it remains.
    remaining = R.anchor_planner_trajectory(track, 3.5, poses, fps)
    assert remaining[0][1] < -0.01, (
        "past the kink, every remaining point must already be in the curve")


def test_anchoring_requires_a_validated_track():
    assert R.ego_poses_from_track(None) is None
    assert R.ego_poses_from_track({
        "heading_validation": {"valid": False},
        "ego_pose_right_positive": [[0, 0, 0]] * 10,
    }) is None, "a rejected heading must not anchor the ribbon"
    assert R.ego_poses_from_track({
        "heading_validation": {"valid": True},
        "ego_pose_right_positive": [[0, 0, 0]] * 10,
    }) is not None


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
