r"""Guard the resolution scaling in final_preview_renderer.

The camera scalars are absolute pixels at 1280x720 while most clips are
3840x2160, so apply_resolution_scaling() rescales them per frame. Two properties
have to hold, and neither is obvious from reading the code:

  * s == 1.0 must be a bit-exact no-op, or every 720p render (and every figure
    already published from one) silently changes.
  * scaling must preserve metric depth, since d = f*H/(v - horizon) is what
    turns the geometry into metres.

The likely regression is someone adding a new pixel constant and not listing it
in RESOLUTION_SCALED_PX -- it would look fine at 1280x720 and be a third of its
intended size at 4K. test_no_unscaled_pixel_constants catches that class.

Runs without a GPU, a clip, or any model. Standalone:

    python tests/test_calibration_scaling.py

or under pytest:

    pytest tests/
"""

import importlib
import os
import sys

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")

if SRC not in sys.path:
    sys.path.insert(0, SRC)


def fresh():
    """A renderer module with its constants back at reference values."""
    import final_preview_renderer

    return importlib.reload(final_preview_renderer)


def test_reference_resolution_is_a_noop():
    R = fresh()
    watched = R.RESOLUTION_SCALED_PX + R.RESOLUTION_SCALED_AREA + ("RIBBON_AIM_BAND",)
    before = {name: getattr(R, name) for name in watched}

    assert R.apply_resolution_scaling(R.CALIB_REF_WIDTH, R.CALIB_REF_HEIGHT) == 1.0

    changed = [n for n in watched if getattr(R, n) != before[n]]
    assert not changed, "s=1.0 must not touch anything, but changed: %s" % changed


def test_projection_scales_linearly():
    R = fresh()
    points = [(10.0, 0.0), (20.0, 1.5), (35.0, -2.0)]
    base = [R.project_ground_point(x, y) for x, y in points]

    R = fresh()
    R.apply_resolution_scaling(3 * R.CALIB_REF_WIDTH, 3 * R.CALIB_REF_HEIGHT)
    scaled = [R.project_ground_point(x, y) for x, y in points]

    for (bu, bv), (su, sv) in zip(base, scaled):
        assert abs(su - 3.0 * bu) < 1e-6, "u did not scale: %s vs %s" % (su, bu)
        assert abs(sv - 3.0 * bv) < 1e-6, "v did not scale: %s vs %s" % (sv, bv)


def test_metric_depth_is_preserved():
    R = fresh()
    rows = [R.HORIZON_V + d for d in (80.0, 150.0, 240.0)]
    base = [R.ground_distance_from_row(v) for v in rows]

    R = fresh()
    R.apply_resolution_scaling(3 * R.CALIB_REF_WIDTH, 3 * R.CALIB_REF_HEIGHT)
    scaled = [R.ground_distance_from_row(3.0 * v) for v in rows]

    for a, b in zip(base, scaled):
        assert abs(a - b) < 1e-9, "depth changed with resolution: %s vs %s" % (a, b)


def test_metric_constants_are_not_scaled():
    R = fresh()
    # CAM_HEIGHT_M is what makes the depth term metric; LANE_WIDTH_REF_M is a
    # metre width whose pixel expansion already carries s via (v - HORIZON_V).
    # Scaling either corrupts the geometry rather than fixing it.
    metric = {n: getattr(R, n) for n in ("CAM_HEIGHT_M", "LANE_WIDTH_REF_M", "IMAGE_SIZE")}

    R.apply_resolution_scaling(3 * R.CALIB_REF_WIDTH, 3 * R.CALIB_REF_HEIGHT)

    for name, value in metric.items():
        assert getattr(R, name) == value, "%s must not scale" % name


def test_areas_scale_quadratically():
    R = fresh()
    before = {n: getattr(R, n) for n in R.RESOLUTION_SCALED_AREA}

    R.apply_resolution_scaling(3 * R.CALIB_REF_WIDTH, 3 * R.CALIB_REF_HEIGHT)

    for name, value in before.items():
        assert getattr(R, name) == value * 9, (
            "%s counts pixels in a region that grows in both axes, so it scales "
            "by s squared, not s" % name
        )


def test_blur_kernels_stay_odd_positive_ints():
    R = fresh()
    R.apply_resolution_scaling(3 * R.CALIB_REF_WIDTH, 3 * R.CALIB_REF_HEIGHT)

    for name in R.RESOLUTION_ODD_KERNELS:
        value = getattr(R, name)
        assert isinstance(value, int) and value > 0 and value % 2 == 1, (
            "cv2 Gaussian kernels must be odd positive ints; %s is %r" % (name, value)
        )


def test_scaling_is_idempotent():
    R = fresh()
    R.apply_resolution_scaling(3 * R.CALIB_REF_WIDTH, 3 * R.CALIB_REF_HEIGHT)
    once = R.LANE_CURVE_PICK_ROW

    R.apply_resolution_scaling(3 * R.CALIB_REF_WIDTH, 3 * R.CALIB_REF_HEIGHT)
    assert R.LANE_CURVE_PICK_ROW == once, "a second call must not scale again"


def test_no_unscaled_pixel_constants():
    """Every *_PX / *_ROWS constant must be classified, not silently forgotten."""
    R = fresh()
    known = set(R.RESOLUTION_SCALED_PX) | set(R.RESOLUTION_SCALED_AREA)

    # Metre, ratio and frame-count values that merely end in a scanned suffix.
    exempt = {"LANE_WIDTH_REF_M", "CALIB_REF_WIDTH", "CALIB_REF_HEIGHT"}

    suspects = []
    for name, value in vars(R).items():
        if not name.isupper() or name in known or name in exempt:
            continue
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        if name.endswith("_PX") or name.endswith("_ROWS"):
            suspects.append(name)

    assert not suspects, (
        "these look like pixel quantities but are not in RESOLUTION_SCALED_PX "
        "or RESOLUTION_SCALED_AREA: %s" % sorted(suspects)
    )


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failures = 0

    for test in tests:
        try:
            test()
            print("PASS  %s" % test.__name__)
        except AssertionError as error:
            failures += 1
            print("FAIL  %s\n        %s" % (test.__name__, error))

    print("\n%d/%d passed" % (len(tests) - failures, len(tests)))

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
