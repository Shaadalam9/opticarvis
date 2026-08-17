r"""Guard the per-clip camera auto-calibration (src/auto_calibrate.py).

The estimator recovers the vanishing point from flow convergence; getting it
wrong re-poisons three consumers at once (planner ego history, VO track,
ribbon projection), so the core math is tested against synthetic frames with a
KNOWN vanishing point, and the wiring against the files.

CPU only, no video files, no models. Standalone:

    .venv/bin/python tests/test_auto_calibrate.py

or under pytest.
"""

import os
import sys

import cv2
import numpy as np

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")

if SRC not in sys.path:
    sys.path.insert(0, SRC)

import auto_calibrate as AC  # noqa: E402


def synthetic_pair(vp_u, vp_v, width=640, height=360, zoom=1.03, seed=7):
    """Two frames of textured ground truth whose flow radiates from (vp_u, vp_v).

    Forward motion is a scale about the vanishing point; warping the first
    frame by that scale produces exactly the radial flow field the estimator
    assumes.
    """
    rng = np.random.default_rng(seed)
    frame_a = rng.integers(0, 255, (height, width), dtype=np.uint8)
    frame_a = cv2.GaussianBlur(frame_a, (5, 5), 0)

    matrix = np.array([
        [zoom, 0.0, vp_u * (1.0 - zoom)],
        [0.0, zoom, vp_v * (1.0 - zoom)],
    ], dtype=np.float64)
    frame_b = cv2.warpAffine(frame_a, matrix, (width, height))

    return frame_a, frame_b


def test_recovers_known_vanishing_point():
    for true_u, true_v in [(320, 180), (250, 200), (400, 150)]:
        frame_a, frame_b = synthetic_pair(true_u, true_v)
        vp = AC.flow_vanishing_point(frame_a, frame_b)

        assert vp is not None, "estimator returned nothing on clean radial flow"
        error = np.hypot(vp[0] - true_u, vp[1] - true_v)
        assert error < 20, (
            "VP (%.0f, %.0f) recovered as (%.0f, %.0f), error %.0f px"
            % (true_u, true_v, vp[0], vp[1], error)
        )


def test_returns_none_on_textureless_frames():
    flat = np.full((360, 640), 128, dtype=np.uint8)
    assert AC.flow_vanishing_point(flat, flat) is None, (
        "no features must mean no estimate, never a fabricated one"
    )


def test_untrusted_estimates_write_nothing():
    """The gates must refuse rather than write a bad calibration."""
    with open(os.path.join(SRC, "auto_calibrate.py"), encoding="utf-8") as handle:
        source = handle.read()

    assert "MIN_SAMPLES" in source and "MAX_IQR_U" in source
    assert "return None, reasons" in source, (
        "gate failures must return no result, so no file is written"
    )
    assert "return 0" in source.split("No calibration written", 1)[1][:400], (
        "a refused calibration is the conservative outcome, not an error"
    )


def test_all_three_consumers_read_the_same_file():
    """Renderer, VO and the planner wrapper must share one calibration source."""
    root = os.path.join(SRC, "..")
    suffix = "_camera_calibration.json"

    for rel in ("src/final_preview_renderer.py", "src/ego_trajectory.py",
                "scripts/alpamayo2_super_wrapper.py"):
        with open(os.path.join(root, rel), encoding="utf-8") as handle:
            assert suffix in handle.read(), (
                "%s no longer reads the per-clip calibration file" % rel
            )

    with open(os.path.join(SRC, "batch_corrected_pipeline.py"), encoding="utf-8") as handle:
        batch = handle.read()

    assert '"OPTICARVIS_AUTO_CALIBRATE", "1"' in batch, (
        "auto-calibration must run by default in the batch"
    )
    assert "OPTICARVIS_CALIBRATION_DIR" in batch, (
        "the planner adapter subprocess must be told where the calibrations live"
    )


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
