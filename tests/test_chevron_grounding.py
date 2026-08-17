r"""Guard the chevrons' ground-plane construction.

The original implementation built each chevron in image space (screen tangent,
screen perpendicular): a rigid 2D glyph that rotated with the centreline's
on-screen direction. On curves that produced near-vertical strokes -- arm
angles up to 70 degrees from horizontal, geometrically impossible for road
paint -- and the painted-on-road illusion collapsed (confirmed by a five-lens
visual inspection with pixel measurements). Chevron vertices are now laid out
in ground coordinates and projected through the flat-ground mapping; under
that projection a mark stays a wide, flat V no matter how the path curves.

These tests rasterise chevrons on synthetic centrelines and measure the marks'
aspect: a flat foreshortened V is much wider than tall; the decal bug made
tall narrow pickets. CPU only. Standalone:

    .venv/bin/python tests/test_chevron_grounding.py

or under pytest.
"""

import os
import sys

import cv2
import numpy as np

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")

if SRC not in sys.path:
    sys.path.insert(0, SRC)

import final_preview_renderer as R  # noqa: E402


def rasterise(centre_points, phase_m=0.0, size=(720, 1280)):
    layer = np.zeros(size, dtype=np.uint8)
    R.draw_centerline_chevrons(layer, np.asarray(centre_points, np.float32), phase_m)
    return layer


def mark_aspects(layer):
    """height/width per connected chevron mark."""
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        (layer > 0).astype(np.uint8))
    aspects = []

    for i in range(1, count):
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]

        if w * h > 40:                      # ignore stroke fragments
            aspects.append(h / float(w))

    return aspects


def straight_centre():
    return [(R.VANISH_U, v) for v in range(700, int(R.HORIZON_V) + 60, -2)]


def curved_centre():
    """A centreline sweeping hard left, like the frames that exposed the bug."""
    points = []

    for v in range(700, int(R.HORIZON_V) + 60, -2):
        depth = R.CAM_FOCAL_PX * R.CAM_HEIGHT_M / (v - R.HORIZON_V)
        lateral = -0.03 * depth * depth      # quadratic left curve
        points.append((R.VANISH_U + R.CAM_FOCAL_PX * lateral / depth, v))

    return points


def test_straight_marks_are_flat():
    aspects = mark_aspects(rasterise(straight_centre()))
    assert len(aspects) >= 3, "expected several marks, got %d" % len(aspects)
    worst = max(aspects)
    assert worst < 0.55, (
        "straight-path chevrons must be wide flat Vs; worst height/width %.2f"
        % worst)


def test_curved_marks_stay_flat():
    """The decal bug made curve chevrons tall pickets; ground marks stay flat."""
    aspects = mark_aspects(rasterise(curved_centre()))
    assert len(aspects) >= 3
    worst = max(aspects)
    assert worst < 0.75, (
        "curved-path chevrons must remain foreshortened ground marks; "
        "worst height/width %.2f says they rotated in image space" % worst)


def test_phase_streams_marks_toward_viewer():
    """Advancing the travelled distance must move every mark down (nearer)."""
    centre = straight_centre()
    layer_a = rasterise(centre, phase_m=0.0)
    layer_b = rasterise(centre, phase_m=0.8)

    def lowest_mark_row(layer):
        rows = np.nonzero(layer.any(axis=1))[0]
        return rows.max() if len(rows) else -1

    # with more distance travelled, the pattern as a whole sits lower (nearer)
    a_rows = np.nonzero(layer_a.any(axis=1))[0]
    b_rows = np.nonzero(layer_b.any(axis=1))[0]
    assert len(a_rows) and len(b_rows)
    assert b_rows.min() >= a_rows.min() - 1, "marks must not migrate away"

    # and the topmost (farthest) mark must have moved down or cycled
    assert b_rows.min() != a_rows.min() or lowest_mark_row(layer_b) != lowest_mark_row(layer_a), (
        "advancing travelled distance changed nothing -- phase is disconnected")


def test_strokes_never_escape_the_band():
    """build_path_overlay must clip marks to the band polygon."""
    with open(os.path.join(SRC, "final_preview_renderer.py"), encoding="utf-8") as handle:
        source = handle.read()

    assert "cv2.bitwise_and(dash_cov, band_mask)" in source, (
        "chevron coverage must be masked to the band polygon"
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
