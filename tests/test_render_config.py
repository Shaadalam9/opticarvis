r"""Guard that every study parameter reaches the rendered pixels.

The preference study optimises four overlay parameters (mask_alpha,
trajectory_alpha, background_dim_alpha, palette_id). Three of them used to be
silently dropped by the renderer: trajectory_alpha mapped to no constant,
background_dim_alpha changed a value the dimming lookup table had already baked
in at import, and palette_id set a variable nothing read. A participant's
optimised style would therefore have rendered with the default look.

These tests apply a config and check the rendered pixels, not just the
globals. CPU only, no video or models.
"""

import contextlib
import importlib
import io
import os
import sys

import numpy as np

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")

if SRC not in sys.path:
    sys.path.insert(0, SRC)

import _paths  # noqa: E402,F401  (every src/<group>/ onto sys.path)

import final_preview_renderer as R  # noqa: E402

HEIGHT, WIDTH = 720, 1280
DEFAULT = {"mask_alpha": 0.14, "trajectory_alpha": 0.55, "background_dim_alpha": 0.06, "palette_id": 0}


def renderer_with(**overrides):
    """A freshly loaded renderer with one render config applied."""
    module = importlib.reload(R)
    config = dict(DEFAULT, **overrides)
    with contextlib.redirect_stdout(io.StringIO()):
        module.apply_render_config_to_globals(config)
    return module


def ribbon(module):
    traj = np.array([(x, 0.0) for x in np.linspace(0, 30, 64)])
    return module.build_path_overlay(module.build_ribbon_geometry(traj), HEIGHT, WIDTH)


def highlighted(module, fill_frame):
    polygon = np.array([[600, 400], [680, 400], [690, 620], [590, 620]], np.int32)
    detection = {"spatial_score": 0.3, "box": [590, 400, 690, 620], "mask": polygon, "distance_m": 7.5}
    frame = fill_frame.copy()
    module.draw_highlights(frame, [detection], module.BOX_COLOUR_CLOSE, module.BOX_COLOUR)
    return frame


def grey_frame():
    return np.full((HEIGHT, WIDTH, 3), 128, dtype=np.uint8)


def test_default_config_keeps_the_tuned_look():
    """Palette 0 and the study defaults are the renderer's shipped constants."""
    module = renderer_with()
    assert module.BOX_COLOUR == (0, 220, 255)
    assert module.BOX_COLOUR_CLOSE == (0, 160, 255)
    assert module.ROAD_PATH_COLOUR == (245, 200, 90)
    assert module.ROAD_PATH_CORE_COLOUR == (245, 235, 180)
    assert module.TEXT_COLOUR == (255, 255, 255)
    assert module.TEXT_BACKGROUND == (0, 0, 0)
    assert module.TRAJECTORY_OPACITY_SCALE == 1.0
    assert module.HIGHLIGHT_FILL_ALPHA == 0.14
    assert np.array_equal(module.DIM_LUT, module.build_dim_lut(0.06))


def test_mask_alpha_changes_the_highlight_fill():
    # Each result is taken before the next reload (see renderer_with).
    weak = highlighted(renderer_with(mask_alpha=0.05), grey_frame())
    strong = highlighted(renderer_with(mask_alpha=0.6), grey_frame())
    inside = (slice(450, 580), slice(620, 660))
    assert not np.array_equal(weak[inside], strong[inside])
    # More opacity pulls the fill further from the grey background.
    assert np.abs(strong[inside].astype(int) - 128).mean() > np.abs(weak[inside].astype(int) - 128).mean()


def test_trajectory_alpha_scales_the_ribbon_opacity():
    hidden = ribbon(renderer_with(trajectory_alpha=0.0))
    default = ribbon(renderer_with(trajectory_alpha=0.55))
    strong = ribbon(renderer_with(trajectory_alpha=1.0))
    assert hidden["a_tot"].max() == 0.0, "trajectory_alpha 0 must hide the ribbon"
    assert default["a_tot"].max() > 0.0
    assert strong["a_tot"].max() > default["a_tot"].max()
    assert strong["a_tot"].max() <= 1.0


def test_background_dim_alpha_changes_the_dimmed_frame():
    frame = grey_frame()
    none = renderer_with(background_dim_alpha=0.0).dim_background(frame)
    heavy = renderer_with(background_dim_alpha=0.4).dim_background(frame)
    assert np.array_equal(none, frame), "0 means no dimming"
    assert int(heavy[0, 0, 0]) == int(128 * 0.6)


def test_palette_id_changes_the_path_and_target_colours():
    # renderer_with reloads the one module in place, so capture each result
    # before configuring the next palette.
    default = renderer_with(palette_id=0)
    default_ribbon = ribbon(default)
    default_highlight = highlighted(default, grey_frame())
    default_vehicle = default.VEHICLE_BOX_COLOUR

    saliency = renderer_with(palette_id=3)
    # configs/render_palettes.json palette 3: target #FF0055, trajectory #CCFF00.
    assert saliency.BOX_COLOUR == (85, 0, 255)
    assert saliency.ROAD_PATH_COLOUR == (0, 255, 204)
    assert not np.array_equal(default_ribbon["premul"], ribbon(saliency)["premul"])
    assert not np.array_equal(default_highlight, highlighted(saliency, grey_frame()))
    # Vehicles keep their own colour so they stay distinct from pedestrians.
    assert saliency.VEHICLE_BOX_COLOUR == default_vehicle


def test_unknown_palette_keeps_the_built_in_colours():
    module = renderer_with(palette_id=9)
    assert module.ROAD_PATH_COLOUR == (245, 200, 90)
    assert module.BOX_COLOUR == (0, 220, 255)


def test_every_study_palette_resolves():
    for palette_id in (0, 1, 2, 3):
        module = renderer_with(palette_id=palette_id)
        for name in ("BOX_COLOUR", "ROAD_PATH_COLOUR", "ROAD_PATH_CORE_COLOUR"):
            colour = getattr(module, name)
            assert len(colour) == 3 and all(0 <= c <= 255 for c in colour), (palette_id, name)


if __name__ == "__main__":
    failures = 0
    for name, test in sorted(globals().items()):
        if not name.startswith("test_") or not callable(test):
            continue
        try:
            test()
            print("PASS  %s" % name)
        except AssertionError as error:
            failures += 1
            print("FAIL  %s\n      %s" % (name, error))
    importlib.reload(R)
    raise SystemExit(1 if failures else 0)
