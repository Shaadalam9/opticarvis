"""Guard the restyle compositor (src/render/restyle_render.py).

It replays a render's dumped geometry through the renderer's own functions
under another render config. These tests pin the frame order it copies from
render_video, that each swept study parameter reaches the pixels, and the
contact sheet. CPU only, synthetic frames, no video or models.
"""

import importlib
import os
import sys

import numpy as np

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")

if SRC not in sys.path:
    sys.path.insert(0, SRC)

import _paths  # noqa: E402,F401  (every src/<group>/ onto sys.path)

import restyle_render as RS  # noqa: E402

HEIGHT, WIDTH = 720, 1280
TAG = "TESTCLIP000_100"
ENV_KEYS = ("OPTICARVIS_VIDEO_ID", "OPTICARVIS_SEGMENT_START_S")


def with_renderer(config, test):
    """Run test(renderer) with a configured renderer; restore the job env after."""
    saved = {key: os.environ.get(key) for key in ENV_KEYS}
    try:
        renderer = RS.load_renderer(TAG, config)
        return test(renderer)
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        import pipeline_common
        importlib.reload(pipeline_common)
        import final_preview_renderer
        importlib.reload(final_preview_renderer)


def record_for(renderer):
    traj = np.array([(x, 0.3) for x in np.linspace(0, 30, 64)])
    geometry = renderer.build_ribbon_geometry(traj)
    polygon = [[600, 420], [680, 420], [690, 640], [590, 640]]
    return {
        "i": 0,
        "label": "Slowing for pedestrians",
        "ribbon": {
            "centre": np.asarray(geometry["centre"]).tolist(),
            "near_v": geometry["near_v"],
            "far_v": geometry["far_v"],
            "phase_m": 1.5,
        },
        "persons": [{"poly": polygon, "box": [590, 420, 690, 640], "score": 0.3, "dist": 8.0, "tid": 1}],
        "vehicles": [],
    }


def frame():
    return np.random.default_rng(3).integers(40, 200, (HEIGHT, WIDTH, 3), dtype=np.uint8)


def composed(config):
    def run(renderer):
        _persons, with_vehicles = RS.compose_frame(
            renderer, frame(), record_for(renderer), None, 5.0, 0, 30.0)
        return with_vehicles
    return with_renderer(config, run)


def test_compose_frame_follows_render_video_order():
    def run(renderer):
        record = record_for(renderer)
        persons_only, with_vehicles = RS.compose_frame(renderer, frame(), record, None, 5.0, 0, 30.0)

        # render_video, step by step, for the same inputs.
        base = renderer.dim_background(frame())
        geometry = RS.ribbon_geometry(renderer, record["ribbon"])
        renderer.blend_path(base, renderer.build_path_overlay(geometry, HEIGHT, WIDTH, 1.5), None)
        renderer.draw_highlights(base, RS.detections(record["persons"]),
                                 renderer.BOX_COLOUR_CLOSE, renderer.BOX_COLOUR, show_class=False)
        expected_persons = base.copy()
        renderer.draw_text_panel(expected_persons, record["label"])
        renderer.draw_text_panel(base, record["label"])

        assert np.array_equal(persons_only, expected_persons)
        assert np.array_equal(with_vehicles, base)
    with_renderer({}, run)


def test_every_swept_variant_changes_the_frame():
    default = composed({})
    for name, overrides in RS.SWEEP:
        if not overrides:
            continue
        assert not np.array_equal(composed(overrides), default), name


def test_sweep_covers_each_parameter_at_its_limits_and_every_palette():
    names = [name for name, _ in RS.SWEEP]
    assert names[0] == "default" and len(names) == 10
    swept = {key: set() for key in RS.DEFAULT_CONFIG}
    for _name, overrides in RS.SWEEP:
        for key, value in overrides.items():
            swept[key].add(value)
    assert swept["mask_alpha"] == {0.0, 0.7}
    assert swept["trajectory_alpha"] == {0.0, 1.0}
    assert swept["background_dim_alpha"] == {0.0, 0.4}
    assert swept["palette_id"] == {1, 2, 3}


def test_contact_sheet_is_a_labelled_grid(tmp_path=None):
    import tempfile

    directory = tmp_path or tempfile.mkdtemp()
    samples = [("variant_%d" % i, frame()) for i in range(7)]
    path = RS.contact_sheet(samples, os.path.join(str(directory), "sheet.jpg"), columns=5, thumb_width=200)
    import cv2
    sheet = cv2.imread(path)
    # 7 tiles in rows of 5 -> 2 rows, 5 columns of 200 px.
    assert sheet.shape[1] == 1000
    assert sheet.shape[0] == 2 * int(round(HEIGHT * 200 / WIDTH))


def test_outputs_appear_only_when_complete_and_playable():
    """A file under a final name must open: mp4's index is written at close."""
    import tempfile

    import cv2
    import shutil
    from overlay_geometry_dump import GeometryDump

    directory = tempfile.mkdtemp()
    clip = os.path.join(directory, "clip.mp4")
    writer = cv2.VideoWriter(clip, cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (WIDTH, HEIGHT))
    for _ in range(6):
        writer.write(frame())
    writer.release()

    def run(renderer):
        dump = GeometryDump(os.path.join(directory, "geometry"), TAG, {
            "clip_video": clip, "fps": 10.0, "width": WIDTH, "height": HEIGHT, "frame_count": 6,
            "camera": {"horizon_v": float(renderer.HORIZON_V), "vanish_u": float(renderer.VANISH_U),
                       "focal_px": float(renderer.CAM_FOCAL_PX), "cam_height_m": float(renderer.CAM_HEIGHT_M)},
        })
        record = record_for(renderer)
        geometry = RS.ribbon_geometry(renderer, record["ribbon"])
        for index in range(6):
            dump.frame(index, 1.0, record["label"], geometry, RS.detections(record["persons"]), [], None,
                       phase_m=0.1 * index)
        dump.close()
        return dump.geometry_path

    geometry_path = with_renderer({}, run)
    output_dir = os.path.join(directory, "variants")
    for h264 in (False,) + ((True,) if shutil.which("ffmpeg") else ()):
        name = "palette_1_h264" if h264 else "palette_1"
        saved = {key: os.environ.get(key) for key in ENV_KEYS}
        try:
            outputs, _ = RS.restyle(geometry_path, {"palette_id": 1}, name, output_dir, h264=h264)
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
        capture = cv2.VideoCapture(outputs[0])
        frames = 0
        while capture.read()[0]:
            frames += 1
        capture.release()
        assert frames == 6, "the finished file must play all frames (%s)" % name

    leftovers = [f for f in os.listdir(output_dir) if ".writing" in f or ".encoding" in f]
    assert not leftovers, leftovers
    import pipeline_common
    importlib.reload(pipeline_common)
    import final_preview_renderer
    importlib.reload(final_preview_renderer)


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
    raise SystemExit(1 if failures else 0)
