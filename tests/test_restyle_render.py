r"""Behavioural test of the post-hoc restyle path, no models needed.

Builds a tiny synthetic clip and geometry dump, composites it twice -- default
style and a deliberately loud variant -- and asserts the styles actually landed
where the geometry says they should. This is the guard for the whole
geometry/compositing split: if the dump format, the edge derivation, or the
style plumbing drifts, colours stop landing on the ribbon and this fails.

Runs on CPU in the project venv (imports final_preview_renderer, which pulls
ultralytics but loads no model). Standalone:

    .venv/bin/python tests/test_restyle_render.py

or under pytest:

    pytest tests/test_restyle_render.py
"""

import gzip
import json
import os
import sys
import tempfile

import cv2
import numpy as np

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")

if SRC not in sys.path:
    sys.path.insert(0, SRC)

WIDTH, HEIGHT, FPS, FRAMES = 320, 180, 10.0, 8

CAMERA = {
    "horizon_v": 60.0,
    "vanish_u": 160.0,
    "focal_px": 250.0,
    "cam_height_m": 1.3,
}


def build_fixture(directory):
    clip_path = os.path.join(directory, "clip.mp4")
    writer = cv2.VideoWriter(
        clip_path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT)
    )

    for _ in range(FRAMES):
        writer.write(np.full((HEIGHT, WIDTH, 3), 128, dtype=np.uint8))

    writer.release()

    near_v, far_v = 160.0, 90.0
    centre = [[160.0, v] for v in range(int(near_v), int(far_v) - 1, -1)]

    person = {
        "poly": [[40, 60], [70, 60], [70, 120], [40, 120]],
        "box": [40, 60, 70, 120],
        "score": 0.8,
        "dist": 12.0,
        "tid": 1,
    }
    vehicle = dict(person, cls="car", box=[240, 70, 290, 120],
                   poly=[[240, 70], [290, 70], [290, 120], [240, 120]])

    geometry_path = os.path.join(directory, "vidtest_0_geometry.jsonl.gz")

    with gzip.open(geometry_path, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "type": "header", "version": 1,
            "clip_video": clip_path, "occlusion_video": None,
            "fps": FPS, "width": WIDTH, "height": HEIGHT, "frame_count": FRAMES,
            "camera": CAMERA, "resolution_scale": WIDTH / 1280.0,
            "chevron_speed_mps": 5.0,
        }) + "\n")

        for index in range(FRAMES):
            handle.write(json.dumps({
                "type": "frame", "i": index,
                "ramp": 1.0 if index >= 2 else 0.0,
                "label": "Vehicle behaviour adjusted" if index >= 2 else "",
                "ribbon": {"centre": centre, "near_v": near_v, "far_v": far_v},
                "persons": [person], "vehicles": [vehicle],
            }) + "\n")

    return clip_path, geometry_path


def style_variant(directory):
    with open(os.path.join(os.path.dirname(SRC), "styles", "default.json"),
              "r", encoding="utf-8") as handle:
        style = json.load(handle)

    # Loud, unambiguous changes: magenta ribbon body at high alpha, red
    # pedestrians, hidden vehicles, hidden chevrons.
    style["ribbon"]["body_color_bgr"] = [255, 0, 255]
    style["ribbon"]["body_alpha"] = 0.9
    style["ribbon"]["rails_color_bgr"] = [255, 0, 255]
    style["chevrons"]["visible"] = False
    style["pedestrians"]["color_bgr"] = [0, 0, 255]
    style["pedestrians"]["close_color_bgr"] = [0, 0, 255]
    style["pedestrians"]["fill_alpha"] = 0.6
    style["vehicles"]["visible"] = False

    path = os.path.join(directory, "loud.json")

    with open(path, "w", encoding="utf-8") as handle:
        json.dump(style, handle)

    return path


def frames_of(path):
    capture = cv2.VideoCapture(path)
    frames = []

    while True:
        ok, frame = capture.read()

        if not ok:
            break

        frames.append(frame)

    capture.release()

    return frames


def run():
    import restyle_render

    directory = tempfile.mkdtemp()
    clip_path, geometry_path = build_fixture(directory)
    loud_style = style_variant(directory)

    default_style = os.path.join(os.path.dirname(SRC), "styles", "default.json")
    out_default = restyle_render.restyle(
        geometry_path, default_style, os.path.join(directory, "out_a"), h264=False)
    out_loud = restyle_render.restyle(
        geometry_path, loud_style, os.path.join(directory, "out_b"), h264=False)

    frames_default = frames_of(out_default[1])   # vehicles variant
    frames_loud = frames_of(out_loud[1])

    assert len(frames_default) == FRAMES, (
        "expected %d frames, got %d" % (FRAMES, len(frames_default)))

    # Frame 0 has ramp 0: both outputs must be the untouched clip.
    assert np.abs(frames_default[0].astype(int) - 128).mean() < 6, (
        "ramp 0 frame must pass through undrawn")

    frame_d = frames_default[5].astype(int)
    frame_l = frames_loud[5].astype(int)

    # Ribbon body region: a point on the centreline, midway down the ribbon.
    v_probe, u_probe = 130, 160
    d_px = frame_d[v_probe, u_probe]
    l_px = frame_l[v_probe, u_probe]

    # Default body is blue-ish teal (BGR 245,200,90): B > R by a wide margin.
    assert d_px[0] > d_px[2] + 20, (
        "default ribbon colour missing at probe: %r" % (d_px,))
    # Loud body is magenta (255,0,255): B and R high, G low.
    assert l_px[0] > l_px[1] + 40 and l_px[2] > l_px[1] + 40, (
        "loud ribbon must be magenta at probe: %r" % (l_px,))

    # Pedestrian fill: default is orange-ish (B low, R high); loud is pure red
    # at fill_alpha 0.6, so red must dominate hard.
    pv, pu = 90, 55
    assert frame_l[pv, pu][2] > frame_l[pv, pu][0] + 40, (
        "loud pedestrian fill must be red at probe: %r" % (frame_l[pv, pu],))

    # Hidden vehicles: the vehicle box region must be untouched overlay-wise in
    # the loud style (just dimmed clip), but coloured in the default style.
    vv, vu = 95, 265
    default_delta = np.abs(frame_d[vv, vu] - 120).max()
    loud_delta = np.abs(frame_l[vv, vu] - 120).max()
    assert default_delta > 12, (
        "default style must draw the vehicle highlight: %r" % (frame_d[vv, vu],))
    assert loud_delta < 12, (
        "vehicles.visible=false must leave the vehicle region undrawn: %r"
        % (frame_l[vv, vu],))

    # Panel: the margins scale with resolution (24 px at the 1280 reference is
    # ~6 px here), so probe from the very corner. Dimmed base is ~120; the 0.42
    # darken pulls panel pixels to ~70, and the text is near-white.
    corner = frames_default[5][2:30, 2:160]
    assert corner.min() < 95, "panel darken missing (min %d)" % corner.min()
    assert corner.max() > 180, "panel text missing (max %d)" % corner.max()

    print("PASS  restyle compositor behavioural test")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
