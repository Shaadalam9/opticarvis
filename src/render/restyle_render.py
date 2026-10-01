r"""Re-composite a rendered clip under another study render config, without models.

A render burns its style into the pixels, but final_preview_renderer.py also
dumps the geometry the models produced for every frame (overlay_geometry_dump.py:
ribbon centreline, selected detections, occlusion map). This tool replays that
geometry through the renderer's OWN functions under a different render config
(the study's mask_alpha / trajectory_alpha / background_dim_alpha / palette_id),
so a style variant costs about a minute of CPU instead of ten of GPU:

    python src/render/restyle_render.py \
        --geometry workflow_outputs/overlay_geometry/<tag>_geometry.jsonl.gz \
        --config configs/render_default.json --name default

It follows render_video's per-frame steps exactly: dim_background, blend_path,
pedestrian highlights, the persons-only output with the text panel, then vehicle
highlights and the panel for the +vehicles output. The config goes through the
renderer's own apply_render_config_to_globals, then the same calibration and
resolution scaling as a render, so a variant differs from the shipped render
only where the config differs. Two inputs are stored lossily and bound the
match: the occlusion map (mp4) and the ribbon edges, which are rebuilt from the
stored centreline.

--sweep writes the study's one-at-a-time sweep (10 variants) plus a contact
sheet comparing them on one frame.
"""

import argparse
import gzip
import importlib
import json
import os
import sys

import cv2
import numpy as np


# Make src/ and every src/<group>/ importable (see src/_paths.py).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import _paths  # noqa: E402,F401


# The study's one-at-a-time sweep: the default, each continuous parameter at its
# range limits (study_service/space.py), and the other three palettes.
DEFAULT_CONFIG = {
    "mask_alpha": 0.14,
    "trajectory_alpha": 0.55,
    "background_dim_alpha": 0.06,
    "palette_id": 0,
}
SWEEP = [
    ("default", {}),
    ("mask_alpha_0.00", {"mask_alpha": 0.0}),
    ("mask_alpha_0.70", {"mask_alpha": 0.7}),
    ("trajectory_alpha_0.00", {"trajectory_alpha": 0.0}),
    ("trajectory_alpha_1.00", {"trajectory_alpha": 1.0}),
    ("background_dim_0.00", {"background_dim_alpha": 0.0}),
    ("background_dim_0.40", {"background_dim_alpha": 0.4}),
    ("palette_1", {"palette_id": 1}),
    ("palette_2", {"palette_id": 2}),
    ("palette_3", {"palette_id": 3}),
]


def read_geometry(path):
    header = None
    frames = {}

    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)

            if record.get("type") == "header":
                header = record
            elif record.get("type") == "frame":
                frames[int(record["i"])] = record

    if header is None:
        raise SystemExit("No header record in " + path)

    return header, frames


def tag_of(geometry_path):
    return os.path.basename(geometry_path).replace("_geometry.jsonl.gz", "")


def load_renderer(tag, config):
    """The renderer module configured exactly as a render of this clip would be.

    The renderer reads the clip identity (and so its calibration file) from the
    environment at import, so the job's VIDEO_ID / start are set first and the
    module is reloaded per variant: every variant starts from pristine globals.
    """
    video_id, start = tag.rsplit("_", 1)
    os.environ["OPTICARVIS_VIDEO_ID"] = video_id
    os.environ["OPTICARVIS_SEGMENT_START_S"] = start

    import pipeline_common
    importlib.reload(pipeline_common)
    import final_preview_renderer
    renderer = importlib.reload(final_preview_renderer)

    # Start from the render config the pipeline renders with (RENDER_CONFIG,
    # default configs/render_default.json) so the fixed settings - contour
    # thickness, label scale - match the shipped render; only the swept keys
    # are overridden.
    full = dict(DEFAULT_CONFIG)
    full.update(renderer.load_render_config())
    full.update(config)
    renderer.apply_render_config_to_globals(full)
    return renderer


def prepare_renderer(renderer, header):
    """render_video's own setup: calibration, then resolution scaling."""
    renderer.apply_calibration_overrides()
    renderer.apply_resolution_scaling(int(header["width"]), int(header["height"]))

    # The dump recorded the EFFECTIVE camera of the render. Geometry and marks
    # must use the same numbers; a mismatch means the calibration changed since.
    camera = header["camera"]
    for name, key in (("HORIZON_V", "horizon_v"), ("VANISH_U", "vanish_u"),
                      ("CAM_FOCAL_PX", "focal_px"), ("CAM_HEIGHT_M", "cam_height_m")):
        recorded = float(camera[key])
        if abs(float(getattr(renderer, name)) - recorded) > 1e-3:
            print("WARNING: %s is %.3f now but %.3f in the render; using the render's."
                  % (name, float(getattr(renderer, name)), recorded))
        setattr(renderer, name, recorded)


def ribbon_geometry(renderer, ribbon):
    """Rebuild the geometry dict the draw code expects from the stored centreline."""
    centre = np.asarray(ribbon["centre"], dtype=np.float32)

    if ribbon.get("ordered"):
        ground = [point for point in (renderer.ground_from_pixel(u, v) for u, v in centre)
                  if point is not None]
        if len(ground) >= 3:
            geometry = renderer.build_arc_ribbon_geometry(ground)
            if geometry is not None:
                return geometry

    v = centre[:, 1]
    half = renderer.RIBBON_HALF_M * (v - renderer.HORIZON_V) / renderer.CAM_HEIGHT_M
    left = np.stack([centre[:, 0] - half, v], axis=1).astype(np.float32)
    right = np.stack([centre[:, 0] + half, v], axis=1).astype(np.float32)

    return {
        "traj": None,
        "centre": centre,
        "left": left,
        "right": right,
        "polygon": np.round(np.vstack([left, right[::-1]])).astype(np.int32),
        "near_v": float(ribbon["near_v"]),
        "far_v": float(ribbon["far_v"]),
    }


def detections(records):
    return [
        {
            "mask": np.asarray(record["poly"], dtype=np.int32) if record.get("poly") else None,
            "box": [int(value) for value in record["box"]],
            "draw_box": [int(value) for value in record["box"]],
            "spatial_score": float(record.get("score", 0.0)),
            "distance_m": record.get("dist"),
            "track_id": record.get("tid", -1),
            "class_name": record.get("cls", "person"),
        }
        for record in records
    ]


def compose_frame(renderer, orig, record, occlusion, chevron_speed_mps, index, fps):
    """One frame, in render_video's order. Returns (persons_only, with_vehicles)."""
    height, width = orig.shape[:2]
    base = renderer.dim_background(orig)

    ribbon = record.get("ribbon")
    if ribbon is not None:
        geometry = ribbon_geometry(renderer, ribbon)
        # The phase the render used (travelled metres); recomputing it from the
        # frame index would un-ground marks the render nailed to the street.
        phase_m = ribbon.get("phase_m")
        if phase_m is None:
            phase_m = (index / fps) * chevron_speed_mps if fps else 0.0
        overlay = renderer.build_path_overlay(geometry, height, width, float(phase_m))
        renderer.blend_path(base, overlay, occlusion)

    label = record.get("label", "")
    renderer.draw_highlights(base, detections(record.get("persons", [])),
                             renderer.BOX_COLOUR_CLOSE, renderer.BOX_COLOUR, show_class=False)
    persons_only = base.copy()
    renderer.draw_text_panel(persons_only, label)

    renderer.draw_highlights(base, detections(record.get("vehicles", [])),
                             renderer.VEHICLE_BOX_COLOUR_CLOSE, renderer.VEHICLE_BOX_COLOUR,
                             show_class=True)
    renderer.draw_text_panel(base, label)
    return persons_only, base


def resolve_clip(header):
    path = header["clip_video"]
    if os.path.isfile(path):
        return path
    from pipeline_common import ALPAMAYO_OUTPUTS
    fallback = os.path.join(ALPAMAYO_OUTPUTS, "crowd_clips", os.path.basename(path))
    if os.path.isfile(fallback):
        return fallback
    raise SystemExit("Cannot find the clip video: " + path)


def restyle(geometry_path, config, name, output_dir, both=False, h264=True, sample_frame=None):
    """Write <output_dir>/<name>_vehicles.mp4 (and <name>.mp4 with both=True).

    Returns (output paths, the composed +vehicles frame at sample_frame or None).
    """
    header, frames = read_geometry(geometry_path)
    tag = tag_of(geometry_path)
    renderer = load_renderer(tag, config)
    prepare_renderer(renderer, header)

    width, height, fps = int(header["width"]), int(header["height"]), float(header["fps"])
    speed = float(header.get("chevron_speed_mps", renderer.CHEVRON_SPEED_MPS))

    clip = cv2.VideoCapture(resolve_clip(header))
    occlusion_video = None
    if header.get("occlusion_video") and os.path.isfile(header["occlusion_video"]):
        occlusion_video = cv2.VideoCapture(header["occlusion_video"])

    os.makedirs(output_dir, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_vehicles = os.path.join(output_dir, name + "_vehicles.mp4")
    out_persons = os.path.join(output_dir, name + ".mp4")
    # Frames go to a ".writing" file that is renamed only once complete. An
    # mp4's index (moov atom) is written at close, so a half-written file
    # under the final name looks finished but will not open.
    writing_vehicles = in_progress_path(out_vehicles)
    writing_persons = in_progress_path(out_persons)
    writer_vehicles = cv2.VideoWriter(writing_vehicles, fourcc, fps, (width, height))
    writer_persons = cv2.VideoWriter(writing_persons, fourcc, fps, (width, height)) if both else None

    sample = None
    index = 0
    while True:
        ok, orig = clip.read()
        if not ok:
            break

        occlusion = None
        if occlusion_video is not None:
            got, occlusion_frame = occlusion_video.read()
            if got:
                occlusion = occlusion_frame[:, :, 0].astype(np.float32) / 255.0

        record = frames.get(index)
        if record is None:
            persons_only = with_vehicles = orig
        else:
            persons_only, with_vehicles = compose_frame(
                renderer, orig, record, occlusion, speed, index, fps)

        writer_vehicles.write(with_vehicles)
        if writer_persons is not None:
            writer_persons.write(persons_only)
        if sample_frame is not None and index == sample_frame:
            sample = with_vehicles.copy()
        index += 1

    clip.release()
    writer_vehicles.release()
    if writer_persons is not None:
        writer_persons.release()
    if occlusion_video is not None:
        occlusion_video.release()

    outputs = [out_vehicles] + ([out_persons] if both else [])
    for path in outputs:
        finish_video(in_progress_path(path), path, h264)

    return outputs, sample


def in_progress_path(final_path):
    stem, extension = os.path.splitext(final_path)
    return stem + ".writing" + extension


def finish_video(writing_path, final_path, h264):
    """Move a completed video to its final name, as H.264 when possible.

    The final name only ever appears through os.replace, so any file carrying
    it is complete and playable.
    """
    if h264:
        from pipeline_common import transcode_h264
        encoding_path = in_progress_path(final_path).replace(".writing", ".encoding")
        try:
            transcode_h264(writing_path, encoding_path, remove_source=False)
            os.replace(encoding_path, final_path)
            os.remove(writing_path)
            return
        except Exception as error:
            print("H.264 transcode failed for %s (%s); keeping mp4v."
                  % (final_path, type(error).__name__))
            if os.path.isfile(encoding_path):
                os.remove(encoding_path)
    os.replace(writing_path, final_path)


def contact_sheet(samples, path, columns=5, thumb_width=480):
    """One frame per variant, labelled, in a grid."""
    tiles = []
    for name, frame in samples:
        height, width = frame.shape[:2]
        tile = cv2.resize(frame, (thumb_width, int(round(height * thumb_width / width))),
                          interpolation=cv2.INTER_AREA)
        cv2.rectangle(tile, (0, 0), (thumb_width, 30), (0, 0, 0), -1)
        cv2.putText(tile, name, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)

    blank = np.zeros_like(tiles[0])
    while len(tiles) % columns:
        tiles.append(blank)
    rows = [np.hstack(tiles[i:i + columns]) for i in range(0, len(tiles), columns)]
    cv2.imwrite(path, np.vstack(rows))
    return path


def pick_sample_frame(geometry_path):
    """A frame with both a ribbon and a highlighted pedestrian, near mid-clip."""
    _header, frames = read_geometry(geometry_path)
    good = sorted(i for i, r in frames.items() if r.get("ribbon") and r.get("persons"))
    pool = good or sorted(frames)
    return pool[len(pool) // 2] if pool else 0


def main():
    parser = argparse.ArgumentParser(description="Re-composite a render under another render config.")
    parser.add_argument("--geometry", required=True, help="<tag>_geometry.jsonl.gz from a render")
    parser.add_argument("--config", help="a render config JSON (render_default.json keys)")
    parser.add_argument("--name", default="restyled")
    parser.add_argument("--sweep", action="store_true",
                        help="write the 10-variant one-at-a-time sweep and a contact sheet")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--both", action="store_true", help="also write the persons-only output")
    parser.add_argument("--no-h264", action="store_true")
    args = parser.parse_args()

    from pipeline_common import WORKFLOW_OUTPUTS
    tag = tag_of(args.geometry)
    output_dir = args.output_dir or os.path.join(WORKFLOW_OUTPUTS, "render_variants", tag)

    if args.sweep:
        sample_index = pick_sample_frame(args.geometry)
        samples = []
        for name, overrides in SWEEP:
            outputs, sample = restyle(args.geometry, overrides, name, output_dir,
                                      both=args.both, h264=not args.no_h264,
                                      sample_frame=sample_index)
            samples.append((name, sample))
            print("Wrote", ", ".join(outputs))
        sheet = contact_sheet([s for s in samples if s[1] is not None],
                              os.path.join(output_dir, "contact_sheet.jpg"))
        print("Contact sheet (frame %d):" % sample_index, sheet)
        return

    config = {}
    if args.config:
        with open(args.config, "r", encoding="utf-8-sig") as handle:
            config = {k: v for k, v in json.load(handle).items() if k in DEFAULT_CONFIG}
    outputs, _ = restyle(args.geometry, config, args.name, output_dir,
                         both=args.both, h264=not args.no_h264)
    for path in outputs:
        print("Wrote", path)


if __name__ == "__main__":
    main()
