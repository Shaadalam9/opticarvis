r"""Re-composite a rendered clip in a new overlay style, without the models.

The render burned its style into the pixels, but the geometry that produced it
was dumped per frame (overlay_geometry_dump.py). This tool replays that
geometry through the renderer's OWN drawing functions with different style
values, so a colour, opacity, width or speed change costs seconds of CPU
instead of minutes of GPU:

    .venv/bin/python src/restyle_render.py \
        --geometry workflow_outputs/overlay_geometry/<tag>_geometry.jsonl.gz \
        --style styles/default.json \
        --output-dir workflow_outputs/final_renders_restyled

Writes <tag>_restyled_<stylename>.mp4 and .._vehicles.mp4, mirroring the two
shipped variants. With --h264 (default) the outputs are H.264.

Reusing final_preview_renderer's functions is the point, not a convenience:
build_path_overlay, blend_path, draw_highlights and reveal_rows_for are the
exact code that rendered the original, so the default style reproduces it by
construction. The style file overrides the module's constants; the geometry
supplies everything the models and the temporal trackers decided.

Only the panel and the fixed-colour label variant are drawn locally: the
renderer hardcodes the panel's 0.42 darken and always labels in the element
colour, and both are style parameters here.
"""

import argparse
import gzip
import json
import os
import sys

import cv2
import numpy as np

SRC_DIR = os.path.dirname(os.path.abspath(__file__))

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import final_preview_renderer as R  # noqa: E402
from pipeline_common import transcode_h264  # noqa: E402


def load_style(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def read_geometry(path):
    header = None
    frames = []

    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)

            if record.get("type") == "header":
                header = record
            elif record.get("type") == "frame":
                frames.append(record)

    if header is None:
        raise SystemExit("No header record in " + path)

    frames.sort(key=lambda item: item["i"])

    return header, frames


def apply_style_to_renderer(style, width, height, camera):
    """Push the style's reference-scale values into the renderer module, then
    let its own resolution scaling do to them exactly what a render would."""
    ribbon = style.get("ribbon", {})
    chevrons = style.get("chevrons", {})
    labels = style.get("distance_labels", {})

    R.ROAD_PATH_COLOUR = tuple(ribbon.get("body_color_bgr", R.ROAD_PATH_COLOUR))
    R.BODY_ALPHA = float(ribbon.get("body_alpha", R.BODY_ALPHA))
    R.BODY_BLUR = int(ribbon.get("body_blur_px", R.BODY_BLUR))
    R.RAILS_ALPHA = float(ribbon.get("rails_alpha", R.RAILS_ALPHA))
    R.RAIL_STROKE_PX = int(ribbon.get("rail_stroke_px", R.RAIL_STROKE_PX))
    R.RIBBON_HALF_M = float(ribbon.get("half_width_m", R.RIBBON_HALF_M))
    R.FADE_IN_PX = float(ribbon.get("fade_in_px", R.FADE_IN_PX))
    R.FADE_OUT_PX = float(ribbon.get("fade_out_px", R.FADE_OUT_PX))
    R.CONTACT_SHADOW_ALPHA = (
        float(ribbon.get("shadow_alpha", R.CONTACT_SHADOW_ALPHA))
        if ribbon.get("shadow_visible", True) else 0.0
    )
    R.CONTACT_SHADOW_OFFSET_PX = int(ribbon.get("shadow_offset_px", R.CONTACT_SHADOW_OFFSET_PX))
    R.CONTACT_SHADOW_BLUR = int(ribbon.get("shadow_blur_px", R.CONTACT_SHADOW_BLUR))

    # The rails share the body colour constant in the renderer; a separate rail
    # colour is composited by build_path_overlay reading ROAD_PATH_COLOUR for
    # both, so differing rail colours are folded via the core-colour slot only
    # when chevrons are hidden. Keep them equal unless you know the layering.
    R.ROAD_PATH_CORE_COLOUR = tuple(chevrons.get("color_bgr", R.ROAD_PATH_CORE_COLOUR))
    R.DASH_ALPHA = (
        float(chevrons.get("alpha", R.DASH_ALPHA))
        if chevrons.get("visible", True) else 0.0
    )
    R.CHEVRON_PERIOD_M = float(chevrons.get("period_m", R.CHEVRON_PERIOD_M))
    R.CHEVRON_LEN_M = float(chevrons.get("length_m", R.CHEVRON_LEN_M))
    R.CHEVRON_HALF_M = float(chevrons.get("half_width_m", R.CHEVRON_HALF_M))

    R.SHOW_DISTANCE_LABEL = bool(labels.get("visible", True)) and labels.get("color_bgr") is None
    R.LABEL_FONT_SCALE = float(labels.get("font_scale", R.LABEL_FONT_SCALE))
    R.LABEL_TEXT_THICKNESS = int(labels.get("text_thickness_px", R.LABEL_TEXT_THICKNESS))
    R.LABEL_OUTLINE_THICKNESS = int(labels.get("outline_thickness_px", R.LABEL_OUTLINE_THICKNESS))
    R.TEXT_BACKGROUND = tuple(labels.get("outline_color_bgr", R.TEXT_BACKGROUND))

    R.apply_resolution_scaling(width, height)

    # The dump recorded the EFFECTIVE camera of the render (post scaling, post
    # per-clip calibration override); ribbon edges and chevron sizes must come
    # from the same numbers or the geometry lands in the wrong place.
    R.HORIZON_V = float(camera["horizon_v"])
    R.VANISH_U = float(camera["vanish_u"])
    R.CAM_FOCAL_PX = float(camera["focal_px"])
    R.CAM_HEIGHT_M = float(camera["cam_height_m"])


def ribbon_geometry_from_centre(centre, near_v, far_v, ordered=False):
    """Rebuild the full geometry dict the renderer's draw code expects.

    Edges derive from the centreline exactly as aimed_ribbon_geometry builds
    them, which is what makes the ribbon width a style parameter.

    ordered centrelines (version-2 dumps: future-anchored / direct geometry)
    are arc-ordered paths that may run across the image; horizontal edges
    would collapse a turn's band to a sliver, so they are rebuilt through the
    renderer's own ground-tangent builder -- same width style, correct rails.
    """
    if ordered:
        ground = []

        for u, v in np.asarray(centre, dtype=np.float64):
            point = R.ground_from_pixel(u, v)

            if point is not None:
                ground.append(point)

        if len(ground) >= 3:
            geometry = R.build_arc_ribbon_geometry(ground)

            if geometry is not None:
                return geometry

    centre_arr = np.asarray(centre, dtype=np.float32)
    v = centre_arr[:, 1]
    half = R.RIBBON_HALF_M * (v - R.HORIZON_V) / R.CAM_HEIGHT_M

    left = np.stack([centre_arr[:, 0] - half, v], axis=1).astype(np.float32)
    right = np.stack([centre_arr[:, 0] + half, v], axis=1).astype(np.float32)
    polygon = np.round(np.vstack([left, right[::-1]])).astype(np.int32)

    return {
        "traj": None,
        "centre": centre_arr,
        "left": left,
        "right": right,
        "polygon": polygon,
        "near_v": float(near_v),
        "far_v": float(far_v),
    }


def detections_from_records(records):
    detections = []

    for record in records:
        poly = record.get("poly")

        detections.append({
            "mask": np.asarray(poly, dtype=np.int32) if poly else None,
            "box": [int(v) for v in record["box"]],
            "draw_box": [int(v) for v in record["box"]],
            "spatial_score": float(record.get("score", 0.0)),
            "distance_m": record.get("dist"),
            "track_id": record.get("tid", -1),
            "class_name": record.get("cls", "person"),
            "class_id": 0,
            "confidence": 1.0,
        })

    return detections


def draw_fixed_colour_labels(frame, records, style, show_class):
    """The renderer always labels in the element colour; this is the override."""
    labels = style.get("distance_labels", {})
    colour = labels.get("color_bgr")

    if not labels.get("visible", True) or colour is None:
        return

    for record in records:
        parts = []

        if show_class and record.get("cls"):
            parts.append(record["cls"])

        if record.get("dist") is not None:
            parts.append("%.0f m" % record["dist"])

        label = " ".join(parts)

        if not label:
            continue

        box = record["box"]
        label_x = int(box[0])
        label_y = max(int(R.LABEL_TOP_CLAMP_PX), int(box[1]) - int(R.LABEL_ABOVE_BOX_PX))

        cv2.putText(frame, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX,
                    R.LABEL_FONT_SCALE, tuple(labels.get("outline_color_bgr", (0, 0, 0))),
                    R.LABEL_OUTLINE_THICKNESS, cv2.LINE_AA)
        cv2.putText(frame, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX,
                    R.LABEL_FONT_SCALE, tuple(colour),
                    R.LABEL_TEXT_THICKNESS, cv2.LINE_AA)


def draw_panel(frame, text, style):
    """Local panel: the renderer hardcodes the 0.42 darken this styles."""
    panel = style.get("panel", {})

    if not panel.get("visible", True) or not text:
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = R.PANEL_FONT_SCALE
    thickness = R.PANEL_TEXT_THICKNESS
    margin = int(R.PANEL_MARGIN_PX)
    padding_x = int(R.PANEL_PADDING_X_PX)
    padding_y = int(R.PANEL_PADDING_Y_PX)

    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
    height, width = frame.shape[:2]
    x1, y1 = margin, margin
    x2 = min(width - margin, x1 + text_width + 2 * padding_x)
    y2 = y1 + text_height + 2 * padding_y

    roi = frame[y1:y2, x1:x2]
    dark = np.full_like(roi, tuple(panel.get("background_color_bgr", (0, 0, 0))))
    alpha = float(panel.get("background_alpha", 0.42))
    cv2.addWeighted(dark, alpha, roi, 1.0 - alpha, 0, roi)
    cv2.putText(frame, text, (x1 + padding_x, y2 - padding_y), font, font_scale,
                tuple(panel.get("text_color_bgr", (255, 255, 255))),
                thickness, cv2.LINE_AA)


def restyle(geometry_path, style_path, output_dir, h264=True):
    style = load_style(style_path)
    header, frames = read_geometry(geometry_path)

    width = int(header["width"])
    height = int(header["height"])
    fps = float(header["fps"])
    camera = header["camera"]

    apply_style_to_renderer(style, width, height, camera)

    clip_path = header["clip_video"]

    if not os.path.isfile(clip_path):
        # Old dumps recorded the path as typed on the render CLI, which may be
        # relative to src/. The canonical clip location is recoverable from the
        # basename alone.
        from pipeline_common import ALPAMAYO_OUTPUTS

        fallback = os.path.join(
            ALPAMAYO_OUTPUTS, "crowd_clips", os.path.basename(clip_path)
        )

        if os.path.isfile(fallback):
            clip_path = fallback

    clip = cv2.VideoCapture(clip_path)

    if not clip.isOpened():
        raise SystemExit("Cannot open clip video: " + clip_path)

    occlusion_video = None

    if header.get("occlusion_video") and os.path.isfile(header["occlusion_video"]):
        occlusion_video = cv2.VideoCapture(header["occlusion_video"])

    style_name = os.path.splitext(os.path.basename(style_path))[0]
    tag = os.path.basename(geometry_path).replace("_geometry.jsonl.gz", "")
    os.makedirs(output_dir, exist_ok=True)

    out_plain = os.path.join(output_dir, "%s_restyled_%s.mp4" % (tag, style_name))
    out_vehicles = os.path.join(output_dir, "%s_restyled_%s_vehicles.mp4" % (tag, style_name))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_plain, fourcc, fps, (width, height))
    writer_vehicles = cv2.VideoWriter(out_vehicles, fourcc, fps, (width, height))

    background = style.get("background", {})
    dim_alpha = float(background.get("dim_alpha", 0.06))
    pedestrians = style.get("pedestrians", {})
    vehicles_style = style.get("vehicles", {})
    labels_visible_inherit = R.SHOW_DISTANCE_LABEL
    chevron_speed = float(style.get("chevrons", {}).get("speed_mps", 5.0))
    ribbon_visible = style.get("ribbon", {}).get("visible", True)

    frame_lookup = {record["i"]: record for record in frames}
    total = int(header.get("frame_count") or len(frames))

    for index in range(total):
        ok, orig = clip.read()

        if not ok:
            break

        occlusion = None

        if occlusion_video is not None:
            got, occ_frame = occlusion_video.read()

            if got:
                occlusion = occ_frame[:, :, 0].astype(np.float32) / 255.0

        record = frame_lookup.get(index)
        ramp_value = float(record["ramp"]) if record else 0.0

        if record is None or ramp_value <= 0.001:
            writer.write(orig)
            writer_vehicles.write(orig)
            continue

        label_text = record.get("label", "")
        base = (orig.astype(np.float32) * (1.0 - dim_alpha * ramp_value)).astype(np.uint8)

        ribbon = record.get("ribbon")

        if ribbon_visible and ribbon is not None:
            geometry = ribbon_geometry_from_centre(
                ribbon["centre"], ribbon["near_v"], ribbon["far_v"],
                ordered=bool(ribbon.get("ordered")),
            )
            # Prefer the phase the render actually used (travelled metres);
            # recomputing from index/fps un-grounds street-nailed marks.
            if ribbon.get("phase_m") is not None:
                phase_m = float(ribbon["phase_m"])
            else:
                phase_m = (index / fps) * chevron_speed if fps else 0.0
            overlay = R.build_path_overlay(geometry, height, width, phase_m)
            reveal = R.reveal_rows_for(ramp_value, geometry["near_v"],
                                       geometry["far_v"], height)
            R.blend_path(base, overlay, occlusion, gain=reveal)

        persons = detections_from_records(record.get("persons", []))
        vehicle_dets = detections_from_records(record.get("vehicles", []))

        def compose(with_vehicles):
            layer = base.copy()

            if pedestrians.get("visible", True):
                R.HIGHLIGHT_FILL_ALPHA = float(pedestrians.get("fill_alpha", 0.14))
                R.CONTOUR_THICKNESS = max(1, int(round(
                    float(pedestrians.get("contour_thickness_px", 2)) * (width / 1280.0))))
                R.SHOW_DISTANCE_LABEL = labels_visible_inherit
                R.draw_highlights(
                    layer, persons,
                    tuple(pedestrians.get("close_color_bgr", (0, 160, 255))),
                    tuple(pedestrians.get("color_bgr", (0, 220, 255))),
                    show_class=False,
                )
                draw_fixed_colour_labels(layer, record.get("persons", []), style, False)

            if with_vehicles and vehicles_style.get("visible", True):
                R.HIGHLIGHT_FILL_ALPHA = float(vehicles_style.get("fill_alpha", 0.14))
                R.CONTOUR_THICKNESS = max(1, int(round(
                    float(vehicles_style.get("contour_thickness_px", 2)) * (width / 1280.0))))
                R.SHOW_DISTANCE_LABEL = labels_visible_inherit
                R.draw_highlights(
                    layer, vehicle_dets,
                    tuple(vehicles_style.get("close_color_bgr", (60, 200, 40))),
                    tuple(vehicles_style.get("color_bgr", (120, 230, 60))),
                    show_class=True,
                )
                draw_fixed_colour_labels(layer, record.get("vehicles", []), style, True)

            draw_panel(layer, label_text, style)

            if ramp_value >= 0.999:
                return layer

            return (base.astype(np.float32) * (1.0 - ramp_value)
                    + layer.astype(np.float32) * ramp_value).astype(np.uint8)

        writer.write(compose(False))
        writer_vehicles.write(compose(True))

    clip.release()
    writer.release()
    writer_vehicles.release()

    if occlusion_video is not None:
        occlusion_video.release()

    outputs = [out_plain, out_vehicles]

    if h264:
        for path in outputs:
            temp = path + ".h264.tmp.mp4"

            try:
                transcode_h264(path, temp, remove_source=False)
                os.replace(temp, path)
            except Exception as error:
                print("H.264 transcode failed for %s (%s); keeping mp4v."
                      % (path, type(error).__name__))

                if os.path.isfile(temp):
                    os.remove(temp)

    return outputs


def main():
    parser = argparse.ArgumentParser(
        description="Re-composite a rendered clip with a different overlay style.",
    )
    parser.add_argument("--geometry", required=True,
                        help="<tag>_geometry.jsonl.gz from a render")
    parser.add_argument("--style", default=os.path.join(
        os.path.dirname(SRC_DIR), "styles", "default.json"))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--no-h264", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir

    if output_dir is None:
        from pipeline_common import WORKFLOW_OUTPUTS

        output_dir = os.path.join(WORKFLOW_OUTPUTS, "final_renders_restyled")

    outputs = restyle(args.geometry, args.style, output_dir, h264=not args.no_h264)

    print("")

    for path in outputs:
        print("Wrote", path)


if __name__ == "__main__":
    main()
