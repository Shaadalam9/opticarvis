r"""Render a clip with the overlay temporally gated and animated.

Usage:
    python render_timeline_clip.py <clip.mp4> <timeline.json> [tag] [ego_track.json] [vo_track.json]

Arguments:
    clip.mp4
        Clip to render.

    timeline.json
        Sliding window Gemma 4 gate timeline. This drives when the overlay
        animates on and off.

    tag
        Short label folded into the output filename. Default is "timeline".

    ego_track.json
        Optional ego motion track from ego_motion.py. Only used when
        OPTICARVIS_EGO_LOOKAHEAD=1. Otherwise it is ignored by the renderer with
        a warning.

    vo_track.json
        Optional future frame VO track from ego_trajectory.py. Only used when
        OPTICARVIS_VO_TRAJECTORY=1. Otherwise it is ignored by the renderer with
        a warning.

    Pass "" for a slot you want to skip, for example:
        python render_timeline_clip.py <clip.mp4> <timeline.json> <tag> "" <vo_track.json>

Everything about the run is derived from the consolidated OptiCarVis layout:
the output directory comes from pipeline_common.workflow_path, the filename stem
comes from the clip being rendered, the H.264 delivery encode runs in process,
and the effective configuration is recorded into the workflow state file so a
shipped video can be traced back to the settings that made it.
"""

import os
import sys

import final_preview_renderer as R
from pipeline_common import (
    clip_stem,
    ensure_dir,
    normalise_path,
    read_json,
    transcode_h264,
    workflow_path,
)


def parse_args(argv):
    if len(argv) < 3:
        print(__doc__)
        raise SystemExit(2)

    optional = [(argv[index] if len(argv) > index else "") for index in (3, 4, 5)]

    return {
        "clip": normalise_path(argv[1]),
        "timeline": normalise_path(argv[2]),
        "tag": optional[0] or "timeline",
        "ego_track": normalise_path(optional[1]) if optional[1] else None,
        "vo_track": normalise_path(optional[2]) if optional[2] else None,
    }


def effective_config():
    """Return the settings that actually shaped this render for the state record."""
    tracked_env = (
        "OPTICARVIS_LANE_SOURCE",
        "OPTICARVIS_VO_TRAJECTORY",
        "OPTICARVIS_EGO_LOOKAHEAD",
        "OPTICARVIS_VIDEO_ID",
        "OPTICARVIS_SEGMENT_START_S",
        "OPTICARVIS_CLIP_VIDEO",
        "OPTICARVIS_PROJECT_ROOT",
    )

    return {
        "lane_source": R.LANE_SOURCE,
        "lane_centering": bool(R.USE_LANE_CENTERING),
        "vo_turn_shaping": bool(R.USE_VO_TRAJECTORY),
        "ego_lookahead": bool(R.USE_EGO_LOOKAHEAD),
        "road_segmentation": bool(R.USE_ROAD_SEGMENTATION),
        "depth_labels": bool(R.USE_DEPTH_DISTANCE),
        "ribbon_half_m": R.RIBBON_HALF_M,
        "vo_turn_lat_lo_m": R.VO_TURN_LAT_LO,
        "vo_turn_lat_hi_m": R.VO_TURN_LAT_HI,
        "env": {
            key: os.environ.get(key)
            for key in tracked_env
            if os.environ.get(key) is not None
        },
    }


def output_paths(clip_path, tag):
    out_dir = workflow_path("final_renders")
    ensure_dir(out_dir)

    stem = "%s_%s_%s" % (clip_stem(clip_path), R.RENDER_VARIANT, tag)

    return {
        "out_dir": out_dir,
        "stem": stem,
        "master": workflow_path("final_renders", stem + ".mp4"),
        "master_vehicles": workflow_path("final_renders", stem + "_vehicles.mp4"),
        "delivered": workflow_path("final_renders", stem + "_h264.mp4"),
        "delivered_vehicles": workflow_path("final_renders", stem + "_vehicles_h264.mp4"),
    }


def main():
    args = parse_args(sys.argv)
    paths = output_paths(args["clip"], args["tag"])

    R.INPUT_VIDEO = args["clip"]
    R.OUTPUT_VIDEO = paths["master"]
    R.OUTPUT_VIDEO_VEHICLES = paths["master_vehicles"]

    timeline = read_json(args["timeline"], "gate timeline")
    ego = read_json(args["ego_track"], "ego motion track") if args["ego_track"] else None
    vo = read_json(args["vo_track"], "VO trajectory track") if args["vo_track"] else None

    summary = R.render_video_timeline(timeline, ego_track=ego, vo_track=vo)

    delivered = transcode_h264(paths["master"], paths["delivered"])
    delivered_vehicles = transcode_h264(paths["master_vehicles"], paths["delivered_vehicles"])

    summary["config"] = effective_config()
    summary["clip_video"] = args["clip"]
    summary["timeline_json"] = args["timeline"]
    summary["ego_track_json"] = args["ego_track"]
    summary["vo_track_json"] = args["vo_track"]
    summary["delivered_video"] = delivered or paths["master"]
    summary["delivered_video_vehicles"] = delivered_vehicles or paths["master_vehicles"]

    R.record_timeline_render(args["tag"], summary)

    print("")
    print("Temporal gated render complete")
    print("==============================")
    print("on_frames: %s / %s" % (summary["on_frames"], summary["frame_count"]))
    print("Output:", summary["delivered_video"])
    print("Output vehicles:", summary["delivered_video_vehicles"])


if __name__ == "__main__":
    main()
