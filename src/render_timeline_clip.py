r"""Render a clip with the overlay temporally gated + animated.

Usage:
    python render_timeline_clip.py <clip.mp4> <timeline.json> [tag] [ego_track.json] [vo_track.json]

  clip.mp4        the clip to render.
  timeline.json   sliding-window Gemma 4 gate timeline; drives when the overlay
                  animates on and off.
  tag             short label folded into the output filename (default "timeline").
  ego_track.json  optional ego-motion track (ego_motion.py). Only used when
                  OPTICARVIS_EGO_LOOKAHEAD=1; ignored (with a warning) otherwise.
  vo_track.json   optional future-frame VO track (ego_trajectory.py), which shapes
                  the ribbon through genuine turns. Only used when
                  OPTICARVIS_VO_TRAJECTORY=1; ignored (with a warning) otherwise.
  Pass "" for a slot you want to skip, e.g. ... <tag> "" <vo_track.json>.

Everything about the run is derived, not hardcoded: the output directory comes
from pipeline_common.workflow_path, the filename stem from the clip being
rendered (so a second clip cannot overwrite the first), the H.264 delivery encode
runs in-process, and the effective configuration is recorded into the workflow
state file so a shipped video can be traced back to the settings that made it.
"""

import os
import sys

import final_preview_renderer as R
from pipeline_common import (
    clip_stem,
    ensure_dir,
    read_json,
    transcode_h264,
    workflow_path,
)


def parse_args(argv):
    if len(argv) < 3:
        print(__doc__)
        raise SystemExit(2)
    optional = [(argv[i] if len(argv) > i else "") for i in (3, 4, 5)]
    return {
        "clip": argv[1],
        "timeline": argv[2],
        "tag": optional[0] or "timeline",
        "ego_track": optional[1] or None,
        "vo_track": optional[2] or None,
    }


def effective_config():
    """The settings that actually shape this render, for the state record."""
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
            for key in ("OPTICARVIS_LANE_SOURCE", "OPTICARVIS_VO_TRAJECTORY",
                        "OPTICARVIS_EGO_LOOKAHEAD", "OPTICARVIS_VIDEO_ID",
                        "OPTICARVIS_SEGMENT_START_S")
            if os.environ.get(key) is not None
        },
    }


def main():
    args = parse_args(sys.argv)

    out_dir = workflow_path("final_renders")
    ensure_dir(out_dir)

    stem = "%s_%s_%s" % (clip_stem(args["clip"]), R.RENDER_VARIANT, args["tag"])
    master = out_dir + "/" + stem + ".mp4"
    master_vehicles = out_dir + "/" + stem + "_vehicles.mp4"

    R.INPUT_VIDEO = args["clip"]
    R.OUTPUT_VIDEO = master
    R.OUTPUT_VIDEO_VEHICLES = master_vehicles

    timeline = read_json(args["timeline"], "gate timeline")
    ego = read_json(args["ego_track"], "ego-motion track") if args["ego_track"] else None
    vo = read_json(args["vo_track"], "VO trajectory track") if args["vo_track"] else None

    summary = R.render_video_timeline(timeline, ego_track=ego, vo_track=vo)

    # Delivery encode, in-process so the artefact is reproducible from one command.
    delivered = transcode_h264(master, out_dir + "/" + stem + "_h264.mp4")
    delivered_vehicles = transcode_h264(
        master_vehicles, out_dir + "/" + stem + "_vehicles_h264.mp4")

    summary["config"] = effective_config()
    summary["clip_video"] = args["clip"]
    summary["timeline_json"] = args["timeline"]
    summary["ego_track_json"] = args["ego_track"]
    summary["vo_track_json"] = args["vo_track"]
    summary["delivered_video"] = delivered or master
    summary["delivered_video_vehicles"] = delivered_vehicles or master_vehicles

    R.record_timeline_render(args["tag"], summary)

    print("")
    print("Temporal-gated render complete")
    print("on_frames: %s / %s" % (summary["on_frames"], summary["frame_count"]))
    print("Output:", summary["delivered_video"])
    print("Output (vehicles):", summary["delivered_video_vehicles"])


if __name__ == "__main__":
    main()
