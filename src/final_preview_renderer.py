r"""Road grounded final preview renderer for the OptiCarVis workflow.

The ribbon is drawn through a calibrated flat-ground pinhole camera, so it lies on
the road with correct perspective foreshortening.

WHERE THE RIBBON'S SHAPE COMES FROM (the contract - keep this accurate):
  * lateral placement: ego-lane detection (UFLDv2 lane instances by default; see
    LANE_SOURCE) sets the ribbon's NEAR end, i.e. which lane the car is in.
  * direction: the FAR end aims at the vanishing point, because a straight path in
    the ego lane converges there. This is deliberate. Earlier versions steered the
    far end from the road-mask centroid and then from the lane detector's far
    column; both tilted the ribbon off the real path, so the ribbon pointed where
    the car was not driving. Do not reintroduce either.
  * curvature: ONLY from the future-frame visual odometry (ego_trajectory.py), and
    only while it confidently reports a genuine turn (see USE_VO_TRAJECTORY and
    vo_turn_weight). A consequence: gentle curves render as near-straight.
The Alpamayo planned trajectory is still supported as a static fallback
(load_path_geometry) but is not the live source for the road-aimed ribbon.

Pedestrians are detected and tracked with YOLO26 + ByteTrack; their segmentation
masks occlude the ribbon (it passes behind them) and drive contour highlights.
Highlight selection is locked to track IDs and the boxes are smoothed over time so
the overlay does not flicker.

Run the full render, from the repo root in the project venv:
    python src/final_preview_renderer.py

Tune the camera scalars quickly on one still frame (no video, no tracking):
    python src/final_preview_renderer.py --calibrate <input.jpg> <output.png>
The calibration overlay draws the horizon line, the vanishing point, and metre
distance ticks so HORIZON_V / VANISH_U / CAM_FOCAL_PX / CAM_HEIGHT_M can be
adjusted by eye.
"""

import json
import math
import os
import sys

import cv2
import numpy as np
from ultralytics import YOLO

from pipeline_common import (
    CLIP_VIDEO,
    STATE_JSON,
    YOLO_SEG_MODEL,
    read_json,
    write_json,
    clamp,
    ensure_dir,
    segment_tag,
    workflow_path,
)


# Single source of truth for the render variant, used in the output filename,
# the workflow-state keys, and the renderer id.
RENDER_VARIANT = "roadline_v3"

EFFECT_PLAN_JSON = workflow_path("mirage", segment_tag() + "_effect_plan.json")
ALPAMAYO_CONTEXT_JSON = workflow_path(
    "alpamayo_traces",
    segment_tag() + "_alpamayo_context.json",
)

INPUT_VIDEO = CLIP_VIDEO

OUTPUT_DIR = workflow_path("final_renders")

OUTPUT_VIDEO = workflow_path(
    "final_renders",
    segment_tag() + "_mirage_preview_" + RENDER_VARIANT + ".mp4",
)

# Second output that also highlights nearby vehicles.
OUTPUT_VIDEO_VEHICLES = workflow_path(
    "final_renders",
    segment_tag() + "_mirage_preview_" + RENDER_VARIANT + "_vehicles.mp4",
)

# Swap via OPTICARVIS_YOLO_SEG_MODEL; see pipeline_common "Models".
MODEL_NAME = YOLO_SEG_MODEL
TRACKER_NAME = os.environ.get("OPTICARVIS_TRACKER", "bytetrack.yaml")
IMAGE_SIZE = 1280
# Low gate: the source is 720p, so most missed pedestrians are small/occluded
# people the detector sees but scores low, not a resolution problem. 0.15 caught
# ~30% more real people on the calibration frames with no visible false
# positives; ByteTrack + the spatial-score selection keep highlights precise.
CONFIDENCE_THRESHOLD = 0.15

PERSON_CLASS_ID = 0
# Common vehicle classes (COCO): bicycle, car, motorcycle, bus, truck.
VEHICLE_CLASS_IDS = [1, 2, 3, 5, 7]
# Classes whose masks should occlude the road ribbon (they sit in front of it).
OCCLUDER_CLASS_IDS = [PERSON_CLASS_ID] + VEHICLE_CLASS_IDS

MAX_PEDESTRIANS_TO_RENDER = 5
MAX_VEHICLES_TO_RENDER = 4
COCO_VEHICLE_NAMES = {1: "bike", 2: "car", 3: "moto", 5: "bus", 7: "truck"}
MIN_PROXIMITY_SCORE = 0.22
CLOSE_BOX_SCORE = 0.55

# Highlight selection stability (track-ID based) and box smoothing.
SELECT_STICKY_MARGIN = 0.05     # keep an incumbent unless a rival beats it by this
BOX_SMOOTH_ALPHA = 0.45         # EMA weight on the new box (lower = smoother)

BACKGROUND_DIM_ALPHA = 0.06

HIGHLIGHT_FILL_ALPHA = 0.14
HIGHLIGHT_FEATHER_PX = 7
CONTOUR_THICKNESS = 2
SHOW_DISTANCE_LABEL = True
# Distance labels use the depth model aligned to the road plane (metric); if the
# depth model is unavailable they fall back to the flat-ground feet estimate.
USE_DEPTH_DISTANCE = True

# Pedestrians in amber/orange, vehicles in green, so the two read as distinct.
BOX_COLOUR = (0, 220, 255)
BOX_COLOUR_CLOSE = (0, 160, 255)
VEHICLE_BOX_COLOUR = (120, 230, 60)
VEHICLE_BOX_COLOUR_CLOSE = (60, 200, 40)
TEXT_COLOUR = (255, 255, 255)
TEXT_BACKGROUND = (0, 0, 0)

# BGR. Cool AR path tint that stays readable on warm sunlit asphalt.
# Alternatives if a different look is wanted:
#   matte mint  -> (210, 235, 225) core (230, 250, 245)
#   road amber  -> ( 80, 200, 250) core (150, 235, 255)
ROAD_PATH_COLOUR = (245, 200, 90)
ROAD_PATH_CORE_COLOUR = (245, 235, 180)


# ---------------------------------------------------------------------------
# Ground-plane projection.
#
# The real planned trajectory (metres, flat ground) is projected through a
# simple pinhole camera so the ribbon lies on the road with correct perspective:
#
#     u = VANISH_U + CAM_FOCAL_PX * (LATERAL_SIGN * y_lat) / x_fwd
#     v = HORIZON_V + CAM_FOCAL_PX * CAM_HEIGHT_M / x_fwd
#
# A constant world width then narrows as 1 / distance (true foreshortening),
# and both rails converge to the vanishing point (VANISH_U, HORIZON_V).
#
# These scalars are tuned by eye with the --calibrate tool and are absolute
# pixels for CALIB_REF_WIDTH x CALIB_REF_HEIGHT; render_video warns if the input
# resolution differs (the projection would otherwise be silently misplaced).
# ---------------------------------------------------------------------------

CALIB_REF_WIDTH = 1280
CALIB_REF_HEIGHT = 720

# Per-clip camera calibration lives in a JSON so a new clip does not require
# editing source; if present it overrides the defaults below. Write a template
# for the current clip with:  python final_preview_renderer.py --save-calibration
CALIBRATION_JSON = workflow_path("calibration", segment_tag() + "_camera_calibration.json")
CALIBRATION_KEYS = (
    "HORIZON_V",
    "VANISH_U",
    "CAM_FOCAL_PX",
    "CAM_HEIGHT_M",
    "LATERAL_SIGN",
    "CALIB_REF_WIDTH",
    "CALIB_REF_HEIGHT",
)

HORIZON_V = 448.0
VANISH_U = 636.0
CAM_FOCAL_PX = 1000.0
CAM_HEIGHT_M = 1.30
LATERAL_SIGN = 1.0
MIN_FORWARD_M = 0.5

# Pixel thresholds that used to be bare literals inside the geometry functions.
# They are named here so apply_resolution_scaling() can reach them: each one is
# an absolute pixel quantity at the reference resolution, and leaving any of
# them unscaled silently changes the geometry rather than raising.
# to_ground: rows nearer the horizon than this are dropped, which sets the MAX
# metric depth of the fit.
DEPTH_HORIZON_EXCLUDE_PX = 30.0
FIT_DEPTH_HORIZON_EXCLUDE_PX = 8.0  # same idea for the road-plane depth fit
LANE_STRADDLE_GUARD_PX = 4.0      # dead zone each side of centre when pairing boundaries
LANE_WIDTH_FLOOR_PX = 40.0        # absolute floor on an accepted lane width
LANE_WIDTH_CEIL_PX = 560.0        # absolute ceiling on an accepted lane width
LANE_WIDTH_CONVERGE_PX = 28.0     # per-row width-change tolerance (absorbs detection noise)
LANE_ROW_TOL_PX = 3.0             # tolerance when a pick row sits just outside a polyline
AIM_CENTRAL_WINDOW_PX = 160.0     # half-width of the central column sampled for the aim
CHEVRON_TANGENT_LOOKBACK_PX = 6.0  # rows back along the centreline used for the tangent
LABEL_TOP_CLAMP_PX = 18.0         # keep a distance label inside the top edge
LABEL_ABOVE_BOX_PX = 8.0          # label offset above its box
RAIL_STROKE_PX = 2                # ribbon rail stroke width
PANEL_FONT_SCALE = 0.72           # passenger-text panel: cv2 font scale is pixel-space
PANEL_TEXT_THICKNESS = 2
PANEL_MARGIN_PX = 24
PANEL_PADDING_X_PX = 16
PANEL_PADDING_Y_PX = 12
LABEL_FONT_SCALE = 0.5            # per-object distance labels
LABEL_OUTLINE_THICKNESS = 3
LABEL_TEXT_THICKNESS = 1
GUIDE_LINE_THICKNESS = 1          # --calibrate overlay: horizon rule
GUIDE_VP_RADIUS_PX = 4            # vanishing-point marker
GUIDE_TICK_RADIUS_PX = 3          # metre distance ticks
GUIDE_LABEL_OFFSET_PX = 8         # tick label offset
GUIDE_FONT_SCALE = 0.5
GUIDE_TEXT_THICKNESS = 1

# Area-like thresholds: these count PIXELS inside a region whose height and
# width both grow with s, so they scale by s squared, not s.
AIM_MIN_MASK_PX = 40              # minimum road-mask pixels for a usable aim
FIT_DEPTH_MIN_ROAD_PX = 200       # minimum road pixels before fitting depth to metric

RIBBON_HALF_M = 0.60        # half-width in metres (a slim guidance path)
PATH_START_M = 4.0
PATH_END_M = 21.0
PATH_SAMPLE_STEP_M = 0.5

# Per-frame ribbon (as opposed to the static planned-trajectory projection): the
# ribbon is rebuilt each frame so its near end tracks the detected ego lane. Its
# direction is the vanishing point unless VO reports a real turn - see the module
# docstring for the full contract.
USE_ROAD_AIMED_RIBBON = True

# What shapes the ribbon:
#   perception (default) -- the road-aimed ribbon: lane centering plus
#     validated-VO turn shaping. Shows where the lane goes.
#   planner -- the Alpamayo trajectory from the context JSON, projected through
#     the camera. Shows where the PLANNER intends to go, which is the thing an
#     explainable-AV overlay is explaining; valid near t0 (the 64 waypoints
#     cover 6.4 s from the planned moment), which is exactly when the gate
#     shows the overlay. Falls back to perception when no context exists.
RIBBON_SOURCE = os.environ.get("OPTICARVIS_RIBBON_SOURCE", "perception")

# Alpamayo's ego frame is FLU: +y is LEFT. The renderer is right-positive
# (LATERAL_SIGN, section 3a of ENGINEERING.md). Verified on Fvt6rD9tt1c_22:
# the planner's own reasoning says "Adapt speed for the left curve ahead", its
# trajectory runs to y=+31 m, and the scene pan at t0 confirms the left curve
# -- so +y must be left, and the raw value must be negated here. Consuming it
# unsigned would mirror every planned turn.
PLANNER_LATERAL_SIGN = float(os.environ.get("OPTICARVIS_PLANNER_LATERAL_SIGN", "-1"))

# The most recent ribbon geometry, published by resolve_frame_overlay for the
# geometry dump. The temporal state that shapes it (lane tracker, VO offsets,
# aim EMA) cannot be reconstructed after the fact, which is the entire reason
# the dump exists.
LAST_RIBBON_GEOMETRY = None
# RIBBON_AIM_BAND is DEAD (kept only because RIBBON_AIM_BIAS is 0): rows that
# used to be read for road heading. RIBBON_AIM_BIAS is DISABLED, and
# deliberately so: aiming the far end at the road-mask centroid drifted the
# ribbon to the street centre on multi-lane roads. With this at 0 the road
# mask has NO effect on the ribbon's shape (it is still used for occlusion
# and the depth fit). Do not re-enable without re-solving the multi-lane
# drift.
RIBBON_AIM_BAND = (55.0, 110.0)
RIBBON_AIM_BIAS = 0.0
RIBBON_AIM_CAP_PX = 70.0          # DEAD while RIBBON_AIM_BIAS is 0
# EMA weight on the far aim across frames (lower = slower, more natural; the
# far end eases rather than darts).
RIBBON_AIM_SMOOTH = 0.07
# Ribbon near end: past the frame bottom (~4.2 m), so the frame edge clips it.
# A frozen near hem at a constant row read as a sticker, not paint.
RIBBON_NEAR_ROWS = 300.0
# ribbon far end, rows below the horizon (~21 m). Shortened from 48 (~27 m): the far field is
# where any per-frame estimate noise is most visible, and the last 6 m of ribbon carried most of
# the perceived jitter on the turn clip while adding little information.
RIBBON_FAR_ROWS = 62.0

# Look-ahead: bend the ribbon into an upcoming turn using the ego-motion track
# (ego_motion.py), read LOOKAHEAD_S seconds into the future. During a turn the
# scene pans hard, so the accumulated future pan tells us how far the ego will
# have turned; we offset the far aim by that (calibrated by sign + gain).
#
# OFF by default and superseded by the VO path below. On straight roads the
# phase-correlation pan random-walks (camera shake, parallax, passing vehicles),
# which made the far end slowly wander ~130 px with no turn present; and a genuine
# turn's yaw signature is barely above that noise floor, so it could not reliably
# tell a turn from noise in the first place. ego_motion.py exists only to feed this
# path. Prefer OPTICARVIS_VO_TRAJECTORY.
USE_EGO_LOOKAHEAD = os.environ.get("OPTICARVIS_EGO_LOOKAHEAD", "0") == "1"
LOOKAHEAD_S = 3.5
EGO_YAW_SIGN = -1.0               # maps scene pan -> ribbon direction
EGO_YAW_GAIN = 1.3               # ribbon far-aim px per px of future scene pan
EGO_LOOKAHEAD_CAP_PX = 360.0     # max far-aim deviation once look-ahead is added
# The raw ego-motion look-ahead is noisy (phase-correlation wobble makes the far
# end swim on a straight road). Deadband small offsets to zero so straights stay
# straight, then EMA the result so a real turn ramps in smoothly.
LOOKAHEAD_DEADBAND_PX = 14.0      # treat |offset| below this as noise -> 0 (straight)
LOOKAHEAD_SMOOTH = 0.10          # EMA weight on the look-ahead offset across frames

# Future-frame VO ribbon (ego_trajectory.py): reconstruct the ego's actual path
# from the pre-recorded frames and use it to shape the ribbon through a genuine
# turn, where lane detection has nothing to follow (no lane lines through an
# intersection). Hybrid: the ribbon stays lane-centered on straights/gentle
# curves and VO only takes over as the path bends hard, so VO's straight-road
# noise never shows. Off by default; enable per turn-clip render.
USE_VO_TRAJECTORY = os.environ.get("OPTICARVIS_VO_TRAJECTORY", "0") == "1"
VO_TURN_LAT_LO = 2.5              # m of path lateral spread below which it stays lane-centered
# m at/above which the ribbon is fully VO-shaped Measured separation (in-window lateral, ribbon
# rows only): straight road peaks at 2.2 m, the real curve runs 4.0-6.2 m, so this band
# suppresses phantom turns (weight 0.00) while a genuine turn still reaches full weight.
VO_TURN_LAT_HI = 5.5
# EMA weight on the turn weight across frames. Replaying the turn clip: at 0.08 (+2.5 px step)
# the drawn bend trailed the instantaneous VO bend by ~0.4 s and stopped ~15 px shy of its full
# depth; 0.20 + 8 px tracks it closely. The VO path is an integrated trajectory and inherently
# smooth, so the light smoothing costs no visible jitter.
VO_WEIGHT_SMOOTH = 0.20
VO_MAX_STEP_PX = 8.0              # max per-row, per-frame move of the VO contribution
# EMA on the applied VO offsets before the rate clamp: the raw per-frame VO projection wiggles a
# few px frame-to-frame, and at full weight that wiggle painted visible jitter on the bend. 0.35
# keeps the response inside ~3 frames (no return of the trailing) while averaging the wiggle
# away.
VO_OFFSET_SMOOTH = 0.35

# Ego-lane centering (Solution B): detect the lane the car is actually in (from
# YOLOP lane lines) and center the ribbon between its markings, instead of
# assuming the ego sits at image centre (which is wrong on multi-lane roads or
# when the car rides off-centre in its lane). Robust by design: it only trusts
# the detection where a converging pair of longitudinal lane lines straddles the
# car; at intersections / crosswalks (no lane lines, only horizontal stripes)
# the confidence collapses and the ribbon falls back to straight in the ego lane.
USE_LANE_CENTERING = True
# Lane source for ego-lane centering:
#   "ufldv2" - UFLDv2 lane *instances* (clean per-lane polylines; straddle-select
#              the ego boundaries). The proper fix; runs on the main GPU.
#   "yolop"  - YOLOP lane *mask* + the row-scan heuristic (legacy fallback).
LANE_SOURCE = os.environ.get("OPTICARVIS_LANE_SOURCE", "ufldv2")
LANE_CONF_FULL_TRUST = 0.25       # valid-row fraction that earns full lane trust
LANE_CENTER_CAP_PX = 140.0        # max lane-centre offset from straight (VANISH_U)
# Anchor tracker (alpha-beta with a velocity state, gains scaled by detection
# trust). Two lessons are baked in, both measured on recorded lane series:
#   * a plain EMA + step cap lags a MOVING lane centre by ~8 frames / 5 px, which
#     read on screen as the ribbon "trailing behind" the car's own lane;
#   * a detection DROPOUT (crosswalk, worn paint, dashed-line gap) is not
#     evidence that the lane centre is at the image centre - it is no new
#     information. Blending toward VANISH_U by trust made the target a square
#     wave (in-lane -> centre -> in-lane every few seconds) and the ribbon
#     visibly wandered chasing it. The tracker now COASTS through dropouts and
#     only relaxes toward straight once the lanes have been gone for a while.
LANE_TRACK_ALPHA = 0.30           # position gain at full trust
LANE_TRACK_BETA = 0.035           # velocity gain at full trust
LANE_TRACK_VMAX = 5.0             # max tracked velocity, px/frame
LANE_TRACK_GATE_PX = 25.0         # innovation clamp: one bad frame moves x <= alpha*this
LANE_TRACK_MIN_TRUST = 0.05       # below this the frame counts as a dropout (coast)
LANE_HOLD_FRAMES = 75             # coast this long (~2.5 s) before relaxing to straight
LANE_RELAX_RATE_PX = 0.6          # px/frame ease toward VANISH_U after the hold expires
LANE_VEL_DECAY = 0.92             # per-frame velocity decay while coasting
LANE_SCAN_BOTTOM_MARGIN_PX = 15   # start the row scan this far above the image bottom
LANE_DILATE_PX = 7                # horizontal dilation to bridge dashed-line gaps
# Depth-scaled width plausibility: a real ego lane spans about LANE_WIDTH_REF_M
# worth of pixels at a given row (empirical for this camera). Pairs far narrower
# than that are the centre line paired with a one-sided marking (e.g. a narrow
# street marked only on the right) and are rejected, so such frames fall back to
# straight instead of leaning toward a spurious half-lane gap.
LANE_WIDTH_REF_M = 1.55
LANE_WIDTH_FRAC_LO = 0.65         # reject pairs narrower than this * expected width
LANE_WIDTH_FRAC_HI = 1.70         # reject pairs wider than this (spanning two lanes)
# how far past the lowest detected lane row the ribbon's near row may sit before we stop trusting
# the (clamped) extrapolation
LANE_EXTRAPOLATE_TOL_PX = 25.0

# Lane-curve fit: unproject the two straddling boundary polylines to the ground
# plane, fit a curvature-capped quadratic to their midline, and let the ribbon
# follow it. Gentle bends with visible paint are followed; the caps mean a
# straight road cannot be bent into a fake curve (measured fits on straight
# frames come out R = 1400-4000 m, i.e. visually straight), and when the
# boundaries do not span the near field the curve decays to zero and the ribbon
# is the straight ground line as before. Real turns remain VO's job - boundary
# coverage collapses exactly where sharp curvature lives.
USE_LANE_CURVE = os.environ.get("OPTICARVIS_LANE_CURVE", "1") == "1"
LANE_CURVE_MAX_K = 0.006          # |quadratic coeff| cap  ->  radius >= ~83 m
LANE_CURVE_MAX_H = 0.15           # |heading| cap at the near anchor (rad-ish)
# trust-scaled gain on the tracked heading/curvature. Decomposing the far-row wiggle on the turn
# clip showed the curve state was the dominant jitter source (anchor 0.69 px, +curve 1.42 px, VO
# negligible); 0.08 halves it while a gentle bend still acquires within ~0.5 s.
LANE_CURVE_SMOOTH = 0.08
LANE_CURVE_STEP_H = 0.004         # max per-frame heading change (a dropout->redetect used
LANE_CURVE_STEP_K = 0.00015       # to slam a fresh fit in at full gain: 12 px far-row step)
LANE_CURVE_DECAY = 0.95           # per-frame decay toward straight when the fit is absent
LANE_CURVE_PICK_ROW = 620.0       # boundaries must span this near-field row to qualify
LANE_CURVE_MIN_SPAN_M = 8.0       # minimum forward overlap of the two boundaries

DASH_PERIOD_M = 3.0
DASH_FILL_M = 1.5

# Flowing chevrons on the per-frame ribbon (replace the static centre dashes):
# arrowheads spaced in world metres that march forward along the path, so the
# overlay communicates direction and intent, and small path corrections read as
# motion rather than error. The world spacing means they compress toward the
# horizon exactly like real road paint.
CHEVRON_PERIOD_M = 3.0            # spacing between chevrons, metres along the path
CHEVRON_LEN_M = 0.9               # forward extent of each chevron, metres
CHEVRON_HALF_M = 0.32             # half-width of the arms, metres
CHEVRON_SPEED_MPS = 5.0           # how fast the chevrons flow forward

HORIZON_CLIP_MARGIN_PX = 4.0
BODY_BLUR = 15

# Opacity ramps, measured in image rows from the ribbon's own near / far ends,
# so they adapt to however tall the projected ribbon turns out to be.
# These are TRUE opacities: alpha comes from a coverage mask, so changing
# ROAD_PATH_COLOUR no longer changes how transparent the ribbon is. The values are
# the ones the old luminance-derived path happened to render, so the look is
# unchanged - they are simply honest now, and the rails and the dashed centre line
# get their own constants instead of sharing one that they silently disagreed on.
BODY_ALPHA = 0.28
RAILS_ALPHA = 0.64
DASH_ALPHA = 0.82
FADE_IN_PX = 16.0
# shorter far fade to suit the shorter ribbon (with 55 the fade regions covered most of the 88
# remaining rows)
FADE_OUT_PX = 35.0

# Subtle contact shadow that grounds the ribbon (offset down, blurred, faded).
CONTACT_SHADOW_ALPHA = 0.18
CONTACT_SHADOW_OFFSET_PX = 6
CONTACT_SHADOW_BLUR = 25

# Objects nearer than the ribbon feather its edge by this many pixels.
OCCLUSION_FEATHER_PX = 9

# Clip the ribbon to the real drivable surface using road segmentation
# (SegFormer/Cityscapes via scene_models). This occludes the ribbon behind
# anything that is not road - people, vehicles, curbs - pixel accurately, and
# supersedes the YOLO object-mask occlusion when available. Falls back to the
# object-mask occlusion if the model or its deps are missing.
USE_ROAD_SEGMENTATION = True
ROAD_MASK_FEATHER_PX = 9

# Temporal gating (sliding-window gate timeline): hysteresis + intro/outro
# animation. The ribbon draws on from the ego outward when an explanation
# starts and retracts far-to-near when it ends; highlights/label/dim fade with
# the ramp.
ANIM_START_S = 0.7        # intro (draw-on) duration
ANIM_END_S = 0.6          # outro (retract) duration
ANIM_MIN_ON_S = 2.0       # minimum time the overlay stays on once triggered
ANIM_OFF_DELAY_S = 1.5    # bridge gaps shorter than this so it does not flicker
ANIM_REVEAL_SOFT_PX = 26.0

# Precomputed lookup table for the (static) background dim.
DIM_LUT = np.clip(
    np.arange(256, dtype=np.float32) * (1.0 - BACKGROUND_DIM_ALPHA), 0, 255
).astype(np.uint8)



def opticarvis_project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_render_config():
    config_path = os.environ.get(
        "OPTICARVIS_RENDER_CONFIG",
        os.path.join(opticarvis_project_root(), "configs", "render_default.json"),
    )

    if not os.path.isfile(config_path):
        print("Render config not found, using renderer defaults:", config_path)
        return {
            "render_config_path": config_path,
            "render_config_loaded": False,
        }

    with open(config_path, "r", encoding="utf-8") as handle:
        config = json.load(handle)

    config["render_config_path"] = config_path
    config["render_config_loaded"] = True

    return config


def apply_render_config_to_globals(render_config):
    """Apply BO render parameters to existing renderer constants where present."""

    float_mapping = {
        "target_mask_alpha": [
            "TARGET_MASK_ALPHA",
            "HIGHLIGHT_FILL_ALPHA",
            "MASK_ALPHA",
            "PERSON_MASK_ALPHA",
        ],
        "background_dim_alpha": [
            "BACKGROUND_DIM_ALPHA",
            "BACKGROUND_ALPHA",
            "DIM_ALPHA",
        ],
        "trajectory_ribbon_alpha": [
            "TRAJECTORY_RIBBON_ALPHA",
            "RIBBON_ALPHA",
            "PATH_ALPHA",
        ],
        "trajectory_ribbon_width_scale": [
            "TRAJECTORY_RIBBON_WIDTH_SCALE",
            "RIBBON_WIDTH_SCALE",
            "PATH_WIDTH_SCALE",
        ],
        "label_font_scale": [
            "LABEL_FONT_SCALE",
            "FONT_SCALE",
            "TEXT_SCALE",
        ],
    }

    int_mapping = {
        "target_contour_thickness": [
            "TARGET_CONTOUR_THICKNESS",
            "CONTOUR_THICKNESS",
            "MASK_CONTOUR_THICKNESS",
        ],
    }

    bool_mapping = {
        "target_mask_visible": [
            "TARGET_MASK_VISIBLE",
            "HIGHLIGHT_MASK_VISIBLE",
        ],
        "target_contour_visible": [
            "TARGET_CONTOUR_VISIBLE",
            "CONTOUR_VISIBLE",
        ],
        "background_dim_visible": [
            "BACKGROUND_DIM_VISIBLE",
        ],
        "trajectory_ribbon_visible": [
            "TRAJECTORY_RIBBON_VISIBLE",
            "RIBBON_VISIBLE",
            "PATH_VISIBLE",
        ],
        "label_visible": [
            "LABEL_VISIBLE",
            "TEXT_LABEL_VISIBLE",
        ],
    }

    applied = {}

    for key, names in float_mapping.items():
        if key not in render_config:
            continue

        value = float(render_config[key])

        for name in names:
            if name in globals():
                globals()[name] = value
                applied[name] = value

    for key, names in int_mapping.items():
        if key not in render_config:
            continue

        value = int(render_config[key])

        for name in names:
            if name in globals():
                globals()[name] = value
                applied[name] = value

    for key, names in bool_mapping.items():
        if key not in render_config:
            continue

        value = bool(render_config[key])

        for name in names:
            if name in globals():
                globals()[name] = value
                applied[name] = value

    if "palette_id" in render_config:
        globals()["OPTICARVIS_RENDER_PALETTE_ID"] = str(render_config["palette_id"])
        applied["OPTICARVIS_RENDER_PALETTE_ID"] = str(render_config["palette_id"])

    print()
    print("Render config")
    print("=============")
    print("loaded:", render_config.get("render_config_loaded", False))
    print("path:", render_config.get("render_config_path", ""))
    print("config_id:", render_config.get("render_config_id", ""))
    print("applied_constants:", applied)

    return applied



def apply_calibration_overrides():
    """Override the camera scalars from the per-clip calibration JSON, if any."""
    if not os.path.isfile(CALIBRATION_JSON):
        return
    data = read_json(CALIBRATION_JSON, "camera calibration")
    module_globals = globals()
    for key in CALIBRATION_KEYS:
        if key in data:
            module_globals[key] = data[key]
    print("Loaded camera calibration:", CALIBRATION_JSON)


# Names scaled by s (linear pixels) and by s squared (pixel counts) when the
# frame differs from the calibration reference. Anything metric (metres),
# dimensionless (ratios, alphas, EMA weights, trust fractions), temporal
# (frames, seconds) or a model input size (IMAGE_SIZE) is deliberately absent.
RESOLUTION_SCALED_PX = (
    "HORIZON_V", "VANISH_U", "CAM_FOCAL_PX",
    "RIBBON_NEAR_ROWS", "RIBBON_FAR_ROWS", "RIBBON_AIM_CAP_PX",
    "EGO_LOOKAHEAD_CAP_PX", "LOOKAHEAD_DEADBAND_PX", "VO_MAX_STEP_PX",
    "LANE_CENTER_CAP_PX", "LANE_TRACK_VMAX", "LANE_TRACK_GATE_PX",
    "LANE_RELAX_RATE_PX", "LANE_SCAN_BOTTOM_MARGIN_PX", "LANE_DILATE_PX",
    "LANE_EXTRAPOLATE_TOL_PX", "LANE_CURVE_PICK_ROW",
    "HORIZON_CLIP_MARGIN_PX", "BODY_BLUR",
    "FADE_IN_PX", "FADE_OUT_PX", "ANIM_REVEAL_SOFT_PX",
    "CONTACT_SHADOW_OFFSET_PX", "CONTACT_SHADOW_BLUR",
    "OCCLUSION_FEATHER_PX", "ROAD_MASK_FEATHER_PX",
    "HIGHLIGHT_FEATHER_PX", "CONTOUR_THICKNESS",
    "DEPTH_HORIZON_EXCLUDE_PX", "FIT_DEPTH_HORIZON_EXCLUDE_PX",
    "LANE_STRADDLE_GUARD_PX", "LANE_WIDTH_FLOOR_PX", "LANE_WIDTH_CEIL_PX",
    "LANE_WIDTH_CONVERGE_PX", "LANE_ROW_TOL_PX", "AIM_CENTRAL_WINDOW_PX",
    "CHEVRON_TANGENT_LOOKBACK_PX", "LABEL_TOP_CLAMP_PX", "LABEL_ABOVE_BOX_PX",
    "RAIL_STROKE_PX", "PANEL_FONT_SCALE", "PANEL_TEXT_THICKNESS",
    "PANEL_MARGIN_PX", "PANEL_PADDING_X_PX", "PANEL_PADDING_Y_PX",
    "LABEL_FONT_SCALE", "LABEL_OUTLINE_THICKNESS", "LABEL_TEXT_THICKNESS",
    "GUIDE_LINE_THICKNESS", "GUIDE_VP_RADIUS_PX", "GUIDE_TICK_RADIUS_PX",
    "GUIDE_LABEL_OFFSET_PX", "GUIDE_FONT_SCALE", "GUIDE_TEXT_THICKNESS",
)

RESOLUTION_SCALED_AREA = ("AIM_MIN_MASK_PX", "FIT_DEPTH_MIN_ROAD_PX")

# Set once per process so a second entry point cannot double-scale the globals.
_RESOLUTION_SCALE_APPLIED = None

# Kernel sizes must stay odd positive ints for cv2's Gaussian blur.
RESOLUTION_ODD_KERNELS = (
    "BODY_BLUR", "CONTACT_SHADOW_BLUR", "OCCLUSION_FEATHER_PX",
    "ROAD_MASK_FEATHER_PX", "HIGHLIGHT_FEATHER_PX",
)

RESOLUTION_INT_NAMES = RESOLUTION_ODD_KERNELS + (
    "CONTOUR_THICKNESS", "RAIL_STROKE_PX", "LANE_SCAN_BOTTOM_MARGIN_PX", "LANE_DILATE_PX",
    "PANEL_TEXT_THICKNESS", "PANEL_MARGIN_PX", "PANEL_PADDING_X_PX",
    "PANEL_PADDING_Y_PX", "LABEL_OUTLINE_THICKNESS", "LABEL_TEXT_THICKNESS",
    "GUIDE_LINE_THICKNESS", "GUIDE_VP_RADIUS_PX", "GUIDE_TICK_RADIUS_PX",
    "GUIDE_LABEL_OFFSET_PX", "GUIDE_TEXT_THICKNESS",
    "CONTACT_SHADOW_OFFSET_PX",
) + RESOLUTION_SCALED_AREA


def apply_resolution_scaling(width, height):
    """Rescale the pixel-space scalars from the calibration reference to a frame.

    The calibration is absolute pixels at CALIB_REF_WIDTH x CALIB_REF_HEIGHT.
    Most clips in this project are 3840x2160, and rendering those with
    reference-resolution scalars puts the horizon, the vanishing point and every
    row offset at a third of where they belong. render_video used to only warn.

    Scaling HORIZON_V, VANISH_U and CAM_FOCAL_PX by the same s leaves the
    projection self-consistent: u and v both scale by s, and the inverse
    d = f*H/(v - HORIZON_V) is unchanged, so metric distances are preserved.

    Returns the applied scale. An exact no-op at the reference resolution --
    the equality check returns before any global is touched, so clips already
    at 1280x720 render bit-identically to before this function existed.
    """
    global _RESOLUTION_SCALE_APPLIED

    if _RESOLUTION_SCALE_APPLIED is not None:
        return _RESOLUTION_SCALE_APPLIED

    if (width, height) == (CALIB_REF_WIDTH, CALIB_REF_HEIGHT):
        _RESOLUTION_SCALE_APPLIED = 1.0
        return 1.0

    scale_x = float(width) / float(CALIB_REF_WIDTH)
    scale_y = float(height) / float(CALIB_REF_HEIGHT)

    if abs(scale_x - scale_y) > 0.01:
        print(
            "WARNING: %dx%d has a different aspect ratio to the %dx%d calibration"
            % (width, height, CALIB_REF_WIDTH, CALIB_REF_HEIGHT)
        )
        print("         a single pinhole scale cannot fix that; re-tune with --calibrate.")

    scope = globals()

    for name in RESOLUTION_SCALED_PX:
        value = scope[name]
        scope[name] = tuple(v * scale_x for v in value) if isinstance(value, tuple) else value * scale_x

    for name in RESOLUTION_SCALED_AREA:
        scope[name] = scope[name] * scale_x * scale_x

    # RIBBON_AIM_BAND is a tuple of horizon-relative rows.
    RIBBON_AIM_BAND_scaled = tuple(v * scale_x for v in scope["RIBBON_AIM_BAND"])
    scope["RIBBON_AIM_BAND"] = RIBBON_AIM_BAND_scaled

    for name in RESOLUTION_INT_NAMES:
        value = int(round(scope[name]))

        if name in RESOLUTION_ODD_KERNELS:
            value = max(1, value | 1)

        scope[name] = value

    print("Scaled camera calibration by %.3f for %dx%d" % (scale_x, width, height))

    _RESOLUTION_SCALE_APPLIED = scale_x

    return scale_x


def save_calibration_template():
    """Write the current camera scalars to the per-clip calibration JSON."""
    ensure_dir(os.path.dirname(CALIBRATION_JSON))
    write_json(CALIBRATION_JSON, {key: globals()[key] for key in CALIBRATION_KEYS})
    print("Wrote camera calibration:", CALIBRATION_JSON)


def get_video_metadata(video_file):
    if not os.path.isfile(video_file):
        print("Missing input video:")
        print(video_file)
        raise SystemExit(1)

    capture = cv2.VideoCapture(video_file)

    if not capture.isOpened():
        print("Could not open input video:")
        print(video_file)
        raise SystemExit(1)

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    capture.release()

    return fps, width, height, frame_count


def dim_background(frame):
    # Single table lookup instead of a full-frame float conversion.
    return cv2.LUT(frame, DIM_LUT)


def draw_text_panel(frame, label_text):
    height, width = frame.shape[:2]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = PANEL_FONT_SCALE
    thickness = PANEL_TEXT_THICKNESS

    text_size, _ = cv2.getTextSize(label_text, font, font_scale, thickness)
    text_width = text_size[0]
    text_height = text_size[1]

    margin = PANEL_MARGIN_PX
    padding_x = PANEL_PADDING_X_PX
    padding_y = PANEL_PADDING_Y_PX

    x1 = margin
    y1 = margin
    x2 = min(width - margin, x1 + text_width + 2 * padding_x)
    y2 = y1 + text_height + 2 * padding_y

    # Darken only the panel ROI toward black, rather than copying the whole frame.
    roi = frame[y1:y2, x1:x2]
    dark = np.full_like(roi, TEXT_BACKGROUND)
    cv2.addWeighted(dark, 0.42, roi, 0.58, 0, roi)

    cv2.putText(
        frame,
        label_text,
        (x1 + padding_x, y2 - padding_y),
        font,
        font_scale,
        TEXT_COLOUR,
        thickness,
        cv2.LINE_AA,
    )


def project_ground_point(x_fwd, y_lat):
    """Project a flat-ground point (metres) to image pixels.

    x_fwd is forward distance ahead of the camera, y_lat is lateral offset.
    Returns a float (u, v) pair, or None for points at or behind the camera.
    """
    if x_fwd <= MIN_FORWARD_M:
        return None

    u = VANISH_U + CAM_FOCAL_PX * (LATERAL_SIGN * y_lat) / x_fwd
    v = HORIZON_V + CAM_FOCAL_PX * CAM_HEIGHT_M / x_fwd

    return u, v


def ground_distance_from_row(v):
    """Inverse of the vertical projection: forward metres for a ground row v.

    Assumes the point sits on the road plane (e.g. a pedestrian's feet). Returns
    None for rows on or above the horizon.
    """
    denom = float(v) - HORIZON_V
    if denom <= 1e-3:
        return None
    return CAM_FOCAL_PX * CAM_HEIGHT_M / denom


def load_trajectory_points():
    """Read the real planned trajectory as a list of (x_fwd, y_lat) metres.

    Filters points at/behind the camera, sorts by forward distance, and drops
    non-increasing duplicates so lateral_at_distance can assume monotonic x.
    Returns None when the context file or its trajectory field is missing.
    """
    if not os.path.isfile(ALPAMAYO_CONTEXT_JSON):
        return None

    with open(ALPAMAYO_CONTEXT_JSON, "r", encoding="utf-8") as handle:
        context = json.load(handle)

    raw_points = context.get("trajectory_points_xyz")

    if not raw_points:
        return None

    points = []
    for point in raw_points:
        if len(point) < 2:
            continue
        x_fwd = float(point[0])
        if x_fwd <= MIN_FORWARD_M:
            continue
        # PLANNER_LATERAL_SIGN: Alpamayo's +y is left, the renderer's is right.
        points.append((x_fwd, PLANNER_LATERAL_SIGN * float(point[1])))

    points.sort(key=lambda item: item[0])

    monotonic = []
    for x_fwd, y_lat in points:
        if monotonic and x_fwd <= monotonic[-1][0] + 1e-6:
            continue
        monotonic.append((x_fwd, y_lat))

    if len(monotonic) < 2:
        return None

    return monotonic


def straight_trajectory_points():
    """Fallback straight-ahead path when no real trajectory is available."""
    points = []
    distance = PATH_START_M
    while distance <= PATH_END_M + 1e-6:
        points.append((distance, 0.0))
        distance += PATH_SAMPLE_STEP_M
    return points


def lateral_at_distance(traj, x_fwd):
    """Linearly interpolate the trajectory lateral offset at a forward distance."""
    if x_fwd <= traj[0][0]:
        return traj[0][1]
    if x_fwd >= traj[-1][0]:
        return traj[-1][1]

    for index in range(1, len(traj)):
        x0, y0 = traj[index - 1]
        x1, y1 = traj[index]
        if x_fwd <= x1:
            span = x1 - x0
            if span <= 1e-6:
                return y1
            ratio = (x_fwd - x0) / span
            return y0 + ratio * (y1 - y0)

    return traj[-1][1]


def build_ribbon_geometry(traj):
    """Project the trajectory into image space as ribbon rails and a fill polygon.

    Samples uniformly in world distance from PATH_START_M to PATH_END_M so the
    projected vertices cluster naturally toward the horizon. Each sample gives a
    centre point plus left / right rails offset by RIBBON_HALF_M in world
    lateral, which keeps the rails parallel (no self intersection). All geometry
    stays float; it is only rounded when drawn.
    """
    centre = []
    left = []
    right = []

    distance = PATH_START_M
    while distance <= PATH_END_M + 1e-6:
        y_lat = lateral_at_distance(traj, distance)

        centre_pt = project_ground_point(distance, y_lat)
        left_pt = project_ground_point(distance, y_lat - RIBBON_HALF_M)
        right_pt = project_ground_point(distance, y_lat + RIBBON_HALF_M)

        distance += PATH_SAMPLE_STEP_M

        if centre_pt is None or left_pt is None or right_pt is None:
            continue

        # Never let the ribbon climb onto or above the horizon.
        if centre_pt[1] < HORIZON_V + HORIZON_CLIP_MARGIN_PX:
            continue

        centre.append(centre_pt)
        left.append(left_pt)
        right.append(right_pt)

    if len(centre) < 2:
        return None

    centre_arr = np.array(centre, dtype=np.float32)
    left_arr = np.array(left, dtype=np.float32)
    right_arr = np.array(right, dtype=np.float32)

    polygon = np.round(np.vstack([left_arr, right_arr[::-1]])).astype(np.int32)

    return {
        "traj": traj,
        "centre": centre_arr,
        "left": left_arr,
        "right": right_arr,
        "polygon": polygon,
        "near_v": float(centre_arr[0][1]),
        "far_v": float(centre_arr[-1][1]),
    }


def ego_lane_center(lane_mask, height, width):
    """Detect the ego lane's centre column from a lane-line mask (Solution B).

    Scans image rows from near the bottom up toward the horizon, tracking the two
    lane lines that straddle the running centre column. A row only counts when it
    has both a left and a right line AND the gap between them converges toward the
    horizon (the signature of real longitudinal lane paint; horizontal crosswalk
    stripes fail this test and are rejected). Returns the tracked lane centre at
    the ribbon's near and far rows plus a confidence = fraction of valid rows.
    Confidence is near zero at intersections / crosswalks, so the caller blends
    back to straight there instead of chasing spurious markings.
    """
    dilated = cv2.dilate(lane_mask, np.ones((1, LANE_DILATE_PX), np.uint8))
    near_v = int(HORIZON_V + RIBBON_NEAR_ROWS)
    far_v = int(HORIZON_V + RIBBON_FAR_ROWS)
    scan_bottom = min(height - 1, height - LANE_SCAN_BOTTOM_MARGIN_PX)
    scan_top = max(0, far_v - 1)

    centre = float(VANISH_U)
    prev_width = None
    per_row = {}
    valid = 0
    scanned = 0
    for v in range(scan_bottom, scan_top, -1):
        scanned += 1
        xs = np.where(dilated[v] > 0)[0]
        candidate = None
        if len(xs):
            left = xs[xs < centre - LANE_STRADDLE_GUARD_PX]
            right = xs[xs > centre + LANE_STRADDLE_GUARD_PX]
            if len(left) and len(right):
                left_x = float(left.max())
                right_x = float(right.min())
                lane_w = right_x - left_x
                expected_w = LANE_WIDTH_REF_M * (v - HORIZON_V) / CAM_HEIGHT_M
                lo = max(LANE_WIDTH_FLOOR_PX, LANE_WIDTH_FRAC_LO * expected_w)
                hi = min(LANE_WIDTH_CEIL_PX, LANE_WIDTH_FRAC_HI * expected_w)
                converges = prev_width is None or lane_w <= prev_width + LANE_WIDTH_CONVERGE_PX
                if lo <= lane_w <= hi and converges:
                    candidate = 0.5 * (left_x + right_x)
                    prev_width = lane_w
                    valid += 1
        if candidate is not None:
            centre = 0.7 * centre + 0.3 * candidate
        per_row[v] = centre

    confidence = valid / max(scanned, 1)
    near_u = per_row.get(near_v, VANISH_U)
    far_u = per_row.get(far_v + 1, per_row.get(scan_top + 1, VANISH_U))
    return near_u, far_u, confidence


def ego_lane_center_from_instances(lanes, height, width):
    """Ego-lane centre from UFLDv2 lane instances (the proper, mask-free path).

    Given clean per-lane polylines, walk near-field rows and at each row pick the
    nearest detected boundary to the left and to the right of the ego column; the
    ego-lane centre is their midpoint. Because it selects by position (not by a
    fixed lane index) it is robust to the car riding off-centre or in any lane.
    Returns the centre column at the ribbon's near and far rows plus a confidence
    = fraction of rows that had a boundary on both sides (near zero at
    intersections / when only one side is painted, so the caller falls back).
    """
    if not lanes:
        return VANISH_U, VANISH_U, 0.0

    near_v = HORIZON_V + RIBBON_NEAR_ROWS
    far_v = HORIZON_V + RIBBON_FAR_ROWS
    scan_lo = int(height * 0.93)
    scan_hi = int(far_v - 2)

    # Pre-sort each lane's points by row for interpolation.
    sorted_lanes = []
    for lane in lanes:
        order = np.argsort(lane[:, 1])
        sorted_lanes.append((lane[order, 1], lane[order, 0]))  # (ys, xs)

    def x_at(ys, xs, y):
        if y < ys[0] - LANE_ROW_TOL_PX or y > ys[-1] + LANE_ROW_TOL_PX:
            return None
        return float(np.interp(y, ys, xs))

    rows = []
    centers = []
    valid = 0
    scanned = 0
    for y in range(scan_lo, scan_hi, -2):
        scanned += 1
        left = []
        right = []
        for ys, xs in sorted_lanes:
            x = x_at(ys, xs, y)
            if x is None:
                continue
            (left if x < VANISH_U else right).append(x)
        if not (left and right):
            continue
        # Plausibility gate: the pair straddling the ego column must be about one
        # lane wide at this row. Without it, the two outer kerbs of a multi-lane
        # street are accepted as "the ego lane" and the ribbon is planted on the
        # street centre - a straight ribbon in the wrong lane still points where
        # the car is not driving. (The legacy mask path always had this gate; the
        # instance path silently lost it.)
        boundary_l = max(left)
        boundary_r = min(right)
        lane_w = boundary_r - boundary_l
        expected_w = LANE_WIDTH_REF_M * (y - HORIZON_V) / CAM_HEIGHT_M
        if not (LANE_WIDTH_FRAC_LO * expected_w <= lane_w <= LANE_WIDTH_FRAC_HI * expected_w):
            continue        # rejected rows must NOT count as valid, so conf stays honest
        rows.append(float(y))
        centers.append(0.5 * (boundary_l + boundary_r))
        valid += 1

    if len(centers) < 2:
        return VANISH_U, VANISH_U, 0.0

    rows_arr = np.asarray(rows[::-1])           # ascending y for np.interp
    centers_arr = np.asarray(centers[::-1])

    # np.interp CLAMPS outside its domain. If the lane polylines stop above the
    # ribbon's near row there is no near-field measurement, so fall back rather
    # than pass off a far-field column as the near-end placement.
    confidence = valid / max(scanned, 1)
    if near_v > rows_arr[-1] + LANE_EXTRAPOLATE_TOL_PX:
        return VANISH_U, VANISH_U, 0.0
    near_u = float(np.interp(near_v, rows_arr, centers_arr))
    far_u = float(np.interp(far_v, rows_arr, centers_arr))
    return near_u, far_u, confidence


def fit_lane_curve(lanes):
    """Ground-plane quadratic fit y(d) of the ego lane's midline, or None.

    Picks the boundary polylines straddling the vanishing column at a near-field
    row, unprojects both to the road plane (d = f*h/(v-horizon), y = (u-VU)*d/f),
    and fits their midline with a distance-weighted, one-pass-robust quadratic.
    Returns (c2, c1, c0) with c2/c1 clamped to the curvature/heading caps.
    """
    if not lanes:
        return None

    pick_row = LANE_CURVE_PICK_ROW
    candidates = []
    for lane in lanes:
        order = np.argsort(lane[:, 1])
        ys, xs = lane[order, 1], lane[order, 0]
        if pick_row < ys[0] - LANE_ROW_TOL_PX or pick_row > ys[-1] + LANE_ROW_TOL_PX:
            continue                      # must span the near field
        candidates.append((float(np.interp(pick_row, ys, xs)), ys, xs))
    left = [c for c in candidates if c[0] < VANISH_U]
    right = [c for c in candidates if c[0] >= VANISH_U]
    if not (left and right):
        return None
    _, yl_v, xl_u = max(left, key=lambda c: c[0])
    _, yr_v, xr_u = min(right, key=lambda c: c[0])

    def to_ground(v, u):
        mask = v > HORIZON_V + DEPTH_HORIZON_EXCLUDE_PX
        d = CAM_FOCAL_PX * CAM_HEIGHT_M / (v[mask] - HORIZON_V)
        y = (u[mask] - VANISH_U) * d / CAM_FOCAL_PX
        order = np.argsort(d)
        return d[order], y[order]

    dl, yl = to_ground(yl_v, xl_u)
    dr, yr = to_ground(yr_v, xr_u)
    if len(dl) < 4 or len(dr) < 4:
        return None
    d0 = max(dl.min(), dr.min(), 6.0)
    d1 = min(dl.max(), dr.max(), 32.0)
    if d1 - d0 < LANE_CURVE_MIN_SPAN_M:
        return None

    grid = np.linspace(d0, d1, 40)
    midline = 0.5 * (np.interp(grid, dl, yl) + np.interp(grid, dr, yr))
    weights = 1.0 / grid                          # near points weigh more
    c2, c1, c0 = np.polyfit(grid, midline, 2, w=weights)
    residual = np.abs(np.polyval([c2, c1, c0], grid) - midline)
    scale = np.median(residual) * 1.48 + 1e-6
    weights = weights / np.maximum(1.0, residual / (2.5 * scale))
    c2, c1, c0 = np.polyfit(grid, midline, 2, w=weights)

    c2 = float(np.clip(c2, -LANE_CURVE_MAX_K, LANE_CURVE_MAX_K))
    c1 = float(np.clip(c1, -LANE_CURVE_MAX_H, LANE_CURVE_MAX_H))
    return c2, c1, float(c0)


def resolve_lane_center(lane_data, height, width, lane_state):
    """Tracked ego-lane anchor (near_u) with coast-through-dropout behaviour.

    An alpha-beta tracker whose gains scale with detection trust:

      * confident detection  -> measurement update toward the measured lane centre
        (full offset, NOT blended toward the image centre - the plausibility gates
        already rejected implausible geometry, so low confidence means "weigh it
        less", not "assume the centre"). The velocity state absorbs steady drift,
        so a moving lane centre is tracked without the EMA's ~8-frame lag that
        read as the ribbon trailing behind.
      * dropout (crosswalk, worn paint, dashed gap) -> COAST: hold course with a
        decaying velocity. The lane does not teleport to the image centre because
        the detector blinked; chasing the old trust-blended target visibly walked
        the ribbon to the centre and back on every dropout.
      * sustained dropout (a real lane-less stretch, > LANE_HOLD_FRAMES) -> ease
        gently toward straight ahead (VANISH_U).

    Returns None when lane centering is off / unavailable so the caller keeps the
    straight-ahead ribbon. lane_data is UFLDv2 lane instances (LANE_SOURCE
    "ufldv2") or a YOLOP lane mask ("yolop").
    """
    if not USE_LANE_CENTERING or lane_data is None or lane_state is None:
        return None

    if LANE_SOURCE == "ufldv2":
        raw_near, _raw_far, conf = ego_lane_center_from_instances(lane_data, height, width)
    else:
        raw_near, _raw_far, conf = ego_lane_center(lane_data, height, width)
    trust = min(1.0, conf / LANE_CONF_FULL_TRUST)

    measurement = VANISH_U + float(
        np.clip(raw_near - VANISH_U, -LANE_CENTER_CAP_PX, LANE_CENTER_CAP_PX))

    x = lane_state.get("near_u")
    vel = lane_state.get("vel", 0.0)
    dropout = lane_state.get("dropout", 0)

    if x is None:
        x = measurement if trust > LANE_TRACK_MIN_TRUST else float(VANISH_U)
        vel = 0.0
    else:
        x = x + vel
        if trust > LANE_TRACK_MIN_TRUST:
            innovation = float(np.clip(measurement - x,
                                       -LANE_TRACK_GATE_PX, LANE_TRACK_GATE_PX))
            x = x + LANE_TRACK_ALPHA * trust * innovation
            vel = vel + LANE_TRACK_BETA * trust * innovation
            dropout = 0
        else:
            vel = vel * LANE_VEL_DECAY
            dropout += 1
            if dropout > LANE_HOLD_FRAMES:
                x = x + float(np.clip(VANISH_U - x, -LANE_RELAX_RATE_PX, LANE_RELAX_RATE_PX))
    vel = float(np.clip(vel, -LANE_TRACK_VMAX, LANE_TRACK_VMAX))
    x = float(np.clip(x, VANISH_U - LANE_CENTER_CAP_PX, VANISH_U + LANE_CENTER_CAP_PX))

    # Lane-curve state (heading + curvature at the near anchor), tracked with the
    # same philosophy: trust-scaled updates when a fit exists, decay toward
    # straight when it does not. The caps make a fake curve impossible to hold.
    heading = lane_state.get("heading", 0.0)
    curvature = lane_state.get("curvature", 0.0)
    fit = None
    if USE_LANE_CURVE and LANE_SOURCE == "ufldv2" and trust > LANE_TRACK_MIN_TRUST:
        fit = fit_lane_curve(lane_data)
    if fit is not None:
        c2, c1, _c0 = fit
        d_near = CAM_FOCAL_PX * CAM_HEIGHT_M / RIBBON_NEAR_ROWS
        h_meas = float(np.clip(c1 + 2.0 * c2 * d_near, -LANE_CURVE_MAX_H, LANE_CURVE_MAX_H))
        k_meas = c2
        heading += float(np.clip(LANE_CURVE_SMOOTH * trust * (h_meas - heading),
                                 -LANE_CURVE_STEP_H, LANE_CURVE_STEP_H))
        curvature += float(np.clip(LANE_CURVE_SMOOTH * trust * (k_meas - curvature),
                                   -LANE_CURVE_STEP_K, LANE_CURVE_STEP_K))
    else:
        heading *= LANE_CURVE_DECAY
        curvature *= LANE_CURVE_DECAY
    heading = float(np.clip(heading, -LANE_CURVE_MAX_H, LANE_CURVE_MAX_H))
    curvature = float(np.clip(curvature, -LANE_CURVE_MAX_K, LANE_CURVE_MAX_K))

    lane_state["near_u"] = x
    lane_state["vel"] = vel
    lane_state["dropout"] = dropout
    lane_state["confidence"] = conf
    lane_state["heading"] = heading
    lane_state["curvature"] = curvature
    return {"near_u": x, "heading": heading, "curvature": curvature, "confidence": conf}


def perspective_profile(rows, near_v):
    """Fraction of the near-end lateral offset that survives at each ribbon row.

    A straight line on the ground images as a STRAIGHT line in the picture that
    converges on the vanishing point, so its horizontal offset from the vanishing
    column shrinks in exact proportion to (v - HORIZON_V) - reaching zero only at
    the horizon, not at the ribbon's far end.

    Interpolating with a smoothstep over row index instead (what this used to do)
    made the ribbon both bend and converge far too early: with the near end one
    lane to the left, the far end landed ~31 px too far toward the image centre
    and the middle of the ribbon bowed ~15 px, which reads as the path drifting
    out of the ego lane into the next one.
    """
    vs = near_v - np.arange(rows + 1, dtype=np.float32)
    return (vs - HORIZON_V) / max(1e-6, float(near_v - HORIZON_V))


def project_vo_centerline(vo_pts):
    """Project a VO future path [[fwd, lat], ...] to an image centreline.

    Returns (vs, us) sorted by image row (ascending v) for interpolation, or None
    if too few points project in front of the horizon.
    """
    projected = []
    for fwd, lat in vo_pts:
        point = project_ground_point(float(fwd), float(lat))
        if point is not None and point[1] >= HORIZON_V + HORIZON_CLIP_MARGIN_PX:
            projected.append(point)
    if len(projected) < 3:
        return None
    arr = np.array(projected, dtype=np.float32)
    order = np.argsort(arr[:, 1])
    return arr[order, 1], arr[order, 0]


def vo_turn_weight(vo_pts):
    """0 on straights, ramping to 1 for a genuine turn, from the path's lateral spread.

    The reconstructed future path barely deviates laterally on a straight road
    (~0.3 m of VO noise) but sweeps several metres through a turn, so its peak
    |lateral| is a robust turn detector that keeps VO out of straight sections.

    Only points that actually project into the ribbon's row window count: the VO
    horizon reaches far past the ribbon's far end, so measuring the whole path let
    a curve *beyond* the drawn ribbon saturate the weight while the in-window path
    was still straight - producing a fully-weighted band that never converged to
    the vanishing point.
    """
    if not vo_pts:
        return 0.0
    near_v = HORIZON_V + RIBBON_NEAR_ROWS
    far_v = HORIZON_V + RIBBON_FAR_ROWS
    in_window = []
    for fwd, lat in vo_pts:
        point = project_ground_point(float(fwd), float(lat))
        if point is not None and far_v <= point[1] <= near_v:
            in_window.append(abs(float(lat)))
    if not in_window:
        return 0.0
    lateral = max(in_window)
    return float(np.clip((lateral - VO_TURN_LAT_LO) / (VO_TURN_LAT_HI - VO_TURN_LAT_LO), 0.0, 1.0))


def aimed_ribbon_geometry(road_mask, height, width, aim_state, lookahead_offset=0.0,
                          lane_center=None, vo_pts=None, vo_state=None):
    """Build the per-frame ribbon: lane-anchored near end, vanishing-point direction.

    lane_center, when supplied, sets the NEAR end's column (which lane the car is
    in). The FAR end aims at VANISH_U - a straight path in the ego lane converges
    there - so the ribbon points straight down the lane by default. vo_pts
    (future-frame VO path), when supplied, is the ONLY source of curvature: its
    projected centreline is shifted to start at the lane centre and blended in by
    vo_turn_weight, so straights stay lane-centred and only a confident turn bends.

    road_mask no longer affects the ribbon's shape (RIBBON_AIM_BIAS is 0); it is
    retained in the signature because callers pass it and it still gates whether
    the per-frame ribbon is built at all. lookahead_offset is the legacy
    ego-motion path, disabled by default.
    """
    v0 = int(HORIZON_V + RIBBON_AIM_BAND[0])
    v1 = int(HORIZON_V + RIBBON_AIM_BAND[1])
    band = road_mask[max(0, v0):min(height, v1)]
    xs = np.where(band > 0)[1]
    mid = VANISH_U
    if len(xs) > AIM_MIN_MASK_PX:
        central = xs[(xs > VANISH_U - AIM_CENTRAL_WINDOW_PX) & (xs < VANISH_U + AIM_CENTRAL_WINDOW_PX)]
        if len(central) > AIM_MIN_MASK_PX:
            mid = float(np.median(central))

    # The lane detection supplies only the near-field lateral placement (which lane
    # the car is in). The ribbon then runs as the straight ground line through that
    # point, which converges on the VANISHING COLUMN at the horizon - note "at the
    # horizon", not at the ribbon's far end: the ribbon stops well short of the
    # horizon, so its far end keeps part of the near-end offset. Collapsing it onto
    # the vanishing column at the far end instead swung the ribbon out of the ego
    # lane toward the middle of the road. Steering the far end from the noisy
    # far-lane column had the same effect, for a different reason.
    # Genuine curvature comes from the VO blend below, only on a confident turn.
    near_anchor = float(lane_center["near_u"]) if lane_center else float(VANISH_U)
    far_anchor = float(VANISH_U)

    road_term = float(np.clip(
        RIBBON_AIM_BIAS * (mid - VANISH_U), -RIBBON_AIM_CAP_PX, RIBBON_AIM_CAP_PX))
    target = far_anchor + road_term + lookahead_offset
    target = float(np.clip(target, VANISH_U - EGO_LOOKAHEAD_CAP_PX, VANISH_U + EGO_LOOKAHEAD_CAP_PX))
    previous = aim_state.get("far_u")
    far_u = target if previous is None else (
        RIBBON_AIM_SMOOTH * target + (1.0 - RIBBON_AIM_SMOOTH) * previous)
    aim_state["far_u"] = far_u

    near_v = HORIZON_V + RIBBON_NEAR_ROWS
    far_v = HORIZON_V + RIBBON_FAR_ROWS
    near_u = near_anchor
    rows = int(round(near_v - far_v))
    row_grid = np.array([near_v - i for i in range(rows + 1)], dtype=np.float32)

    # Baseline centreline from the tracked ground curve. y(d) is a quadratic in
    # forward distance through the near anchor; with heading = curvature = 0 this
    # is EXACTLY the straight ground line (offset from the vanishing column decays
    # with (v - HORIZON_V), still partly present at the far end - see
    # perspective_profile's docstring for why it must not reach the VP early).
    # The legacy far_u term is zero unless the (disabled) look-ahead is active.
    heading = float(lane_center.get("heading", 0.0)) if lane_center else 0.0
    curvature = float(lane_center.get("curvature", 0.0)) if lane_center else 0.0
    depths = CAM_FOCAL_PX * CAM_HEIGHT_M / (row_grid - HORIZON_V)
    d_near = float(depths[0])
    y_near = (near_u - VANISH_U) * d_near / CAM_FOCAL_PX
    y_lat = y_near + heading * (depths - d_near) + curvature * (depths - d_near) ** 2
    base_u = (VANISH_U + CAM_FOCAL_PX * y_lat / depths
              + (far_u - VANISH_U) * (1.0 - (row_grid - HORIZON_V) / (near_v - HORIZON_V))
              ).astype(np.float32)

    # Hybrid VO turn shape. The weight ramps up only as the in-window path bends
    # hard and is EMA smoothed; the VO centreline is shifted so its near end sits
    # at the lane centre (the ribbon starts in-lane and curves out along the real
    # path). The projected VO rows MUST span the ribbon's rows: otherwise np.interp
    # clamps every row to one column and the shift cancels it exactly, which
    # rendered a dead-vertical (or wrong-side) ribbon while reporting full VO
    # weight. Partial coverage is tapered instead of silently clamped.
    target_w = 0.0
    vo_offsets = None
    if USE_VO_TRAJECTORY and vo_pts:
        projected = project_vo_centerline(vo_pts)
        if projected is not None:
            vo_vs, vo_us = projected
            covered = (row_grid >= vo_vs[0]) & (row_grid <= vo_vs[-1])
            if covered.any():
                vo_curve = np.interp(row_grid, vo_vs, vo_us).astype(np.float32)
                vo_curve = vo_curve + (near_u - float(np.interp(near_v, vo_vs, vo_us)))
                offsets = (vo_curve - base_u) * covered.astype(np.float32)
                # Only trust VO where it actually has samples for these rows.
                coverage = float(covered.mean())
                if coverage > 0.35:
                    vo_offsets = offsets
                    target_w = vo_turn_weight(vo_pts) * coverage

    if vo_state is not None:
        prev_w = vo_state.get("weight")
        vo_weight = target_w if prev_w is None else (
            VO_WEIGHT_SMOOTH * target_w + (1.0 - VO_WEIGHT_SMOOTH) * prev_w)
        vo_state["weight"] = vo_weight
    else:
        vo_weight = target_w

    # Smooth + rate-limit the VO contribution per row, so the far end cannot jump
    # frame to frame (the lane rate limiter only bounds the near anchor). The EMA
    # averages the raw projection's per-frame wiggle; the clamp bounds the worst case.
    applied = np.zeros(rows + 1, dtype=np.float32)
    if vo_offsets is not None and vo_weight > 0.005:
        applied = vo_offsets * vo_weight
    if vo_state is not None:
        prev_applied = vo_state.get("applied")
        if prev_applied is not None and len(prev_applied) == len(applied):
            smoothed = VO_OFFSET_SMOOTH * applied + (1.0 - VO_OFFSET_SMOOTH) * prev_applied
            delta = np.clip(smoothed - prev_applied, -VO_MAX_STEP_PX, VO_MAX_STEP_PX)
            applied = prev_applied + delta
        vo_state["applied"] = applied

    centre = []
    left = []
    right = []
    for i in range(rows + 1):
        v = near_v - i
        cu = float(base_u[i]) + float(applied[i])
        half = RIBBON_HALF_M * (v - HORIZON_V) / CAM_HEIGHT_M
        centre.append((cu, v))
        left.append((cu - half, v))
        right.append((cu + half, v))

    centre_arr = np.array(centre, dtype=np.float32)
    left_arr = np.array(left, dtype=np.float32)
    right_arr = np.array(right, dtype=np.float32)
    polygon = np.round(np.vstack([left_arr, right_arr[::-1]])).astype(np.int32)

    return {
        "traj": None,
        "centre": centre_arr,
        "left": left_arr,
        "right": right_arr,
        "polygon": polygon,
        "near_v": float(near_v),
        "far_v": float(far_v),
    }


def row_alpha_profile(height, near_v, far_v):
    """Per image row opacity ramp measured from the ribbon's near / far ends.

    Rows just inside the near end ramp up from 0 over FADE_IN_PX; the mid field
    is fully opaque; rows near the far end ramp back down to 0 over FADE_OUT_PX,
    so the ribbon fades into the distance instead of hard stopping.
    """
    rows = np.arange(height, dtype=np.float32)

    near_ramp = np.clip((near_v - rows) / max(1.0, FADE_IN_PX), 0.0, 1.0)
    far_ramp = np.clip((rows - far_v) / max(1.0, FADE_OUT_PX), 0.0, 1.0)

    return np.minimum(near_ramp, far_ramp)


def draw_world_dashes(layer, traj, colour=255):
    """Paint a dashed centre line whose dashes are spaced in world metres.

    Because each dash spans a fixed world distance, the projected dashes compress
    toward the horizon exactly like real lane paint. Dash thickness tapers with
    distance (smaller near the horizon).
    """
    distance = PATH_START_M
    while distance < PATH_END_M:
        seg_start = distance
        seg_end = min(distance + DASH_FILL_M, PATH_END_M)
        distance += DASH_PERIOD_M

        points = []
        sample = seg_start
        while sample <= seg_end + 1e-6:
            y_lat = lateral_at_distance(traj, sample)
            projected = project_ground_point(sample, y_lat)
            if projected is not None and projected[1] >= HORIZON_V + HORIZON_CLIP_MARGIN_PX:
                points.append(projected)
            sample += 0.25

        if len(points) < 2:
            continue

        mid_v = points[len(points) // 2][1]
        thickness = int(clamp(round((mid_v - HORIZON_V) * 0.02), 1, 5))

        arr = np.round(np.array(points, dtype=np.float32)).astype(np.int32)
        cv2.polylines(layer, [arr], False, colour, thickness, cv2.LINE_AA)


def draw_centerline_dashes(layer, centre_arr, colour=255):
    """Dashed centre line for the per-frame ribbon (image-space, per centre point)."""
    period = 16
    fill = 8
    n = len(centre_arr)
    i = 0
    while i < n - 1:
        seg = centre_arr[i:min(i + fill, n)]
        if len(seg) >= 2:
            mid_v = seg[len(seg) // 2][1]
            thickness = int(clamp(round((mid_v - HORIZON_V) * 0.02), 1, 5))
            cv2.polylines(layer, [np.round(seg).astype(np.int32)], False,
                          colour, thickness, cv2.LINE_AA)
        i += period


def draw_centerline_chevrons(layer, centre_arr, phase_m=0.0, colour=255,
                             assume_ordered=False):
    """Chevrons as WORLD ground-plane marks, projected like the band itself.

    The previous implementation built each chevron in image space (screen
    tangent, screen perpendicular) -- a rigid 2D glyph that rotated with the
    centreline's on-screen direction. On curves that produced arm angles up to
    70 degrees from horizontal, geometrically impossible for road paint, and
    the illusion collapsed. Here every chevron vertex is laid out in ground
    coordinates (apex on the path, arms mirrored about the LOCAL WORLD tangent)
    and projected through the same flat-ground mapping as the band, so a curve
    yaws the marks on the asphalt and foreshortening does the rest.

    phase_m is the vehicle's travelled distance: marks sit at fixed positions
    ON THE GROUND (arclength k*P - phase along the path window), so as the car
    advances the pattern streams toward the viewer at exactly the road
    texture's rate and is driven over -- the opposite of the old forward march,
    which was the strongest anti-grounding cue in the clip.
    """
    if len(centre_arr) < 3:
        return

    vs = centre_arr[:, 1]
    us = centre_arr[:, 0]

    # Anchor/arc centrelines arrive in path order already; sorting them by row
    # would scramble a turn that folds back (v is not monotone along the path).
    if assume_ordered:
        v_s = vs.astype(np.float64)
        u_s = us.astype(np.float64)
    else:
        order = np.argsort(vs)[::-1]       # near (large v) -> far
        v_s = vs[order].astype(np.float64)
        u_s = us[order].astype(np.float64)

    keep = v_s > HORIZON_V + 4.0
    v_s, u_s = v_s[keep], u_s[keep]

    if len(v_s) < 3:
        return

    # image centreline -> world ground path (forward metres, lateral metres)
    depth = CAM_FOCAL_PX * CAM_HEIGHT_M / (v_s - HORIZON_V)
    lateral = (u_s - VANISH_U) * depth / CAM_FOCAL_PX
    seg = np.hypot(np.diff(depth), np.diff(lateral))
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(arc[-1])

    if total < CHEVRON_PERIOD_M * 0.5:
        return

    def world_at(s_pos):
        d_i = float(np.interp(s_pos, arc, depth))
        y_i = float(np.interp(s_pos, arc, lateral))
        s_a = min(s_pos + 0.5, total)
        s_b = max(s_pos - 0.5, 0.0)
        t_x = float(np.interp(s_a, arc, depth) - np.interp(s_b, arc, depth))
        t_y = float(np.interp(s_a, arc, lateral) - np.interp(s_b, arc, lateral))
        norm = math.hypot(t_x, t_y)

        if norm < 1e-9:
            t_x, t_y, norm = 1.0, 0.0, 1.0

        return d_i, y_i, t_x / norm, t_y / norm

    def project(d_w, y_w):
        d_w = max(d_w, 0.3)
        return (VANISH_U + CAM_FOCAL_PX * y_w / d_w,
                HORIZON_V + CAM_FOCAL_PX * CAM_HEIGHT_M / d_w)

    # marks fixed on the ground: arclength (k*P - travelled) mod P walks the
    # pattern toward the car as it drives
    s_pos = (-phase_m) % CHEVRON_PERIOD_M - CHEVRON_PERIOD_M

    while s_pos < total:
        s_pos += CHEVRON_PERIOD_M

        if s_pos < 0.0 or s_pos > total:
            continue

        d_i, y_i, t_x, t_y = world_at(s_pos)
        n_x, n_y = -t_y, t_x

        apex = project(d_i + t_x * 0.6 * CHEVRON_LEN_M,
                       y_i + t_y * 0.6 * CHEVRON_LEN_M)
        tail_l = project(d_i - t_x * 0.4 * CHEVRON_LEN_M - n_x * CHEVRON_HALF_M,
                         y_i - t_y * 0.4 * CHEVRON_LEN_M - n_y * CHEVRON_HALF_M)
        tail_r = project(d_i - t_x * 0.4 * CHEVRON_LEN_M + n_x * CHEVRON_HALF_M,
                         y_i - t_y * 0.4 * CHEVRON_LEN_M + n_y * CHEVRON_HALF_M)

        v_mark = HORIZON_V + CAM_FOCAL_PX * CAM_HEIGHT_M / max(d_i, 0.3)
        thickness = int(clamp(round((v_mark - HORIZON_V) * 0.02), 1, 5))

        for tail in (tail_l, tail_r):
            cv2.line(layer,
                     (int(round(apex[0])), int(round(apex[1]))),
                     (int(round(tail[0])), int(round(tail[1]))),
                     colour, thickness, cv2.LINE_AA)


def build_path_overlay(geometry, height, width, chevron_phase_m=0.0):
    """Precompute the static ribbon as a premultiplied overlay.

    The ribbon is identical for every frame, so its shadow / body / dashed-mark
    layers are rasterised and folded once into (premul, a_tot) such that the
    per-frame composite over a background is simply:
        frame = frame * (1 - a_tot) + premul
    (see blend_path, which also scales by a per-frame occlusion mask). This
    replaces per-frame fills, blurs, colour conversions and the Python dash loop.
    """
    if geometry is None:
        return None

    profile = row_alpha_profile(height, geometry["near_v"], geometry["far_v"])
    profile_col = profile[:, None]

    # Contact shadow: the ribbon polygon shifted down and blurred (coverage only,
    # so a black layer still gets a real alpha).
    shadow_cov = np.zeros((height, width), dtype=np.uint8)
    shadow_poly = geometry["polygon"].copy()
    shadow_poly[:, 1] = shadow_poly[:, 1] + CONTACT_SHADOW_OFFSET_PX
    cv2.fillPoly(shadow_cov, [shadow_poly], 255)
    shadow_blur = CONTACT_SHADOW_BLUR | 1
    shadow_cov = cv2.GaussianBlur(shadow_cov, (shadow_blur, shadow_blur), 0)

    # Every layer's alpha comes from a real COVERAGE mask, never from the fill
    # colour's luminance. Deriving alpha from luminance (what this used to do) tied
    # opacity to the colour choice - BODY_ALPHA 0.42 actually rendered at 0.28, the
    # rails and the dashed centre line landed at 0.64 and 0.82 while sharing one
    # constant, and blurred edges faded toward black instead of toward transparent.
    body_cov = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(body_cov, [geometry["polygon"]], 255)
    body_blur = BODY_BLUR | 1
    body_cov = cv2.GaussianBlur(body_cov, (body_blur, body_blur), 0)

    rails_cov = np.zeros((height, width), dtype=np.uint8)
    left = np.round(geometry["left"]).astype(np.int32)
    right = np.round(geometry["right"]).astype(np.int32)
    cv2.polylines(rails_cov, [left], False, 255, RAIL_STROKE_PX, cv2.LINE_AA)
    cv2.polylines(rails_cov, [right], False, 255, RAIL_STROKE_PX, cv2.LINE_AA)

    band_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(band_mask, [geometry["polygon"]], 255)

    dash_cov = np.zeros((height, width), dtype=np.uint8)
    if geometry.get("traj") is not None:
        draw_world_dashes(dash_cov, geometry["traj"], colour=255)
    else:
        draw_centerline_chevrons(dash_cov, geometry["centre"], chevron_phase_m, colour=255,
                                 assume_ordered=bool(geometry.get("ordered")))

    # A mark is road paint only while it stays inside the band; a stroke
    # escaping the rails onto bare asphalt breaks the illusion instantly.
    dash_cov = cv2.bitwise_and(dash_cov, band_mask)

    a_shadow = (shadow_cov.astype(np.float32) / 255.0) * profile_col * CONTACT_SHADOW_ALPHA
    a_body = (body_cov.astype(np.float32) / 255.0) * profile_col * BODY_ALPHA
    a_rails = (rails_cov.astype(np.float32) / 255.0) * profile_col * RAILS_ALPHA
    a_dashes = (dash_cov.astype(np.float32) / 255.0) * profile_col * DASH_ALPHA

    def flat(colour):
        layer = np.empty((height, width, 3), dtype=np.float32)
        layer[:, :] = colour
        return layer

    black = np.zeros((height, width, 3), dtype=np.float32)
    layers = [
        (black, a_shadow),
        (flat(ROAD_PATH_COLOUR), a_body),
        (flat(ROAD_PATH_COLOUR), a_rails),
        (flat(ROAD_PATH_CORE_COLOUR), a_dashes),
    ]

    premul = np.zeros((height, width, 3), dtype=np.float32)
    a_tot = np.zeros((height, width), dtype=np.float32)
    for colour, alpha in layers:
        alpha_col = alpha[:, :, None]
        premul = premul * (1.0 - alpha_col) + colour * alpha_col
        a_tot = a_tot * (1.0 - alpha) + alpha

    return {"premul": premul, "a_tot": a_tot}


def blend_path(frame, overlay, occlusion=None, gain=1.0):
    """Composite the precomputed ribbon overlay.

    occlusion: per-pixel [0,1] map where the ribbon is hidden (behind objects /
    off the road). gain: scalar or per-row (height,) [0,1] opacity multiplier,
    used for the animated reveal/hide of the ribbon.
    """
    if overlay is None:
        return

    a_tot = overlay["a_tot"]
    premul = overlay["premul"]
    height, width = a_tot.shape

    factor = np.ones((height, width), dtype=np.float32)
    if occlusion is not None:
        factor *= (1.0 - occlusion)
    if np.ndim(gain) == 0:
        factor *= float(gain)
    elif np.ndim(gain) == 1:
        factor *= gain.astype(np.float32)[:, None]
    else:
        factor *= gain.astype(np.float32)

    a_eff = a_tot * factor
    contrib = premul * factor[:, :, None]
    blended = frame.astype(np.float32) * (1.0 - a_eff[:, :, None]) + contrib
    frame[:, :] = blended.astype(np.uint8)


def resolve_frame_overlay(road_binary, height, width, static_overlay, static_geometry,
                          aim_state, lookahead_offset=0.0, lane_data=None, lane_state=None,
                          vo_pts=None, vo_state=None, chevron_phase_m=0.0,
                          direct_geometry=None, anchor_geometry=None):
    """Return (overlay, near_v, far_v) for this frame.

    With the road-aimed ribbon enabled and a road mask available, the ribbon
    geometry + overlay are rebuilt per frame so the ribbon follows the road (and
    bends into a turn via lookahead_offset); otherwise the precomputed static
    (trajectory) overlay is reused. When lane data is supplied (UFLDv2 instances
    or a YOLOP mask), the ribbon is additionally centred in the detected ego lane;
    when a VO future path is supplied, it shapes the ribbon through real turns.
    """
    global LAST_RIBBON_GEOMETRY

    # Future-anchored mode, the highest-precedence source: the geometry was
    # traced on the actual street pixels the car later drives over (homography
    # chains to the future frames), so it stays on the road through the exact
    # situations that break the projected sources below -- pull-away heading
    # noise, fast yaw, turn arcs that fold back on themselves.
    if anchor_geometry is not None:
        overlay = build_path_overlay(anchor_geometry, height, width, chevron_phase_m)
        LAST_RIBBON_GEOMETRY = anchor_geometry
        return overlay, anchor_geometry["near_v"], anchor_geometry["far_v"]

    # Direct mode: the geometry IS the measured future path, projected with no
    # filtering between the measurement and the pixels. The aimed machinery
    # below smooths and rate-limits in screen space, which delays the bend --
    # the curve then draws at the wrong distance and the ribbon leaves the
    # street. With a validated, calibrated future path there is nothing to
    # smooth away: the drawn band lies on the driven road by construction.
    if direct_geometry is not None:
        overlay = build_path_overlay(direct_geometry, height, width, chevron_phase_m)
        LAST_RIBBON_GEOMETRY = direct_geometry
        return overlay, direct_geometry["near_v"], direct_geometry["far_v"]

    # Planner mode: the ribbon is the Alpamayo trajectory, projected once for
    # the clip; only the chevron phase animates. The overlay window sits at the
    # planned moment, where those 6.4 s of waypoints are valid, so the shape is
    # not advanced per frame. Occlusion and reveal still apply in blend_path.
    if RIBBON_SOURCE == "planner" and static_geometry is not None:
        overlay = build_path_overlay(static_geometry, height, width, chevron_phase_m)
        LAST_RIBBON_GEOMETRY = static_geometry
        return overlay, static_geometry["near_v"], static_geometry["far_v"]

    if USE_ROAD_AIMED_RIBBON and road_binary is not None:
        lane_center = resolve_lane_center(lane_data, height, width, lane_state)
        geometry = aimed_ribbon_geometry(
            road_binary, height, width, aim_state, lookahead_offset, lane_center,
            vo_pts=vo_pts, vo_state=vo_state)
        overlay = build_path_overlay(geometry, height, width, chevron_phase_m)
        LAST_RIBBON_GEOMETRY = geometry
        return overlay, geometry["near_v"], geometry["far_v"]
    near_v = static_geometry["near_v"] if static_geometry else float(height)
    far_v = static_geometry["far_v"] if static_geometry else 0.0
    LAST_RIBBON_GEOMETRY = static_geometry
    return static_overlay, near_v, far_v


def build_occlusion_mask(result, height, width):
    """Union of occluder (person + vehicle) masks as a feathered [0,1] map.

    Objects in the street sit in front of the far road ribbon, so wherever their
    segmentation mask lands the ribbon should be hidden. Returns None if there
    are no occluder masks in this frame.
    """
    if result.masks is None or result.boxes is None:
        return None

    class_ids = result.boxes.cls.cpu().numpy()
    polygons = result.masks.xy

    occlusion = np.zeros((height, width), dtype=np.uint8)
    any_polygon = False

    for index in range(len(polygons)):
        if index >= len(class_ids):
            break
        if int(class_ids[index]) not in OCCLUDER_CLASS_IDS:
            continue
        polygon = polygons[index]
        if polygon is None or len(polygon) < 3:
            continue
        cv2.fillPoly(occlusion, [np.round(polygon).astype(np.int32)], 255)
        any_polygon = True

    if not any_polygon:
        return None

    if OCCLUSION_FEATHER_PX > 0:
        feather = OCCLUSION_FEATHER_PX | 1
        occlusion = cv2.GaussianBlur(occlusion, (feather, feather), 0)

    return occlusion.astype(np.float32) / 255.0


def travelled_distance_per_frame(vo_track):
    """Cumulative metres driven, per frame, from the VO track's ego poses.

    Chevron marks live at fixed points ON THE GROUND; the pattern phase must
    advance by the distance actually driven, or the marks slide against the
    asphalt. Arclength is robust even when the track's heading failed its
    validation -- position deltas do not carry the yaw bias.
    """
    if vo_track is None:
        return None

    poses = vo_track.get("ego_pose_right_positive")

    if not poses or len(poses) < 2:
        return None

    distances = [0.0]

    for (x0, y0, _p0), (x1, y1, _p1) in zip(poses, poses[1:]):
        distances.append(distances[-1] + math.hypot(x1 - x0, y1 - y0))

    return distances


class LateralMotionTracker(object):
    """Per-frame horizontal scene shift (full-res px), for anchor feed-forward.

    The band's lateral anchors are EMA-smoothed in SCREEN coordinates, so when
    the ego yaw swings the whole image sideways the anchors lag and the band
    detaches from the road (measured at a 10:1 road-to-band motion mismatch).
    Feeding the measured per-frame shift forward moves every anchor WITH the
    world; the smoothing then only fights estimation noise, as intended.
    """

    def __init__(self, frame_width):
        self.scale = frame_width / 320.0
        self.previous = None

    def update(self, frame_bgr):
        gray = cv2.resize(
            cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY), (320, 180)
        ).astype(np.float32)[:90, :]
        shift = 0.0

        if self.previous is not None:
            (dx, _dy), _response = cv2.phaseCorrelate(self.previous, gray)
            shift = clamp(dx * self.scale, -40.0, 40.0)

        self.previous = gray

        return shift


def apply_lateral_feed_forward(shift_px, lane_state, aim_state, vo_state):
    if abs(shift_px) < 1e-6:
        return

    if lane_state and lane_state.get("near_u") is not None:
        lane_state["near_u"] += shift_px

    if aim_state and aim_state.get("far_u") is not None:
        aim_state["far_u"] += shift_px

    if vo_state and vo_state.get("applied") is not None:
        vo_state["applied"] = vo_state["applied"] + shift_px


# Draw the measured future path directly (no aimed-machinery filtering)
# whenever a validated VO track supplies it. The aimed ribbon remains the
# fallback for clips without one.
DIRECT_FUTURE_PATH = os.environ.get("OPTICARVIS_DIRECT_FUTURE_PATH", "1") == "1"


class FuturePathSmoother(object):
    """Light EMA on the future path, in WORLD coordinates.

    Per-frame VO paths carry estimation noise; smoothing them in world space
    damps the jitter without delaying the geometry the way the screen-space
    EMAs did (a bend delayed is a bend drawn at the wrong distance). The path
    is resampled onto a fixed forward grid so frames average like with like.
    """

    GRID_M = np.arange(1.0, 34.0, 0.5)
    ALPHA = 0.45

    def __init__(self):
        self.lateral = None

    def update(self, vo_pts):
        pts = [(float(x), float(y)) for x, y in vo_pts if float(x) > 0.2]

        if len(pts) < 3:
            self.lateral = None
            return None

        pts.sort(key=lambda p: p[0])
        xs = np.array([p[0] for p in pts])
        ys = np.array([p[1] for p in pts])
        grid_lat = np.interp(self.GRID_M, xs, ys, left=ys[0], right=ys[-1])

        if self.lateral is None:
            self.lateral = grid_lat
        else:
            self.lateral = self.lateral + self.ALPHA * (grid_lat - self.lateral)

        # only the span the VO actually measured
        span = self.GRID_M <= xs[-1] + 0.5

        return list(zip(self.GRID_M[span], self.lateral[span]))


def direct_future_geometry(smoother, vo_pts):
    if not DIRECT_FUTURE_PATH or vo_pts is None or smoother is None:
        return None

    traj = smoother.update(vo_pts)

    if traj is None or len(traj) < 3:
        return None

    # Arc-parameterised build: the band ends where the measurement ends. The
    # old y(x) build sampled a fixed 4-21 m forward window and extended the
    # last measured lateral as a constant -- on a 4 m-lookahead track at a turn
    # onset that drew a 17 m near-straight band aimed at the median.
    return build_arc_ribbon_geometry(traj)


# The future-anchor sidecar (future_anchor.py): per-frame polylines of the
# street pixels the car later drives over, found by chaining ground-plane
# homographies to the future frames. No world model stands between those
# anchors and the road, so they survive exactly the moments the flat-ground
# projection breaks: pull-away heading noise, fast yaw, folded-back turn arcs.
USE_FUTURE_ANCHOR = os.environ.get("OPTICARVIS_FUTURE_ANCHOR", "1") == "1"
ANCHOR_GAP_BRIDGE_FRAMES = 5
ANCHOR_RESAMPLE_M = 0.4
ANCHOR_SMOOTH_TAPS = 5


def ground_from_pixel(u, v):
    """Inverse of project_ground_point: image pixel -> flat-ground metres."""
    denom = float(v) - HORIZON_V

    if denom <= 1e-3:
        return None

    x_fwd = CAM_FOCAL_PX * CAM_HEIGHT_M / denom
    y_lat = LATERAL_SIGN * (float(u) - VANISH_U) * x_fwd / CAM_FOCAL_PX

    return x_fwd, y_lat


def build_arc_ribbon_geometry(path_xy):
    """Ribbon geometry from a ground polyline, parameterised by ITS OWN arc.

    build_ribbon_geometry models the path as a single-valued lateral offset
    over a fixed 4-21 m forward window; a real 96-degree turn reaches at most
    ~12 m of forward distance and folds back, which that representation cannot
    express. Here the path is resampled along its arclength, lightly box-
    smoothed IN GROUND COORDINATES (screen-space smoothing is what used to
    delay bends), given rails perpendicular to the local ground tangent, and
    projected. It truncates where the data ends instead of extrapolating.
    """
    pts = [(float(x), float(y)) for x, y in path_xy if float(x) > MIN_FORWARD_M]

    if len(pts) < 3:
        return None

    arr = np.asarray(pts, dtype=np.float64)
    seg = np.hypot(np.diff(arr[:, 0]), np.diff(arr[:, 1]))
    keep = np.concatenate([[True], seg > 1e-6])
    arr = arr[keep]

    if len(arr) < 3:
        return None

    seg = np.hypot(np.diff(arr[:, 0]), np.diff(arr[:, 1]))
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(arc[-1])

    if total < 3.0 * ANCHOR_RESAMPLE_M:
        return None

    grid = np.arange(0.0, total + 1e-9, ANCHOR_RESAMPLE_M)
    xs = np.interp(grid, arc, arr[:, 0])
    ys = np.interp(grid, arc, arr[:, 1])

    if ANCHOR_SMOOTH_TAPS > 1 and len(grid) > ANCHOR_SMOOTH_TAPS:
        kernel = np.ones(ANCHOR_SMOOTH_TAPS) / float(ANCHOR_SMOOTH_TAPS)
        pad = ANCHOR_SMOOTH_TAPS // 2
        xs = np.convolve(np.pad(xs, pad, mode="edge"), kernel, mode="valid")
        ys = np.convolve(np.pad(ys, pad, mode="edge"), kernel, mode="valid")

    tang_x = np.gradient(xs)
    tang_y = np.gradient(ys)
    norm = np.maximum(np.hypot(tang_x, tang_y), 1e-9)
    # right-of-travel ground normal in the (forward, right-lateral) frame
    n_x = -tang_y / norm
    n_y = tang_x / norm

    centre = []
    left = []
    right = []

    for i in range(len(xs)):
        centre_pt = project_ground_point(xs[i], ys[i])
        left_pt = project_ground_point(xs[i] - n_x[i] * RIBBON_HALF_M,
                                       ys[i] - n_y[i] * RIBBON_HALF_M)
        right_pt = project_ground_point(xs[i] + n_x[i] * RIBBON_HALF_M,
                                        ys[i] + n_y[i] * RIBBON_HALF_M)

        if centre_pt is None or left_pt is None or right_pt is None:
            continue

        if centre_pt[1] < HORIZON_V + HORIZON_CLIP_MARGIN_PX:
            continue

        centre.append(centre_pt)
        left.append(left_pt)
        right.append(right_pt)

    if len(centre) < 3:
        return None

    centre_arr = np.array(centre, dtype=np.float32)
    left_arr = np.array(left, dtype=np.float32)
    right_arr = np.array(right, dtype=np.float32)
    polygon = np.round(np.vstack([left_arr, right_arr[::-1]])).astype(np.int32)

    return {
        "traj": None,                    # chevrons, like the aimed band
        "ordered": True,                 # centre already runs near -> far
        "centre": centre_arr,
        "left": left_arr,
        "right": right_arr,
        "polygon": polygon,
        "near_v": float(np.max(centre_arr[:, 1])),
        "far_v": float(np.min(centre_arr[:, 1])),
    }


def parse_future_anchors(data, frame_count):
    """Validate + resolution-scale a future_anchor_track sidecar.

    Returns a per-frame list of Nx3 arrays [u, v, s_m] (or None entries), in
    the render's pixel space, or None when the sidecar cannot be trusted --
    the renderer then falls down the ladder to the direct flat-ground path.
    """
    if data is None:
        return None

    if not USE_FUTURE_ANCHOR:
        print("WARNING: a future-anchor sidecar exists but "
              "OPTICARVIS_FUTURE_ANCHOR=0 -> ignoring it.")
        return None

    if data.get("type") != "future_anchor_track" or int(data.get("version", 0)) != 1:
        print("WARNING: unrecognised future-anchor sidecar "
              "(type=%r version=%r) -> ignoring it."
              % (data.get("type"), data.get("version")))
        return None

    frames = data.get("anchors") or []

    if abs(len(frames) - frame_count) > 2:
        print("WARNING: future-anchor sidecar covers %d frames but the video "
              "has %d -> stale sidecar, ignoring it." % (len(frames), frame_count))
        return None

    scale = float(_RESOLUTION_SCALE_APPLIED or 1.0)
    calib = data.get("calib") or {}
    horizon_ref = HORIZON_V / scale
    vanish_ref = VANISH_U / scale
    h_skew = abs(float(calib.get("horizon_v", horizon_ref)) - horizon_ref)
    u_skew = abs(float(calib.get("vanish_u", vanish_ref)) - vanish_ref)

    if h_skew > 8.0 or u_skew > 8.0:
        print("WARNING: future-anchor sidecar was computed with a different "
              "camera calibration (skew %.1f/%.1f px) -> stale, ignoring it."
              % (h_skew, u_skew))
        return None

    if h_skew > 1.0 or u_skew > 1.0:
        print("WARNING: future-anchor calibration differs slightly from the "
              "render's (%.1f/%.1f px); proceeding." % (h_skew, u_skew))

    out = []
    usable = 0

    for pts in frames:
        if pts and len(pts) >= 3:
            arr = np.asarray(pts, dtype=np.float64)
            arr[:, 0] *= scale
            arr[:, 1] *= scale
            out.append(arr)
            usable += 1
        else:
            out.append(None)

    print("Future-anchored ribbon enabled: %d/%d frames have anchors."
          % (usable, len(out)))

    return out


def load_future_anchors_for_job():
    """The future-anchor sidecar written by future_anchor.py, if present.

    OPTICARVIS_FUTURE_ANCHOR_JSON overrides the path for manual runs.
    """
    if not USE_FUTURE_ANCHOR:
        return None

    path = os.environ.get("OPTICARVIS_FUTURE_ANCHOR_JSON") or workflow_path(
        "ego_trajectory", segment_tag() + "_future_anchors.json")

    if not os.path.isfile(path):
        return None

    try:
        return read_json(path)
    except (ValueError, OSError) as error:
        print("WARNING: could not read future anchors %s (%s); rendering "
              "without them." % (path, type(error).__name__))
        return None


def displace_ground_polyline(ground_pts, offset_fn, s_travelled):
    """Shift each point perpendicular to the local tangent by offset(s_abs).

    ground_pts: [(x_fwd, y_lat, s_m), ...]. Positive offsets displace to the
    ego's right of travel (right-positive, matching every lateral in this
    file). Used to draw the PLAN as a lateral offset riding the anchored
    driven path, so divergence reads as sideways offset on real street pixels.
    """
    if len(ground_pts) < 2:
        return [(x, y) for x, y, _s in ground_pts]

    xs = np.array([p[0] for p in ground_pts])
    ys = np.array([p[1] for p in ground_pts])
    tang_x = np.gradient(xs)
    tang_y = np.gradient(ys)
    norm = np.maximum(np.hypot(tang_x, tang_y), 1e-9)
    n_x = -tang_y / norm
    n_y = tang_x / norm

    out = []

    for i, (x, y, s_m) in enumerate(ground_pts):
        offset = float(offset_fn(s_travelled + s_m))
        out.append((x + n_x[i] * offset, y + n_y[i] * offset))

    return out


def anchor_geometry_for_frame(anchor_frames, index, offset_fn=None, s_travelled=0.0):
    """Ribbon geometry from this frame's future anchors, or None.

    The anchors are already street pixels; they are unprojected to ground
    metres only as a DRAWING parameterisation (resampling, smoothing, rail
    offsets) and reprojected with the same model, which round-trips exactly.
    """
    if anchor_frames is None or index >= len(anchor_frames):
        return None

    pts = anchor_frames[index]

    if pts is None:
        return None

    ground = []

    for u, v, s_m in pts:
        point = ground_from_pixel(u, v)

        if point is None:
            continue

        ground.append((point[0], point[1], float(s_m)))

    if len(ground) < 3:
        return None

    if offset_fn is not None:
        path = displace_ground_polyline(ground, offset_fn, s_travelled)
    else:
        path = [(x, y) for x, y, _s in ground]

    return build_arc_ribbon_geometry(path)


def planner_offset_function(planner_track, vo_track, fps):
    """Signed lateral offset (metres, right-positive) of the PLAN from the
    DRIVEN path at matched absolute arclength, as a callable, or None.

    Both curves live in the VO track's right-positive world frame; the plan
    was already flipped once by PLANNER_LATERAL_SIGN in load_planner_track,
    and no further sign appears here -- introducing one would be the mirror
    bug the load_planner_track comment warns about. Outside the plan's
    arclength span the offset fades to zero: the band then shows the driven
    path rather than an extrapolated plan.
    """
    if planner_track is None or not fps:
        return None

    poses = ego_poses_from_track(vo_track)

    if poses is None:
        return None

    frame_t0 = int(round(planner_track["t0_s"] * fps))

    if not 0 <= frame_t0 < len(poses):
        return None

    arc = [0.0]

    for (x0, y0, _p0), (x1, y1, _p1) in zip(poses, poses[1:]):
        arc.append(arc[-1] + math.hypot(x1 - x0, y1 - y0))

    arc = np.asarray(arc)
    drv_x = np.array([p[0] for p in poses], dtype=np.float64)
    drv_y = np.array([p[1] for p in poses], dtype=np.float64)

    x0, y0, psi0 = poses[frame_t0]
    cos0, sin0 = math.cos(psi0), math.sin(psi0)
    plan_x = []
    plan_y = []

    for px, py in planner_track["points"]:
        plan_x.append(x0 + cos0 * px - sin0 * py)
        plan_y.append(y0 + sin0 * px + cos0 * py)

    plan_x = np.asarray([x0] + plan_x)
    plan_y = np.asarray([y0] + plan_y)
    plan_arc = np.concatenate(
        [[0.0], np.cumsum(np.hypot(np.diff(plan_x), np.diff(plan_y)))])

    s_base = float(arc[frame_t0])
    span = float(plan_arc[-1])

    if span < 1.0:
        return None

    def offset(s_abs):
        s_rel = s_abs - s_base

        if s_rel < 0.0 or s_rel > span:
            return 0.0

        px = float(np.interp(s_rel, plan_arc, plan_x))
        py = float(np.interp(s_rel, plan_arc, plan_y))
        dx = float(np.interp(s_abs, arc, drv_x))
        dy = float(np.interp(s_abs, arc, drv_y))
        tx = float(np.interp(s_abs + 0.5, arc, drv_x)) - float(np.interp(s_abs - 0.5, arc, drv_x))
        ty = float(np.interp(s_abs + 0.5, arc, drv_y)) - float(np.interp(s_abs - 0.5, arc, drv_y))
        norm = math.hypot(tx, ty)

        if norm < 1e-6:
            return 0.0

        # right-of-travel normal; the offset is the divergence resolved onto it
        value = ((px - dx) * (-ty) + (py - dy) * tx) / norm
        # fade at both ends of the plan span so the band never steps
        fade = min(1.0, s_rel / 1.0, (span - s_rel) / 2.0)

        return value * max(0.0, fade)

    return offset


def load_planner_track():
    """The full planner plan for per-frame advancing: points, yaws, t0, step.

    Returns None when the context lacks the yaw/t0 fields (older context files)
    -- the planner ribbon then stays static, the pre-advance behaviour.
    """
    if not os.path.isfile(ALPAMAYO_CONTEXT_JSON):
        return None

    with open(ALPAMAYO_CONTEXT_JSON, "r", encoding="utf-8") as handle:
        context = json.load(handle)

    raw_points = context.get("trajectory_points_xyz")
    raw_yaws = context.get("trajectory_yaw_rad")
    t0 = context.get("planner_t0_local_s")
    step = context.get("trajectory_dt_s")

    if not raw_points or not raw_yaws or t0 is None or not step:
        return None

    count = min(len(raw_points), len(raw_yaws))

    if count < 4:
        return None

    # Mirror FLU into the renderer's right-positive frame consistently:
    # lateral AND yaw flip together, or the frame transfer rotates the wrong way.
    points = [
        (float(p[0]), PLANNER_LATERAL_SIGN * float(p[1]))
        for p in raw_points[:count]
    ]
    yaws = [PLANNER_LATERAL_SIGN * float(y) for y in raw_yaws[:count]]

    return {"points": points, "yaws": yaws, "t0_s": float(t0), "step_s": float(step)}


def ego_poses_from_track(vo_track):
    """Per-frame actual ego poses for world-anchoring the planner ribbon.

    Only from a track whose heading passed the scene-pan validation: anchoring
    on a biased heading re-detaches the ribbon from the street.
    """
    if vo_track is None:
        return None

    if (vo_track.get("heading_validation") or {}).get("valid") is not True:
        return None

    poses = vo_track.get("ego_pose_right_positive")

    return poses if poses and len(poses) > 2 else None


def anchor_planner_trajectory(track, clip_time_s, ego_poses, fps):
    """The plan, pinned to the world where it was made, seen from the car's
    ACTUAL pose now.

    The plan-pose advance assumes the car follows the plan; a human drove this
    footage, and where the driver diverges the ribbon detaches from the street.
    World-anchoring uses the validated VO pose instead: the planned curve stays
    on the road at the spot it was planned for, and approaches exactly as fast
    as the car actually approaches it.
    """
    frame_t0 = int(round(track["t0_s"] * fps))
    frame_now = int(round(clip_time_s * fps))

    if not (0 <= frame_t0 < len(ego_poses) and 0 <= frame_now < len(ego_poses)):
        return None

    x0, y0, psi0 = ego_poses[frame_t0]
    xt, yt, psit = ego_poses[frame_now]

    # Travelled displacement, expressed in the ego frame at the planned moment
    # (the frame the plan's points live in).
    dx = xt - x0
    dy = yt - y0
    cos0, sin0 = math.cos(psi0), math.sin(psi0)
    travelled_x = cos0 * dx + sin0 * dy
    travelled_y = -sin0 * dx + cos0 * dy

    delta_psi = psit - psi0
    cos_d, sin_d = math.cos(delta_psi), math.sin(delta_psi)

    remaining = []

    for px, py in track["points"]:
        qx = px - travelled_x
        qy = py - travelled_y
        remaining.append((cos_d * qx + sin_d * qy, -sin_d * qx + cos_d * qy))

    remaining = [(x, y) for x, y in remaining if x > MIN_FORWARD_M]

    return remaining if len(remaining) >= 2 else None


def advance_planner_trajectory(track, clip_time_s):
    """The remaining plan, re-expressed in the pose the plan has reached.

    Fallback for clips without a validated VO track: assumes the car follows
    its own plan. anchor_planner_trajectory is the honest version.
    """
    elapsed = max(0.0, clip_time_s - track["t0_s"])
    position = elapsed / track["step_s"]
    index = int(position)
    frac = position - index
    points = track["points"]
    yaws = track["yaws"]

    if index >= len(points) - 2:
        return None

    # Pose along the plan, interpolated between waypoints.
    px = points[index][0] + frac * (points[index + 1][0] - points[index][0])
    py = points[index][1] + frac * (points[index + 1][1] - points[index][1])
    yaw = yaws[index] + frac * (yaws[index + 1] - yaws[index])

    cos_y = math.cos(yaw)
    sin_y = math.sin(yaw)
    remaining = []

    for x, y in points[index + 1:]:
        dx = x - px
        dy = y - py
        remaining.append((cos_y * dx + sin_y * dy, -sin_y * dx + cos_y * dy))

    remaining = [(x, y) for x, y in remaining if x > MIN_FORWARD_M]

    if len(remaining) < 2:
        return None

    return remaining


def planner_geometry_for_frame(track, frame_index, fps, ego_poses=None):
    """Per-frame ribbon geometry from the advanced plan, or None when the
    plan's horizon is exhausted (the caller falls back to perception).

    With validated actual ego poses the plan is anchored to the world;
    otherwise it advances along its own predicted poses. Either way the plan
    ends at its horizon: past t0 + 6.4 s there is nothing honest to draw.
    """
    if track is None or not fps:
        return None

    clip_time_s = frame_index / fps
    horizon_s = track["t0_s"] + (len(track["points"]) - 1) * track["step_s"]

    if clip_time_s > horizon_s:
        return None

    remaining = None

    if ego_poses is not None:
        remaining = anchor_planner_trajectory(track, clip_time_s, ego_poses, fps)

    if remaining is None:
        remaining = advance_planner_trajectory(track, clip_time_s)

    if remaining is None:
        return None

    geometry = build_ribbon_geometry(remaining)

    if geometry is not None:
        geometry["traj"] = None  # chevrons, matching the perception ribbon

    return geometry


def load_path_geometry():
    traj = load_trajectory_points()
    if traj is None:
        traj = straight_trajectory_points()
    geometry = build_ribbon_geometry(traj)

    # In planner mode the ribbon animates like the perception ribbon: chevrons
    # flowing along the centreline rather than static world dashes. traj=None
    # is the switch build_path_overlay keys on, and it also keeps the geometry
    # dump/compositor contract, which stores the centreline alone.
    if RIBBON_SOURCE == "planner" and geometry is not None:
        geometry["traj"] = None

    return geometry


def draw_calibration_guides(frame):
    width = frame.shape[1]

    cv2.line(
        frame,
        (0, int(round(HORIZON_V))),
        (width, int(round(HORIZON_V))),
        (0, 0, 255),
        GUIDE_LINE_THICKNESS,
        cv2.LINE_AA,
    )
    cv2.circle(
        frame,
        (int(round(VANISH_U)), int(round(HORIZON_V))),
        GUIDE_VP_RADIUS_PX,
        (0, 0, 255),
        -1,
    )

    for dist in (5, 10, 15, 20, 30, 40):
        projected = project_ground_point(float(dist), 0.0)
        if projected is None:
            continue
        u = int(round(projected[0]))
        v = int(round(projected[1]))
        cv2.circle(frame, (u, v), GUIDE_TICK_RADIUS_PX, (255, 255, 255), -1)
        cv2.putText(
            frame,
            str(dist) + "m",
            (u + GUIDE_LABEL_OFFSET_PX, v),
            cv2.FONT_HERSHEY_SIMPLEX,
            GUIDE_FONT_SCALE,
            (255, 255, 255),
            GUIDE_TEXT_THICKNESS,
            cv2.LINE_AA,
        )


def preview_calibration(image_path, output_path, draw_guides=True):
    frame = cv2.imread(image_path)

    if frame is None:
        print("Could not read calibration image:")
        print(image_path)
        raise SystemExit(1)

    apply_calibration_overrides()

    height, width = frame.shape[:2]

    # Without this, tuning from a 4K still would bake compensating garbage into
    # the calibration JSON.
    apply_resolution_scaling(width, height)

    geometry = load_path_geometry()
    overlay = build_path_overlay(geometry, height, width)
    blend_path(frame, overlay, None)

    if draw_guides:
        draw_calibration_guides(frame)

    cv2.imwrite(output_path, frame)
    print("Wrote calibration preview:", output_path)


def box_to_spatial_score(x1, y1, x2, y2, width, height):
    box_width = max(0.0, float(x2 - x1))
    box_height = max(0.0, float(y2 - y1))

    x_center_norm = ((float(x1) + float(x2)) / 2.0) / float(width)
    y_center_norm = ((float(y1) + float(y2)) / 2.0) / float(height)
    width_norm = box_width / float(width)
    height_norm = box_height / float(height)
    area_norm = width_norm * height_norm

    size_score = clamp(height_norm / 0.38, 0.0, 1.0)
    lower_image_score = clamp((y_center_norm - 0.25) / 0.70, 0.0, 1.0)
    area_score = clamp(area_norm / 0.08, 0.0, 1.0)

    proximity_score = (
        0.55 * size_score
        + 0.30 * lower_image_score
        + 0.15 * area_score
    )

    centre_relevance = 1.0 - clamp(abs(x_center_norm - 0.50) / 0.50, 0.0, 1.0)

    spatial_score = (
        0.80 * proximity_score
        + 0.20 * centre_relevance
    )

    return round(clamp(spatial_score, 0.0, 1.0), 3)


def class_display_name(class_id):
    if class_id == PERSON_CLASS_ID:
        return "person"
    return COCO_VEHICLE_NAMES.get(class_id, str(class_id))


def extract_detections(result, allowed_ids, width, height):
    """Qualifying detections for the given class ids (class + track id + mask).

    Selection/truncation happens later in select_and_smooth, which needs the full
    tracked pool, so nothing is dropped here beyond the confidence / proximity
    gates.
    """
    detections = []

    boxes = result.boxes

    if boxes is None or boxes.xyxy is None:
        return detections

    xyxy_values = boxes.xyxy.cpu().numpy()
    class_ids = boxes.cls.cpu().numpy()
    confidences = boxes.conf.cpu().numpy()
    track_ids = boxes.id.cpu().numpy() if boxes.id is not None else None
    mask_polygons = result.masks.xy if result.masks is not None else None

    for index in range(len(xyxy_values)):
        class_id = int(class_ids[index])

        if class_id not in allowed_ids:
            continue

        confidence = float(confidences[index])

        if confidence < CONFIDENCE_THRESHOLD:
            continue

        x1 = int(clamp(float(xyxy_values[index][0]), 0, width - 1))
        y1 = int(clamp(float(xyxy_values[index][1]), 0, height - 1))
        x2 = int(clamp(float(xyxy_values[index][2]), 0, width - 1))
        y2 = int(clamp(float(xyxy_values[index][3]), 0, height - 1))

        if x2 <= x1 or y2 <= y1:
            continue

        score = box_to_spatial_score(x1, y1, x2, y2, width, height)

        if score < MIN_PROXIMITY_SCORE:
            continue

        track_id = int(track_ids[index]) if track_ids is not None else -1

        mask_poly = None
        if mask_polygons is not None and index < len(mask_polygons):
            polygon = mask_polygons[index]
            if polygon is not None and len(polygon) >= 3:
                mask_poly = np.round(polygon).astype(np.int32)

        detections.append(
            {
                "box": [x1, y1, x2, y2],
                "class_id": class_id,
                "class_name": class_display_name(class_id),
                "track_id": track_id,
                "confidence": round(confidence, 3),
                "spatial_score": score,
                "mask": mask_poly,
            }
        )

    return detections


def select_and_smooth(detections, state, max_count):
    """Pick up to max_count detections with track-ID hysteresis and EMA boxes.

    An incumbent (highlighted last frame) gets a SELECT_STICKY_MARGIN score bonus,
    so it is only displaced when a rival clearly outranks it. Each chosen box is
    exponentially smoothed per track id so the highlight does not jitter.
    """
    smoothed = state["smoothed"]
    previous = set(state["selected"])

    tracked = [d for d in detections if d["track_id"] != -1]
    pool = tracked if tracked else detections

    def effective_score(detection):
        bonus = SELECT_STICKY_MARGIN if detection["track_id"] in previous else 0.0
        return detection["spatial_score"] + bonus

    ranked = sorted(pool, key=effective_score, reverse=True)
    chosen = ranked[:max_count]

    for detection in chosen:
        box = np.array(detection["box"], dtype=np.float32)
        track_id = detection["track_id"]
        if track_id != -1:
            previous_box = smoothed.get(track_id)
            if previous_box is None:
                smoothed_box = box
            else:
                smoothed_box = BOX_SMOOTH_ALPHA * box + (1.0 - BOX_SMOOTH_ALPHA) * previous_box
            smoothed[track_id] = smoothed_box
            detection["draw_box"] = [int(round(v)) for v in smoothed_box]
        else:
            detection["draw_box"] = list(detection["box"])

    # Forget tracks that are no longer visible so the memory stays bounded.
    visible = {d["track_id"] for d in detections}
    for track_id in list(smoothed.keys()):
        if track_id not in visible:
            del smoothed[track_id]

    state["selected"] = [d["track_id"] for d in chosen if d["track_id"] != -1]

    return chosen


def fit_depth_to_metric(depth_norm, road_soft):
    """Scale+shift align relative depth to metric using flat-ground road pixels.

    Monocular relative depth has an unknown scale/offset. On road pixels we know
    the metric distance from the pinhole ground model, so we fit 1/Z = a*D + b
    (linear in inverse depth) and can then read any pixel out in metres,
    consistent with the calibrated geometry. Returns (a, b) or None.
    """
    if road_soft is None:
        return None
    ys, xs = np.where(road_soft > 0.5)
    keep = ys > (HORIZON_V + FIT_DEPTH_HORIZON_EXCLUDE_PX)
    ys, xs = ys[keep], xs[keep]
    if len(ys) < FIT_DEPTH_MIN_ROAD_PX:
        return None
    if len(ys) > 6000:
        idx = np.linspace(0, len(ys) - 1, 6000).astype(int)
        ys, xs = ys[idx], xs[idx]
    z = (CAM_FOCAL_PX * CAM_HEIGHT_M) / (ys.astype(np.float64) - HORIZON_V)
    d = depth_norm[ys, xs].astype(np.float64)
    matrix = np.vstack([d, np.ones_like(d)]).T
    solution = np.linalg.lstsq(matrix, 1.0 / z, rcond=None)[0]
    return float(solution[0]), float(solution[1])


def object_depth_distance(depth_norm, fit, box, mask):
    """Metric distance (m) for one object from the aligned depth map."""
    if fit is None:
        return None
    a, b = fit
    height, width = depth_norm.shape

    if mask is not None and len(mask) >= 3:
        x, y, w, h = cv2.boundingRect(mask)
        x = max(0, x)
        y = max(0, y)
        w = min(w, width - x)
        h = min(h, height - y)
        if w <= 0 or h <= 0:
            return None
        region_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(region_mask, [mask - [x, y]], 255)
        values = depth_norm[y:y + h, x:x + w][region_mask > 0]
    else:
        x1, y1, x2, y2 = box
        values = depth_norm[y1:y2, x1:x2].reshape(-1)

    if values.size == 0:
        return None

    inverse = a * float(np.median(values)) + b
    if inverse <= 1e-6:
        return None
    return 1.0 / inverse


def attach_distances(selected, depth_norm, fit):
    """Set detection['distance_m'] from the aligned depth map (None if no depth)."""
    for detection in selected:
        distance = None
        if depth_norm is not None:
            distance = object_depth_distance(
                depth_norm,
                fit,
                detection.get("draw_box", detection["box"]),
                detection.get("mask"),
            )
        detection["distance_m"] = distance


def draw_highlights(frame, selected, close_colour, far_colour, show_class=False):
    """Draw contour highlights (or box fallback) with a class/distance label."""
    height, width = frame.shape[:2]

    for detection in selected:
        score = detection["spatial_score"]
        colour = close_colour if score >= CLOSE_BOX_SCORE else far_colour
        box = detection.get("draw_box", detection["box"])
        polygon = detection.get("mask")

        if polygon is not None and len(polygon) >= 3:
            x, y, w, h = cv2.boundingRect(polygon)
            x = max(0, x)
            y = max(0, y)
            w = min(w, width - x)
            h = min(h, height - y)

            if w > 0 and h > 0:
                roi = frame[y:y + h, x:x + w]
                mask_roi = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(mask_roi, [polygon - [x, y]], 255)
                if HIGHLIGHT_FEATHER_PX > 0:
                    feather = HIGHLIGHT_FEATHER_PX | 1
                    mask_roi = cv2.GaussianBlur(mask_roi, (feather, feather), 0)
                alpha = (mask_roi.astype(np.float32) / 255.0 * HIGHLIGHT_FILL_ALPHA)[:, :, None]
                colour_arr = np.array(colour, dtype=np.float32)
                roi[:] = (roi.astype(np.float32) * (1.0 - alpha) + colour_arr * alpha).astype(np.uint8)

            cv2.polylines(frame, [polygon], True, colour, CONTOUR_THICKNESS, cv2.LINE_AA)
        else:
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), colour, 2)

        if SHOW_DISTANCE_LABEL:
            distance = detection.get("distance_m")
            if distance is None:
                distance = ground_distance_from_row(box[3])
            parts = []
            if show_class:
                parts.append(detection.get("class_name", ""))
            if distance is not None:
                parts.append("%.0f m" % distance)
            label = " ".join(part for part in parts if part)
            if label:
                label_x = box[0]
                label_y = max(int(LABEL_TOP_CLAMP_PX), box[1] - int(LABEL_ABOVE_BOX_PX))
                cv2.putText(frame, label, (label_x, label_y),
                            cv2.FONT_HERSHEY_SIMPLEX, LABEL_FONT_SCALE, TEXT_BACKGROUND,
                            LABEL_OUTLINE_THICKNESS, cv2.LINE_AA)
                cv2.putText(frame, label, (label_x, label_y),
                            cv2.FONT_HERSHEY_SIMPLEX, LABEL_FONT_SCALE, colour,
                            LABEL_TEXT_THICKNESS, cv2.LINE_AA)

    return len(selected)


def explanation_enabled(effect_plan):
    """Whether the gate/plan requested a visualization (False -> render clean).

    When Gemma 4 decides the behaviour is obvious (proper_time_to_explain=False),
    the planner sets display_target to 'none'; the renderer then produces a clean
    pass-through with no ribbon, highlights, or label.
    """
    target = effect_plan.get("display_target", "none")
    return bool(target) and target != "none"


def _smoothstep(x):
    x = 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)
    return x * x * (3.0 - 2.0 * x)


def build_anim_schedule(timeline, frame_count, fps):
    """Per-frame ramp [0,1] and label from the sliding-window gate timeline.

    Window decisions get hysteresis (bridge short gaps, enforce a minimum on-
    duration) so brief flips do not flicker, then each resulting active interval
    is ramped up at its start (intro) and down at its end (outro).
    """
    windows = timeline.get("windows", [])

    raw = np.zeros(frame_count, dtype=bool)
    win_label = [""] * frame_count
    for window in windows:
        f0 = max(0, int(round(window["t_start"] * fps)))
        f1 = min(frame_count, int(round(window["t_end"] * fps)))
        if window.get("proper_time_to_explain"):
            raw[f0:f1] = True
            text = window.get("passenger_facing_text", "") or ""
            for fi in range(f0, f1):
                if not win_label[fi]:
                    win_label[fi] = text

    off_delay = max(1, int(round(ANIM_OFF_DELAY_S * fps)))
    min_on = max(1, int(round(ANIM_MIN_ON_S * fps)))

    # Bridge short off-gaps between on-runs.
    active = raw.copy()
    i = 0
    while i < frame_count:
        if not active[i]:
            j = i
            while j < frame_count and not active[j]:
                j += 1
            if 0 < i and j < frame_count and (j - i) <= off_delay:
                active[i:j] = True
            i = j
        else:
            i += 1

    # Collect intervals and enforce a minimum on-duration.
    intervals = []
    i = 0
    while i < frame_count:
        if active[i]:
            a = i
            while i + 1 < frame_count and active[i + 1]:
                i += 1
            b = i
            if b - a + 1 < min_on:
                b = min(frame_count - 1, a + min_on - 1)
            intervals.append((a, b))
            i = b + 1
        else:
            i += 1

    ramp = np.zeros(frame_count, dtype=np.float32)
    label = [""] * frame_count
    start_f = max(1.0, ANIM_START_S * fps)
    end_f = max(1.0, ANIM_END_S * fps)
    for (a, b) in intervals:
        text = ""
        for fi in range(a, b + 1):
            if win_label[fi]:
                text = win_label[fi]
                break
        for fi in range(a, b + 1):
            up = _smoothstep((fi - a) / start_f)
            down = _smoothstep((b - fi) / end_f)
            ramp[fi] = min(up, down)
            label[fi] = text

    return ramp, label


def reveal_rows_for(ramp_value, near_v, far_v, height):
    """Per-row [0,1] reveal for the ribbon at a given animation ramp value.

    ramp 0 -> nothing; ramp 1 -> whole ribbon. In between the ribbon is drawn on
    from the near end (bottom) outward to the far end, so raising the ramp sweeps
    it on and lowering it retracts it far-to-near.
    """
    rows = np.arange(height, dtype=np.float32)
    threshold = near_v - ramp_value * (near_v - far_v)
    return np.clip((rows - threshold) / ANIM_REVEAL_SOFT_PX, 0.0, 1.0)


def parse_vo_track(vo_track):
    """Validated per-frame future paths from an ego_trajectory.py track file.

    Shared by both render paths; previously only the timeline path could take
    a VO track at all, so the batch's renders could never follow a real turn.
    """
    if vo_track is None:
        return None

    if not USE_VO_TRAJECTORY:
        print("WARNING: a VO track was supplied but OPTICARVIS_VO_TRAJECTORY is not set"
              " -> ignoring it (VO turn shaping is off by default).")
        return None

    validation = vo_track.get("heading_validation") or {}

    if validation.get("valid") is False:
        print("VO track failed its heading validation (%s); ribbon stays "
              "straight in the ego lane." % validation.get("method", "unknown"))
        return None

    vo_trajs = vo_track.get("future_trajectories")

    if not vo_trajs or not any(vo_trajs):
        print("WARNING: VO track has no 'future_trajectories' -> VO turn shaping inactive.")
        return None

    convention = vo_track.get("lateral_convention")
    if convention != "right_positive":
        print("WARNING: VO track lateral_convention=%r (expected 'right_positive');"
              " regenerate it with ego_trajectory.py or turns may be mirrored."
              % (convention,))

    non_empty = sum(1 for p in vo_trajs if p)
    print("VO turn shaping enabled: %d/%d frames have a future path."
          % (non_empty, len(vo_trajs)))

    return vo_trajs


def load_vo_track_for_job():
    """The batch's VO track, written by ego_trajectory.py as a pipeline stage.

    Returns None when absent -- the ribbon then stays straight in the ego lane,
    which is the documented conservative fallback.
    """
    # ego_trajectory.py's own default output path for this job.
    path = workflow_path("ego_trajectory", segment_tag() + "_ego_trajectory.json")

    if not USE_VO_TRAJECTORY or not os.path.isfile(path):
        return None

    try:
        return read_json(path)
    except (ValueError, OSError) as error:
        print("WARNING: could not read VO track %s (%s); rendering without it."
              % (path, type(error).__name__))
        return None


def render_video_timeline(timeline, ego_track=None, vo_track=None):
    """Render both outputs with the overlay animated on/off per the gate timeline."""
    apply_calibration_overrides()

    fps, width, height, frame_count = get_video_metadata(INPUT_VIDEO)

    # This path had no resolution handling at all, and it is the one
    # render_timeline_clip.py drives for every shipped video.
    apply_resolution_scaling(width, height)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))
    writer_vehicles = cv2.VideoWriter(OUTPUT_VIDEO_VEHICLES, fourcc, fps, (width, height))
    if not writer.isOpened() or not writer_vehicles.isOpened():
        print("Could not open output video writer(s).")
        raise SystemExit(1)

    geometry = load_path_geometry()
    overlay = build_path_overlay(geometry, height, width)
    travelled_m = travelled_distance_per_frame(vo_track)
    lateral_tracker = LateralMotionTracker(width)
    path_smoother = FuturePathSmoother()
    planner_track = load_planner_track() if RIBBON_SOURCE == "planner" else None
    planner_ego_poses = (
        ego_poses_from_track(vo_track) if RIBBON_SOURCE == "planner" else None
    )

    if RIBBON_SOURCE == "planner" and planner_track is None:
        print("Planner ribbon requested but the context lacks yaw/t0; "
              "drawing the static planner path.")
    elif RIBBON_SOURCE == "planner":
        print("Planner ribbon anchoring: %s"
              % ("world (validated VO poses)" if planner_ego_poses
                 else "plan poses (no validated VO)"))

    planner_offset = (planner_offset_function(planner_track, vo_track, fps)
                      if RIBBON_SOURCE == "planner" else None)
    anchor_frames = parse_future_anchors(load_future_anchors_for_job(), frame_count)

    # In planner mode the anchors only serve as the street-true carrier for
    # the plan's lateral offset; without a computable offset, drawing the bare
    # driven path would misrepresent the plan, so the planner machinery below
    # keeps the frame instead.
    if RIBBON_SOURCE == "planner" and planner_offset is None and anchor_frames is not None:
        print("Planner ribbon: no plan-vs-driven offset available; "
              "anchors unused, falling back to the planner projection.")
        anchor_frames = None

    anchor_last = {"geometry": None, "age": 0}

    ramp, label_per_frame = build_anim_schedule(timeline, frame_count, fps)
    on_frames = int((ramp > 0.001).sum())
    print("timeline: %d/%d frames show the overlay." % (on_frames, frame_count))

    # Post-hoc restyling: dump the per-frame geometry the models produced, so
    # src/restyle_render.py can re-composite any style without the models. On
    # by default -- a batch that skips it forfeits cheap restyles forever.
    dump = None

    if os.environ.get("OPTICARVIS_DUMP_GEOMETRY", "1") == "1":
        from overlay_geometry_dump import GeometryDump
        from pipeline_common import WORKFLOW_OUTPUTS

        dump = GeometryDump(
            os.path.join(WORKFLOW_OUTPUTS, "overlay_geometry"),
            segment_tag(),
            {
                # Absolute: the render may be launched from src/ with a
                # relative clip path, and the compositor runs from anywhere.
                "clip_video": os.path.abspath(INPUT_VIDEO),
                "fps": fps,
                "width": width,
                "height": height,
                "frame_count": frame_count,
                # Effective (post-scaling, post-calibration-override) camera
                # constants: the compositor re-derives ribbon edges and chevron
                # sizes from these.
                "camera": {
                    "horizon_v": float(HORIZON_V),
                    "vanish_u": float(VANISH_U),
                    "focal_px": float(CAM_FOCAL_PX),
                    "cam_height_m": float(CAM_HEIGHT_M),
                },
                "resolution_scale": float(_RESOLUTION_SCALE_APPLIED or 1.0),
                "chevron_speed_mps": float(CHEVRON_SPEED_MPS),
            },
        )
        print("Overlay geometry dump:", dump.geometry_path)

    cum_pan = None
    la_frames = 0
    if USE_EGO_LOOKAHEAD and ego_track is not None:
        cum_pan = ego_track.get("cum_pan_px")
        la_frames = int(round(LOOKAHEAD_S * fps))
        print("Ego look-ahead enabled: %.1fs (%d frames ahead)." % (LOOKAHEAD_S, la_frames))
    elif ego_track is not None:
        print("WARNING: an ego track was supplied but OPTICARVIS_EGO_LOOKAHEAD is not set"
              " -> ignoring it (the look-ahead is disabled by default).")

    vo_trajs = parse_vo_track(vo_track)

    scene = None
    road_seg_enabled = USE_ROAD_SEGMENTATION
    depth_enabled = USE_DEPTH_DISTANCE
    lane_enabled = USE_LANE_CENTERING
    try:
        import scene_models as scene
        if road_seg_enabled:
            scene.load_road_model()
        if depth_enabled:
            scene.load_depth_model()
    except Exception as error:
        print("Scene models unavailable (%s); using object-mask occlusion." % error)
        road_seg_enabled = False
        depth_enabled = False
        lane_enabled = False

    if lane_enabled:
        try:
            if LANE_SOURCE == "ufldv2":
                scene.load_lane_instance_model()
                print("Ego-lane centering enabled (UFLDv2 lane instances).")
            else:
                scene.load_lane_model()
                print("Ego-lane centering enabled (YOLOP lane lines).")
        except Exception as error:
            print("Lane model unavailable (%s); ribbon stays straight in the ego lane." % error)
            lane_enabled = False

    model = YOLO(MODEL_NAME)
    results = model.track(
        source=INPUT_VIDEO, stream=True, imgsz=IMAGE_SIZE, conf=CONFIDENCE_THRESHOLD,
        classes=OCCLUDER_CLASS_IDS, tracker=TRACKER_NAME, persist=True, verbose=False,
    )

    person_state = {"smoothed": {}, "selected": []}
    vehicle_state = {"smoothed": {}, "selected": []}
    vehicle_ids = set(VEHICLE_CLASS_IDS)
    aim_state = {}
    lane_state = {}
    vo_state = {}
    lookahead_smooth = None

    rendered_frames = 0
    for index, result in enumerate(results):
        ramp_value = float(ramp[index]) if index < len(ramp) else 0.0
        label_text = label_per_frame[index] if index < len(label_per_frame) else ""
        orig = result.orig_img

        if ramp_value <= 0.001:
            writer.write(orig)
            writer_vehicles.write(orig)
            rendered_frames += 1

            if dump is not None:
                dump.frame(index, 0.0, "", None, [], [], None)

            continue

        if road_seg_enabled:
            road_binary = scene.road_mask(orig)
            feather = ROAD_MASK_FEATHER_PX | 1
            road_soft = cv2.GaussianBlur(
                (road_binary * 255).astype(np.uint8), (feather, feather), 0
            ).astype(np.float32) / 255.0
            occlusion = 1.0 - road_soft
        else:
            road_binary = None
            road_soft = None
            occlusion = build_occlusion_mask(result, height, width)

        depth_norm = None
        depth_fit = None
        if depth_enabled:
            depth_norm = scene.depth_map(orig)
            fit_mask = road_soft if road_soft is not None else scene.road_mask(orig).astype(np.float32)
            depth_fit = fit_depth_to_metric(depth_norm, fit_mask)

        lookahead_offset = 0.0
        if cum_pan is not None and index < len(cum_pan):
            j = min(index + la_frames, len(cum_pan) - 1)
            raw_lookahead = EGO_YAW_SIGN * EGO_YAW_GAIN * (cum_pan[j] - cum_pan[index])
            # Deadband (soft threshold) so noise-level yaw reads as straight, then EMA.
            if abs(raw_lookahead) <= LOOKAHEAD_DEADBAND_PX:
                raw_lookahead = 0.0
            else:
                raw_lookahead -= np.sign(raw_lookahead) * LOOKAHEAD_DEADBAND_PX
            lookahead_smooth = raw_lookahead if lookahead_smooth is None else (
                LOOKAHEAD_SMOOTH * raw_lookahead + (1.0 - LOOKAHEAD_SMOOTH) * lookahead_smooth)
            lookahead_offset = lookahead_smooth

        if lane_enabled:
            lane_data = (scene.lane_instances(orig) if LANE_SOURCE == "ufldv2"
                         else scene.lane_line_mask(orig))
        else:
            lane_data = None

        vo_pts = vo_trajs[index] if (vo_trajs is not None and index < len(vo_trajs)) else None

        apply_lateral_feed_forward(
            lateral_tracker.update(orig), lane_state, aim_state, vo_state)

        frame_geometry = geometry

        if RIBBON_SOURCE == "planner":
            frame_geometry = planner_geometry_for_frame(planner_track, index, fps, planner_ego_poses)

        # phase = distance actually driven: marks hold their ground positions
        phase_m = (
            travelled_m[index]
            if travelled_m is not None and index < len(travelled_m)
            else (index / fps) * CHEVRON_SPEED_MPS if fps else 0.0
        )

        anchor_geom = anchor_geometry_for_frame(
            anchor_frames, index, offset_fn=planner_offset,
            s_travelled=(travelled_m[index]
                         if travelled_m is not None and index < len(travelled_m) else 0.0))

        # Bridge short anchor gaps by holding the last geometry: a brief
        # freeze reads better than the band popping to a different source.
        if anchor_geom is not None:
            anchor_last["geometry"] = anchor_geom
            anchor_last["age"] = 0
        elif (anchor_last["geometry"] is not None
                and anchor_last["age"] < ANCHOR_GAP_BRIDGE_FRAMES):
            anchor_last["age"] += 1
            anchor_geom = anchor_last["geometry"]
        else:
            anchor_last["geometry"] = None

        frame_overlay, frame_near_v, frame_far_v = resolve_frame_overlay(
            road_binary, height, width, overlay, frame_geometry, aim_state, lookahead_offset,
            lane_data=lane_data, lane_state=lane_state, vo_pts=vo_pts, vo_state=vo_state,
            chevron_phase_m=phase_m,
            direct_geometry=(direct_future_geometry(path_smoother, vo_pts)
                             if RIBBON_SOURCE == "perception" else None),
            anchor_geometry=anchor_geom)
        base = (orig.astype(np.float32) * (1.0 - BACKGROUND_DIM_ALPHA * ramp_value)).astype(np.uint8)
        reveal = reveal_rows_for(ramp_value, frame_near_v, frame_far_v, height)
        blend_path(base, frame_overlay, occlusion, gain=reveal)

        persons = extract_detections(result, {PERSON_CLASS_ID}, width, height)
        vehicles = extract_detections(result, vehicle_ids, width, height)
        selected_persons = select_and_smooth(persons, person_state, MAX_PEDESTRIANS_TO_RENDER)
        attach_distances(selected_persons, depth_norm, depth_fit)
        selected_vehicles = select_and_smooth(vehicles, vehicle_state, MAX_VEHICLES_TO_RENDER)
        attach_distances(selected_vehicles, depth_norm, depth_fit)

        if dump is not None:
            dump.frame(index, ramp_value, label_text, LAST_RIBBON_GEOMETRY,
                       selected_persons, selected_vehicles, occlusion,
                       phase_m=phase_m)

        def compose(with_vehicles):
            layer = base.copy()
            draw_highlights(layer, selected_persons, BOX_COLOUR_CLOSE, BOX_COLOUR, show_class=False)
            if with_vehicles:
                draw_highlights(layer, selected_vehicles, VEHICLE_BOX_COLOUR_CLOSE,
                                VEHICLE_BOX_COLOUR, show_class=True)
            if label_text:
                draw_text_panel(layer, label_text)
            if ramp_value >= 0.999:
                return layer
            return (base.astype(np.float32) * (1.0 - ramp_value)
                    + layer.astype(np.float32) * ramp_value).astype(np.uint8)

        writer.write(compose(False))
        writer_vehicles.write(compose(True))
        rendered_frames += 1

    writer.release()
    writer_vehicles.release()

    if dump is not None:
        dump.close()

    return {
        "input_video": INPUT_VIDEO,
        "output_video": OUTPUT_VIDEO,
        "output_video_vehicles": OUTPUT_VIDEO_VEHICLES,
        "fps": fps,
        "width": width,
        "height": height,
        "frame_count": frame_count,
        "rendered_frames": rendered_frames,
        "on_frames": on_frames,
        "model": MODEL_NAME,
        "tracker": TRACKER_NAME,
        "path_style": "temporal_gated_animated",
        "overlay_geometry": dump.geometry_path if dump is not None and dump.enabled else None,
    }


def render_video(effect_plan, vo_track=None):
    apply_calibration_overrides()

    fps, width, height, frame_count = get_video_metadata(INPUT_VIDEO)

    # Was a warning that rendered anyway; the geometry is now actually corrected.
    apply_resolution_scaling(width, height)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))
    writer_vehicles = cv2.VideoWriter(OUTPUT_VIDEO_VEHICLES, fourcc, fps, (width, height))

    if not writer.isOpened() or not writer_vehicles.isOpened():
        print("Could not open output video writer(s).")
        raise SystemExit(1)

    # Respect the Gemma 4 gate: if the behaviour is obvious (no explanation
    # warranted), render a clean pass-through with no overlay at all, and skip
    # the heavy per-frame models entirely.
    if not explanation_enabled(effect_plan):
        print("Gate decision: no explanation warranted -> clean pass-through (no overlay).")
        capture = cv2.VideoCapture(INPUT_VIDEO)
        rendered_frames = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            writer.write(frame)
            writer_vehicles.write(frame)
            rendered_frames += 1
        capture.release()
        writer.release()
        writer_vehicles.release()
        return {
            "input_video": INPUT_VIDEO,
            "output_video": OUTPUT_VIDEO,
            "output_video_vehicles": OUTPUT_VIDEO_VEHICLES,
            "fps": fps,
            "width": width,
            "height": height,
            "frame_count": frame_count,
            "rendered_frames": rendered_frames,
            "rendered_person_total": 0,
            "rendered_vehicle_total": 0,
            "max_people_in_frame": 0,
            "model": MODEL_NAME,
            "tracker": TRACKER_NAME,
            "depth_labels": False,
            "road_segmentation": False,
            "explanation_enabled": False,
            "path_style": "clean_no_explanation",
        }

    label_text = effect_plan.get(
        "label_text",
        "Yielding for pedestrians in the crosswalk",
    )

    # Camera and planned path are static for the clip, so project and rasterise
    # the ribbon overlay once.
    geometry = load_path_geometry()
    overlay = build_path_overlay(geometry, height, width)
    travelled_m = travelled_distance_per_frame(vo_track)
    lateral_tracker = LateralMotionTracker(width)
    path_smoother = FuturePathSmoother()
    planner_track = load_planner_track() if RIBBON_SOURCE == "planner" else None
    planner_ego_poses = (
        ego_poses_from_track(vo_track) if RIBBON_SOURCE == "planner" else None
    )

    if RIBBON_SOURCE == "planner" and planner_track is None:
        print("Planner ribbon requested but the context lacks yaw/t0; "
              "drawing the static planner path.")
    elif RIBBON_SOURCE == "planner":
        print("Planner ribbon anchoring: %s"
              % ("world (validated VO poses)" if planner_ego_poses
                 else "plan poses (no validated VO)"))

    planner_offset = (planner_offset_function(planner_track, vo_track, fps)
                      if RIBBON_SOURCE == "planner" else None)
    anchor_frames = parse_future_anchors(load_future_anchors_for_job(), frame_count)

    if RIBBON_SOURCE == "planner" and planner_offset is None and anchor_frames is not None:
        print("Planner ribbon: no plan-vs-driven offset available; "
              "anchors unused, falling back to the planner projection.")
        anchor_frames = None

    anchor_last = {"geometry": None, "age": 0}

    if overlay is None:
        print("Could not build road path geometry; rendering without a path ribbon.")

    scene = None
    road_seg_enabled = USE_ROAD_SEGMENTATION
    depth_enabled = USE_DEPTH_DISTANCE
    if road_seg_enabled or depth_enabled:
        try:
            import scene_models as scene
            if road_seg_enabled:
                scene.load_road_model()
                print("Road segmentation enabled: ribbon clipped to the drivable surface.")
            if depth_enabled:
                scene.load_depth_model()
                print("Depth labels enabled: distances from road-anchored monocular depth.")
        except Exception as error:
            print("Scene models unavailable (%s); object-mask occlusion + flat-ground labels." % error)
            road_seg_enabled = False
            depth_enabled = False

    # Ego-lane centering, same as the timeline path. Without it this path produced
    # a ribbon pinned to the image centre column on every frame (RIBBON_AIM_BIAS is
    # 0 and the far end aims at the vanishing point), i.e. a path that ignores which
    # lane the car is actually in.
    lane_enabled = USE_LANE_CENTERING and scene is not None
    if lane_enabled:
        try:
            if LANE_SOURCE == "ufldv2":
                scene.load_lane_instance_model()
                print("Ego-lane centering enabled (UFLDv2 lane instances).")
            else:
                scene.load_lane_model()
                print("Ego-lane centering enabled (YOLOP lane lines).")
        except Exception as error:
            print("Lane model unavailable (%s); ribbon stays straight ahead." % error)
            lane_enabled = False

    model = YOLO(MODEL_NAME)

    # ByteTrack associates detections across frames: stable IDs and fewer
    # single-frame dropouts than plain per-frame predict().
    results = model.track(
        source=INPUT_VIDEO,
        stream=True,
        imgsz=IMAGE_SIZE,
        conf=CONFIDENCE_THRESHOLD,
        classes=OCCLUDER_CLASS_IDS,
        tracker=TRACKER_NAME,
        persist=True,
        verbose=False,
    )

    person_state = {"smoothed": {}, "selected": []}
    vehicle_state = {"smoothed": {}, "selected": []}
    vehicle_ids = set(VEHICLE_CLASS_IDS)
    aim_state = {}
    lane_state = {}
    vo_state = {}
    vo_trajs = parse_vo_track(vo_track)

    dump = None

    if os.environ.get("OPTICARVIS_DUMP_GEOMETRY", "1") == "1":
        from overlay_geometry_dump import GeometryDump
        from pipeline_common import WORKFLOW_OUTPUTS

        dump = GeometryDump(
            os.path.join(WORKFLOW_OUTPUTS, "overlay_geometry"),
            segment_tag(),
            {
                "clip_video": os.path.abspath(INPUT_VIDEO),
                "fps": fps,
                "width": width,
                "height": height,
                "frame_count": frame_count,
                "camera": {
                    "horizon_v": float(HORIZON_V),
                    "vanish_u": float(VANISH_U),
                    "focal_px": float(CAM_FOCAL_PX),
                    "cam_height_m": float(CAM_HEIGHT_M),
                },
                "resolution_scale": float(_RESOLUTION_SCALE_APPLIED or 1.0),
                "chevron_speed_mps": float(CHEVRON_SPEED_MPS),
            },
        )
        print("Overlay geometry dump:", dump.geometry_path)

    rendered_frames = 0
    rendered_person_total = 0
    rendered_vehicle_total = 0
    max_people_in_frame = 0

    for result in results:
        persons = extract_detections(result, {PERSON_CLASS_ID}, width, height)
        vehicles = extract_detections(result, vehicle_ids, width, height)

        road_soft = None
        road_binary = None
        if road_seg_enabled:
            road_binary = scene.road_mask(result.orig_img)
            feather = ROAD_MASK_FEATHER_PX | 1
            road_soft = cv2.GaussianBlur(
                (road_binary * 255).astype(np.uint8), (feather, feather), 0
            ).astype(np.float32) / 255.0
            occlusion = 1.0 - road_soft
        else:
            occlusion = build_occlusion_mask(result, height, width)

        depth_norm = None
        depth_fit = None
        if depth_enabled:
            depth_norm = scene.depth_map(result.orig_img)
            fit_mask = road_soft
            if fit_mask is None:
                fit_mask = scene.road_mask(result.orig_img).astype(np.float32)
            depth_fit = fit_depth_to_metric(depth_norm, fit_mask)

        if lane_enabled:
            lane_data = (scene.lane_instances(result.orig_img) if LANE_SOURCE == "ufldv2"
                         else scene.lane_line_mask(result.orig_img))
        else:
            lane_data = None

        vo_pts = (
            vo_trajs[rendered_frames]
            if (vo_trajs is not None and rendered_frames < len(vo_trajs))
            else None
        )

        apply_lateral_feed_forward(
            lateral_tracker.update(result.orig_img), lane_state, aim_state, vo_state)

        frame_geometry = geometry

        if RIBBON_SOURCE == "planner":
            frame_geometry = planner_geometry_for_frame(planner_track, rendered_frames, fps, planner_ego_poses)

        phase_m = (
            travelled_m[rendered_frames]
            if travelled_m is not None and rendered_frames < len(travelled_m)
            else (rendered_frames / fps) * CHEVRON_SPEED_MPS if fps else 0.0
        )

        anchor_geom = anchor_geometry_for_frame(
            anchor_frames, rendered_frames, offset_fn=planner_offset,
            s_travelled=(travelled_m[rendered_frames]
                         if travelled_m is not None and rendered_frames < len(travelled_m)
                         else 0.0))

        if anchor_geom is not None:
            anchor_last["geometry"] = anchor_geom
            anchor_last["age"] = 0
        elif (anchor_last["geometry"] is not None
                and anchor_last["age"] < ANCHOR_GAP_BRIDGE_FRAMES):
            anchor_last["age"] += 1
            anchor_geom = anchor_last["geometry"]
        else:
            anchor_last["geometry"] = None

        frame_overlay, _, _ = resolve_frame_overlay(
            road_binary, height, width, overlay, frame_geometry, aim_state,
            lane_data=lane_data, lane_state=lane_state,
            vo_pts=vo_pts, vo_state=vo_state,
            chevron_phase_m=phase_m,
            direct_geometry=(direct_future_geometry(path_smoother, vo_pts)
                             if RIBBON_SOURCE == "perception" else None),
            anchor_geometry=anchor_geom)
        base = dim_background(result.orig_img)
        blend_path(base, frame_overlay, occlusion)

        selected_persons = select_and_smooth(persons, person_state, MAX_PEDESTRIANS_TO_RENDER)
        attach_distances(selected_persons, depth_norm, depth_fit)
        rendered_person_total += draw_highlights(
            base, selected_persons, BOX_COLOUR_CLOSE, BOX_COLOUR, show_class=False
        )
        if len(selected_persons) > max_people_in_frame:
            max_people_in_frame = len(selected_persons)

        # Version WITHOUT vehicle highlights (persons + path only).
        frame_persons_only = base.copy()
        draw_text_panel(frame_persons_only, label_text)
        writer.write(frame_persons_only)

        # Version WITH vehicle highlights, drawn on top of the shared base.
        selected_vehicles = select_and_smooth(vehicles, vehicle_state, MAX_VEHICLES_TO_RENDER)
        attach_distances(selected_vehicles, depth_norm, depth_fit)
        rendered_vehicle_total += draw_highlights(
            base, selected_vehicles, VEHICLE_BOX_COLOUR_CLOSE, VEHICLE_BOX_COLOUR, show_class=True
        )
        draw_text_panel(base, label_text)
        writer_vehicles.write(base)

        # This path has no per-frame animation ramp: the overlay is on for the
        # whole clip (the gate said explain), so the ramp dumps as 1.0.
        if dump is not None:
            dump.frame(rendered_frames, 1.0, label_text, LAST_RIBBON_GEOMETRY,
                       selected_persons, selected_vehicles, occlusion,
                       phase_m=phase_m)

        rendered_frames += 1

    writer.release()
    writer_vehicles.release()

    if dump is not None:
        dump.close()

    return {
        "input_video": INPUT_VIDEO,
        "output_video": OUTPUT_VIDEO,
        "output_video_vehicles": OUTPUT_VIDEO_VEHICLES,
        "fps": fps,
        "width": width,
        "height": height,
        "frame_count": frame_count,
        "rendered_frames": rendered_frames,
        "rendered_person_total": rendered_person_total,
        "rendered_vehicle_total": rendered_vehicle_total,
        "max_people_in_frame": max_people_in_frame,
        "model": MODEL_NAME,
        "tracker": TRACKER_NAME,
        "depth_labels": depth_enabled,
        "road_segmentation": road_seg_enabled,
        "explanation_enabled": True,
        "max_pedestrians_to_render": MAX_PEDESTRIANS_TO_RENDER,
        "max_vehicles_to_render": MAX_VEHICLES_TO_RENDER,
        "min_proximity_score": MIN_PROXIMITY_SCORE,
        "path_style": "trajectory_ground_projection_ribbon",
    }


def record_timeline_render(tag, summary):
    """Record a timeline render (and its effective config) into the workflow state.

    render_video_timeline is invoked directly by render_timeline_clip.py, which
    never went through update_workflow_state - so timeline renders, i.e. every
    shipped video, were unrecorded and the state file still advertised outputs
    from an older run. This writes the render under a tag-keyed entry and prunes
    output paths that no longer exist, so the state cannot keep claiming artefacts
    that are gone.
    """
    state = read_json(STATE_JSON) if os.path.isfile(STATE_JSON) else {}
    outputs = state.setdefault("outputs", {})

    stale = [key for key, value in outputs.items()
             if isinstance(value, str) and value.lower().endswith(".mp4")
             and not os.path.isfile(value)]
    for key in stale:
        outputs.pop(key)
    if stale:
        print("Pruned %d stale output path(s) from the workflow state." % len(stale))

    record_key = "%s_timeline_render_%s" % (RENDER_VARIANT, tag)
    outputs[record_key + "_video"] = summary.get("delivered_video")
    outputs[record_key + "_video_vehicles"] = summary.get("delivered_video_vehicles")

    state["current_stage"] = record_key + "_complete"
    state[record_key] = {
        "status": "complete",
        "clip_video": summary.get("clip_video"),
        "timeline_json": summary.get("timeline_json"),
        "ego_track_json": summary.get("ego_track_json"),
        "vo_track_json": summary.get("vo_track_json"),
        "delivered_video": summary.get("delivered_video"),
        "delivered_video_vehicles": summary.get("delivered_video_vehicles"),
        "frame_count": summary.get("frame_count"),
        "rendered_frames": summary.get("rendered_frames"),
        "on_frames": summary.get("on_frames"),
        "fps": summary.get("fps"),
        "model": summary.get("model"),
        "tracker": summary.get("tracker"),
        "path_style": summary.get("path_style"),
        "config": summary.get("config"),
    }

    ensure_dir(os.path.dirname(STATE_JSON))
    write_json(STATE_JSON, state)
    print("Recorded render in workflow state:", STATE_JSON)


def update_workflow_state(render_summary):
    workflow_state = read_json(STATE_JSON)

    render_key = RENDER_VARIANT + "_final_preview_render"

    workflow_state["current_stage"] = render_key + "_complete"
    workflow_state["outputs"][RENDER_VARIANT + "_final_preview_video"] = OUTPUT_VIDEO
    workflow_state["outputs"][RENDER_VARIANT + "_final_preview_video_vehicles"] = OUTPUT_VIDEO_VEHICLES

    workflow_state[render_key] = {
        "status": "complete",
        "output_video": OUTPUT_VIDEO,
        "output_video_vehicles": OUTPUT_VIDEO_VEHICLES,
        "rendered_frames": render_summary["rendered_frames"],
        "rendered_person_total": render_summary["rendered_person_total"],
        "rendered_vehicle_total": render_summary["rendered_vehicle_total"],
        "max_people_in_frame": render_summary["max_people_in_frame"],
        "renderer": "opencv_live_segmentation_" + RENDER_VARIANT + "_preview",
        "model": render_summary["model"],
        "tracker": render_summary["tracker"],
        "depth_labels": render_summary["depth_labels"],
        "road_segmentation": render_summary["road_segmentation"],
        "path_style": render_summary["path_style"],
    }

    write_json(STATE_JSON, workflow_state)


def print_summary(effect_plan, render_summary):
    print("")
    print("Roadline final preview renderer")
    print("===============================")
    print("display_target:", effect_plan.get("display_target"))
    print("display_intensity:", effect_plan.get("display_intensity"))
    print("label_text:", effect_plan.get("label_text"))
    print("model:", render_summary["model"])
    print("tracker:", render_summary["tracker"])
    print("path_style:", render_summary["path_style"])
    print("rendered_frames:", render_summary["rendered_frames"])
    print("rendered_person_total:", render_summary["rendered_person_total"])
    print("rendered_vehicle_total:", render_summary["rendered_vehicle_total"])
    print("max_people_in_frame:", render_summary["max_people_in_frame"])
    print("depth_labels:", render_summary["depth_labels"])
    print("explanation_enabled:", render_summary["explanation_enabled"])
    print("")
    print("Output video (persons):", OUTPUT_VIDEO)
    print("Output video (persons + vehicles):", OUTPUT_VIDEO_VEHICLES)
    print("Updated state:", STATE_JSON)


def main():
    # Fast calibration path: draw the projected ribbon onto one still image so
    # the camera scalars can be tuned by eye without running the video pipeline.
    #   python final_preview_renderer.py --calibrate <input.jpg> <output.png>
    if len(sys.argv) >= 2 and sys.argv[1] == "--save-calibration":
        save_calibration_template()
        return

    if len(sys.argv) >= 4 and sys.argv[1] == "--calibrate":
        preview_calibration(sys.argv[2], sys.argv[3])
        return

    ensure_dir(OUTPUT_DIR)

    effect_plan = read_json(EFFECT_PLAN_JSON)

    render_config = load_render_config()
    applied_render_config = apply_render_config_to_globals(render_config)

    render_summary = render_video(effect_plan, vo_track=load_vo_track_for_job())

    if isinstance(render_summary, dict):
        render_summary["render_config"] = render_config
        render_summary["applied_render_config"] = applied_render_config
    update_workflow_state(render_summary)

    print_summary(effect_plan, render_summary)


if __name__ == "__main__":
    main()
