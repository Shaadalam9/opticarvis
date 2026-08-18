r"""Future-anchored ribbon geometry: map the driven path back into each frame.

The clips are pre recorded, so at every frame the car's true future is not a
prediction -- it is VISIBLE in the later frames of the same clip. The renderer
previously re-derived that future through a world model (VO heading + flat
ground pinhole), and every modelling error -- heading noise at pull-away, the
flat-ground assumption, the single-valued y(x) ribbon -- pushed the drawn band
off the street exactly when the drive got interesting (measured on the Cape
Town 96-degree intersection turn: band aimed at the median while the car swept
left across the junction).

This module removes the model from the loop. The ground spot the ego vehicle
occupies at a future time t+k appears in the CURRENT frame t at a pixel we can
find by pure image registration:

1. For consecutive frames, estimate the ground-plane homography from features
   tracked inside a road band (LK optical flow + RANSAC). Moving traffic is
   rejected as outliers.
2. Compose hops to map ACROSS time. Direct homographies over
   KEYFRAME_STRIDE-frame gaps shorten an n-frame chain to ~n/stride hops,
   which is what keeps drift small through fast turns.
3. In every future frame t+k the car's near-future ground point is the SAME
   fixed pixel (the calibrated projection of the spot REF_AHEAD_M ahead of the
   camera). Map that pixel back through the composed chain into frame t.

The chained points land on the actual street pixels the car later drives over
-- heading error, calibration drift and ground slope cancel by construction,
because no world coordinates are involved in the anchor's placement. World
metres enter only as the anchor's arclength tag s_m (from the VO track's pose
arclength, which is heading-bias robust: position deltas carry no yaw bias),
used by the renderer for span limits and chevron phase.

Output: <tag>_future_anchors.json beside the VO track, one anchor polyline
[[u, v, s_m], ...] per frame, in calibration pixel space (1280x720).

Run for the current pipeline_common job:
    python future_anchor.py

Run with explicit paths:
    python future_anchor.py <clip.mp4> <ego_trajectory.json> <out_anchors.json>
"""

import os
import sys

import cv2
import numpy as np

from pipeline_common import (
    CLIP_VIDEO,
    ensure_dir,
    normalise_path,
    read_json,
    segment_tag,
    workflow_path,
    write_json,
)

CALIB_REF_WIDTH = 1280
CALIB_REF_HEIGHT = 720

# Fallback calibration; per clip the values embedded in the VO track win, so
# the anchors and the arclengths always come from the same camera model.
FOCAL_PX = 1000.0
CAM_HEIGHT_M = 1.30
HORIZON_V = 448.0
VANISH_U = 636.0

# Feature band for the ground homography, as a METRIC forward window. The band
# used to be hardcoded rows (470, 714) -- which silently encoded the dev rig's
# horizon of 448, since row -> distance is d = f*H/(v - horizon). On a rig whose
# horizon sits elsewhere those same rows sample a different slab of world: 125 px
# too low on one clip put the whole band on the ego vehicle's own bonnet, and
# RANSAC then fitted the bonnet -- a rigid plane that never moves -- instead of
# the road. Every hop came back as an identity matrix, the chain "succeeded",
# and the pipeline reported a moving car (23-43 km/h on its own speed OSD) as
# stationary. Deriving the rows from the clip's calibration keeps the band on
# the same stretch of road for every rig.
GROUND_BAND_NEAR_M = 5.0
GROUND_BAND_FAR_M = 59.0
GROUND_BAND_FALLBACK = (470, 714)

# Bonnet guard. A metric band alone is not enough: a tall bonnet can still reach
# into the 5 m row. The bonnet is rigid with the camera, so with the vehicle
# moving its rows show near-zero optical flow while the road above streams --
# that contrast is what identifies it. (Frame differencing does not: the bonnet
# carries moving reflections and reads as mostly non-static.)
HOOD_STATIC_PX = 0.5
HOOD_MOVING_PX = 4.0
HOOD_PROBE_ROWS = 24
MAX_FEATURES = 600
RANSAC_THRESH_PX = 2.5

# A hop is trusted only when RANSAC kept enough of its tracks; below the floor
# the chain is declared broken at that frame and anchors simply stop there --
# a shorter truthful band beats a fabricated one.
MIN_HOP_INLIERS = 20
MIN_HOP_INLIER_RATIO = float(os.environ.get("OPTICARVIS_ANCHOR_MIN_INLIER_RATIO", "0.5"))

KEYFRAME_STRIDE = max(1, int(os.environ.get("OPTICARVIS_ANCHOR_KEYFRAME_STRIDE", "8")))
REF_AHEAD_M = float(os.environ.get("OPTICARVIS_ANCHOR_REF_AHEAD_M", "4.5"))

# How much future each frame carries: enough arclength to fill the drawn band
# (PATH_END_M) with margin, and a hard frame cap for red-light waits where
# arclength stops growing.
LOOKAHEAD_ARC_M = 17.0
LOOKAHEAD_MAX_S = 10.0

_LK = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)


def gray_frame(frame):
    """Greyscale at the calibration reference resolution (see ego_trajectory)."""
    if (frame.shape[1], frame.shape[0]) != (CALIB_REF_WIDTH, CALIB_REF_HEIGHT):
        frame = cv2.resize(
            frame,
            (CALIB_REF_WIDTH, CALIB_REF_HEIGHT),
            interpolation=cv2.INTER_AREA,
        )

    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def ground_band_rows(calib, height=CALIB_REF_HEIGHT):
    """Image rows covering GROUND_BAND_NEAR_M..GROUND_BAND_FAR_M of road.

    Falls back to the historical constant when no calibration is available, so
    a dev-rig clip is bit-identical to before.
    """
    if not calib:
        return GROUND_BAND_FALLBACK

    horizon = float(calib["horizon_v"])
    reach = float(calib["focal_px"]) * float(calib["cam_height_m"])
    top = int(round(horizon + reach / GROUND_BAND_FAR_M))
    bottom = int(round(horizon + reach / GROUND_BAND_NEAR_M))
    top = max(top, int(round(horizon)) + 4)
    bottom = min(bottom, height - 6)

    if bottom - top < 40:
        return GROUND_BAND_FALLBACK

    return (top, bottom)


def detect_hood_row(grays, band, samples=12):
    """Topmost row of the ego vehicle's bonnet, or None when it is not in view.

    Measured once per clip on frame pairs that actually move: scanning up from
    the bottom, bonnet rows hold still (< HOOD_STATIC_PX) while road rows stream
    (> HOOD_MOVING_PX). Returns the row to clamp the band's lower edge above.
    """
    if len(grays) < 4:
        return None

    height = grays[0].shape[0]
    step = max(1, len(grays) // max(samples, 1))
    displacement = {}

    for index in range(0, len(grays) - 1, step):
        gray_a, gray_b = grays[index], grays[index + 1]
        points = cv2.goodFeaturesToTrack(gray_a, MAX_FEATURES, 0.01, 7)

        if points is None or len(points) < 40:
            continue

        moved, status, _error = cv2.calcOpticalFlowPyrLK(gray_a, gray_b, points, None, **_LK)
        good = status.ravel() == 1

        if good.sum() < 40:
            continue

        src = points[good].reshape(-1, 2)
        shift = np.hypot(*(moved[good].reshape(-1, 2) - src).T)

        for row, value in zip(src[:, 1], shift):
            displacement.setdefault(int(row) // HOOD_PROBE_ROWS, []).append(float(value))

    if not displacement:
        return None

    medians = {band_index: float(np.median(values))
               for band_index, values in displacement.items() if len(values) >= 8}

    if not medians:
        return None

    scene_motion = float(np.median(list(medians.values())))

    if scene_motion < HOOD_MOVING_PX:
        return None                      # clip is not moving; nothing to infer

    hood_top = None
    band_index = (height - 1) // HOOD_PROBE_ROWS

    while band_index >= 0:
        median = medians.get(band_index)

        if median is None or median >= HOOD_STATIC_PX:
            break

        hood_top = band_index * HOOD_PROBE_ROWS
        band_index -= 1

    return hood_top


def estimate_hop(gray_a, gray_b, band=GROUND_BAND_FALLBACK):
    """Ground-plane homography frame a -> frame b, or (None, 0.0) if untrusted."""
    mask = np.zeros_like(gray_a)
    mask[band[0]:band[1], :] = 255
    points = cv2.goodFeaturesToTrack(gray_a, MAX_FEATURES, 0.01, 7, mask=mask)

    if points is None or len(points) < MIN_HOP_INLIERS:
        return None, 0.0

    moved, status, _error = cv2.calcOpticalFlowPyrLK(gray_a, gray_b, points, None, **_LK)
    good = status.ravel() == 1

    if good.sum() < MIN_HOP_INLIERS:
        return None, 0.0

    src = points[good].reshape(-1, 2)
    dst = moved[good].reshape(-1, 2)

    matrix, inliers = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_THRESH_PX)

    if matrix is None or inliers is None:
        return None, 0.0

    ratio = float(inliers.sum()) / float(len(src))

    if inliers.sum() < MIN_HOP_INLIERS or ratio < MIN_HOP_INLIER_RATIO:
        return None, ratio

    return matrix / matrix[2, 2], ratio


def compose_chain(hops_1, hops_k, start, end, stride=KEYFRAME_STRIDE):
    """H mapping frame `start` pixels to frame `end` pixels, keyframes preferred.

    hops_1[t] maps t -> t+1; hops_k[m] maps m*stride -> (m+1)*stride. Returns
    None as soon as a needed hop is missing -- the caller stops anchoring there.
    """
    matrix = np.eye(3)
    t = start

    while t < end:
        if t % stride == 0 and t + stride <= end and hops_k.get(t // stride) is not None:
            matrix = hops_k[t // stride] @ matrix
            t += stride
            continue

        hop = hops_1[t] if t < len(hops_1) else None

        if hop is None:
            return None

        matrix = hop @ matrix
        t += 1

    return matrix / matrix[2, 2]


def reference_pixel(focal_px, cam_height_m, horizon_v, vanish_u, ref_ahead_m=REF_AHEAD_M):
    """Calibration-space pixel of the ground point ref_ahead_m dead ahead.

    May land below the visible frame (it does at this camera pitch) -- a
    homography maps plane points regardless of the frame border, so the chain
    does not care.
    """
    return (
        float(vanish_u),
        float(horizon_v) + float(focal_px) * float(cam_height_m) / float(ref_ahead_m),
    )


def back_project(matrix, pixel):
    """Apply the inverse of `matrix` (a -> b) to a pixel given in frame b."""
    vec = np.linalg.inv(matrix) @ np.array([pixel[0], pixel[1], 1.0])

    if abs(vec[2]) < 1e-9:
        return None

    return float(vec[0] / vec[2]), float(vec[1] / vec[2])


# The renderer refuses a band under 3 * ANCHOR_RESAMPLE_M of ground arclength
# (final_preview_renderer.build_arc_ribbon_geometry); mirror that here so this
# module's reported yield means "the renderer can draw this", not "a list exists".
MIN_DRAWABLE_SPAN_M = 1.2


def anchor_span_m(polyline):
    """Ground arclength the polyline covers, from its s_m tags."""
    if not polyline or len(polyline) < 3:
        return 0.0

    return float(polyline[-1][2]) - float(polyline[0][2])


def pose_arclength(poses):
    """Cumulative driven metres per frame; position deltas carry no yaw bias."""
    arc = [0.0]

    for (x0, y0, _p0), (x1, y1, _p1) in zip(poses, poses[1:]):
        arc.append(arc[-1] + float(np.hypot(x1 - x0, y1 - y0)))

    return arc


def resolve_ground_band(grays, calib):
    """The rows to fit the ground homography on, for THIS clip's rig."""
    band = ground_band_rows(calib)
    hood_top = detect_hood_row(grays, band)

    if hood_top is not None and hood_top > band[0] + 40:
        band = (band[0], min(band[1], hood_top - 4))
        print("Bonnet detected from row %d; ground band clamped to %s"
              % (hood_top, (band,)))
    elif hood_top is not None:
        print("WARNING: the ego vehicle's bonnet appears to fill the ground band "
              "(static from row %d); the homography may fit the bonnet instead of "
              "the road." % hood_top)

    print("Ground band rows: %d-%d (%.0f-%.0f m ahead)"
          % (band[0], band[1], GROUND_BAND_FAR_M, GROUND_BAND_NEAR_M))

    return band


def anchor_track(grays, poses, fps, calib, progress=None):
    """Per-frame anchor polylines [[u, v, s_m], ...] in calibration pixels."""
    count = min(len(grays), len(poses))
    band = resolve_ground_band(grays, calib)
    hops_1 = []

    for t in range(count - 1):
        matrix, _ratio = estimate_hop(grays[t], grays[t + 1], band)
        hops_1.append(matrix)

        if progress and t % 200 == 0:
            progress("hop %d/%d" % (t, count - 1))

    hops_k = {}

    for m in range(count // KEYFRAME_STRIDE):
        a = m * KEYFRAME_STRIDE
        b = min(a + KEYFRAME_STRIDE, count - 1)

        if b - a == KEYFRAME_STRIDE:
            matrix, _ratio = estimate_hop(grays[a], grays[b], band)
            hops_k[m] = matrix

    arc = pose_arclength(poses)
    p_ref = reference_pixel(
        calib["focal_px"], calib["cam_height_m"], calib["horizon_v"], calib["vanish_u"])
    frame_cap = max(1, int(round(LOOKAHEAD_MAX_S * fps)))
    horizon_floor = calib["horizon_v"] + 4.0

    anchors = []
    failed_spans = []

    for t in range(count):
        polyline = []
        matrix = np.eye(3)
        j = t
        broken = False

        while j < count - 1:
            if arc[j] - arc[t] > LOOKAHEAD_ARC_M or j - t >= frame_cap:
                break

            # Dense 1-frame hops close to the car (the band's near field needs
            # tightly spaced anchors), keyframe hops beyond -- long chains then
            # compose ~n/stride matrices instead of n, which is what keeps
            # drift small through the fast parts of a turn.
            use_keyframe = (
                j - t >= 2 * KEYFRAME_STRIDE
                and j % KEYFRAME_STRIDE == 0
                and j + KEYFRAME_STRIDE < count
                and hops_k.get(j // KEYFRAME_STRIDE) is not None
            )

            if use_keyframe:
                matrix = hops_k[j // KEYFRAME_STRIDE] @ matrix
                j += KEYFRAME_STRIDE
            elif hops_1[j] is not None:
                matrix = hops_1[j] @ matrix
                j += 1
            else:
                broken = True
                break

            point = back_project(matrix, p_ref)

            if point is None:
                broken = True
                break

            u, v = point

            if v <= horizon_floor:
                continue

            polyline.append([
                round(u, 2),
                round(v, 2),
                round((arc[j] - arc[t]) + REF_AHEAD_M, 3),
            ])

        if broken:
            failed_spans.append([t, j])

        anchors.append(polyline)

    return anchors, failed_spans


def load_calibration(vo_track):
    """Camera constants from the VO track itself, so both sidecars agree."""
    return {
        "focal_px": float(vo_track.get("focal_px", FOCAL_PX)),
        "cam_height_m": float(vo_track.get("cam_height_m", CAM_HEIGHT_M)),
        "horizon_v": float(vo_track.get("horizon_v", HORIZON_V)),
        "vanish_u": float(vo_track.get("vanish_u", VANISH_U)),
    }


def read_grays(clip_video):
    if not os.path.isfile(clip_video):
        print("Missing clip video:")
        print(clip_video)
        raise SystemExit(1)

    capture = cv2.VideoCapture(clip_video)

    if not capture.isOpened():
        print("Could not open clip video:")
        print(clip_video)
        raise SystemExit(1)

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    grays = []

    while True:
        ok, frame = capture.read()

        if not ok:
            break

        grays.append(gray_frame(frame))

    capture.release()

    return fps, grays


def default_vo_json():
    return workflow_path("ego_trajectory", segment_tag() + "_ego_trajectory.json")


def default_output_json():
    return workflow_path("ego_trajectory", segment_tag() + "_future_anchors.json")


def parse_args(argv):
    if len(argv) == 1:
        return {
            "clip": CLIP_VIDEO,
            "vo": default_vo_json(),
            "output": default_output_json(),
        }

    if len(argv) == 4:
        return {
            "clip": normalise_path(argv[1]),
            "vo": normalise_path(argv[2]),
            "output": normalise_path(argv[3]),
        }

    print(__doc__)
    raise SystemExit(2)


def main():
    args = parse_args(sys.argv)

    if not os.path.isfile(args["vo"]):
        print("No VO track at %s -- the anchors need its ego poses; run "
              "ego_trajectory.py first. The renderer will fall back to the "
              "direct flat-ground path." % args["vo"])
        raise SystemExit(1)

    vo_track = read_json(args["vo"])
    poses = vo_track.get("ego_pose_right_positive")

    if not poses or len(poses) < 2:
        print("VO track has no usable ego poses; not writing anchors.")
        raise SystemExit(1)

    calib = load_calibration(vo_track)
    fps, grays = read_grays(args["clip"])

    if not (fps and fps > 1.0):
        print("Refusing to continue: video reports fps=%r." % fps)
        raise SystemExit(1)

    anchors, failed_spans = anchor_track(
        grays, poses, fps, calib, progress=lambda msg: print("  " + msg))

    # Drawable, not merely present: a chain that locked onto a rigid plane
    # emits plenty of anchors, all frozen within a pixel of the reference point.
    # Counting those as successes is how a plane-lock masquerades as a
    # stationary vehicle, so the yield figure is the renderer's own threshold.
    with_anchors = sum(1 for a in anchors if anchor_span_m(a) >= MIN_DRAWABLE_SPAN_M)
    present = sum(1 for a in anchors if len(a) >= 3)

    payload = {
        "type": "future_anchor_track",
        "version": 1,
        "clip_video": os.path.abspath(args["clip"]),
        "fps": fps,
        "frame_count": len(anchors),
        "coord_space": "calib_%dx%d" % (CALIB_REF_WIDTH, CALIB_REF_HEIGHT),
        "calib": calib,
        "lateral_convention": "right_positive",
        "ref_ahead_m": REF_AHEAD_M,
        "keyframe_stride": KEYFRAME_STRIDE,
        "source_vo_track": os.path.abspath(args["vo"]),
        "quality": {
            "frames_with_anchors": with_anchors,
            "failed_spans": failed_spans,
        },
        "anchors": anchors,
    }

    ensure_dir(os.path.dirname(args["output"]))
    write_json(args["output"], payload)

    print("Future anchors: %d/%d frames drawable -> %s"
          % (with_anchors, len(anchors), args["output"]))

    if present > with_anchors * 4 and present > 20:
        print("")
        print("WARNING: %d frames carry anchors but only %d span enough ground "
              "to draw. That pattern means the homographies locked onto a RIGID "
              "plane (typically the ego vehicle's own bonnet) rather than the "
              "road: the chain composes identity matrices, so the car reads as "
              "stationary however fast it is moving. Check the ground band "
              "against this clip's calibration." % (present, with_anchors))

    if with_anchors < len(anchors) // 2:
        print("WARNING: fewer than half the frames have drawable anchors; the "
              "renderer falls back to the direct flat-ground path for the rest.")


if __name__ == "__main__":
    main()
