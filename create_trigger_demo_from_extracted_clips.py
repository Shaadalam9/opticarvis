"""
Create annotated and trigger-demo videos from the extracted candidate clips.

Use this after the clip-mining script has created:
    _output/top_clip_candidates.csv
    _output/ranked_clip_candidates.csv
    _output/extracted_clips/candidate_rank_XXX_....mp4

This script processes only the MP4 clips that still exist in EXTRACTED_CLIPS_DIR.
So if you deleted unwanted snippets, only the remaining snippets are processed.

What it creates
---------------
1. Annotated videos:
   - pedestrian and vehicle bounding boxes
   - track IDs
   - frame number and local time
   - trigger states

2. Trigger-demo videos:
   - a simple proof-of-concept explanation cue
   - monitor / interaction / risk / clutter states
   - main pedestrian highlighted
   - optional pedestrian attribute proxy scores

3. Optional NEW attribute output CSVs:
   - written only to OUTPUT_DIR/pedestrian_attribute_outputs
   - existing bbox/ranking CSV files are read-only inputs and are never modified

No command-line parser is used. Edit the CONFIG section and run the file.

Frame-rate note
---------------
This script aligns each extracted MP4 frame with the tracker CSV by frame count,
not by converting through FPS. Frame 0 of the extracted clip maps to
start_frame_local in the CSV, frame 1 maps to start_frame_local + 1, etc. This
avoids drift when the video FPS is 29.97 rather than exactly 30.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


# =============================================================================
# CONFIG
# =============================================================================

OUTPUT_DIR = "_output"
EXTRACTED_CLIPS_DIR = os.path.join(OUTPUT_DIR, "extracted_clips")
TOP_CANDIDATES_CSV = os.path.join(OUTPUT_DIR, "top_clip_candidates.csv")
RANKED_CANDIDATES_CSV = os.path.join(OUTPUT_DIR, "ranked_clip_candidates.csv")

ANNOTATED_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "annotated_clips")
TRIGGER_DEMO_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "trigger_demo_clips")
ATTRIBUTE_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "pedestrian_attribute_outputs")

CREATE_ANNOTATED_VIDEO = True
CREATE_TRIGGER_DEMO_VIDEO = True
OVERWRITE_OUTPUTS = True

# Only process clips that still exist in EXTRACTED_CLIPS_DIR.
PROCESS_ONLY_EXISTING_EXTRACTED_CLIPS = True

# COCO-style class IDs. Change if your YOLO model uses different ids.
PERSON_CLASS_IDS = {0}
VEHICLE_CLASS_IDS = {2, 3, 5, 7}

MIN_CONFIDENCE = 0.40

# Interaction region for normalised coordinates.
ROI_X_MIN = 0.25
ROI_X_MAX = 0.75
ROI_Y_MIN = 0.35
ROI_Y_MAX = 1.00

# Trigger thresholds for this first demo.
CLUTTER_OBJECT_COUNT_THRESHOLD = 5
RISK_REQUIRES_VEHICLE_PRESENT = True

# ----------------------- Pedestrian attribute stage -----------------------
# This is deliberately read-only with respect to your existing CSV files.
# The script reads bbox/ranking CSVs and optionally writes new attribute outputs
# into ATTRIBUTE_OUTPUT_DIR only. It never overwrites or appends to source CSVs.
ENABLE_PEDESTRIAN_ATTRIBUTES = True

# "proxy" uses only the existing tracker output and is runnable immediately.
# "external_csv" lets you plug in predictions from DAF/EfficientPIE later.
ATTRIBUTE_MODE = "proxy"  # options: "proxy", "external_csv"

# Optional external attribute predictions. Supported simple format:
# frame_local, track_id, attribute_name, score
# or wide columns such as crossing_score / walking_score / looking_score.
EXTERNAL_ATTRIBUTE_DIR = os.path.join(OUTPUT_DIR, "external_attribute_predictions")
FALLBACK_TO_PROXY_ATTRIBUTES = True

# Save newly computed attribute features. This creates NEW files only.
SAVE_ATTRIBUTE_OUTPUT_CSV = True

# Number of frames used to smooth tracker-derived attribute proxies.
ATTRIBUTE_SMOOTHING_WINDOW_FRAMES = 7

# Normalised-motion constants used by the proxy attribute mode.
# These are only for a first feasibility demo, not final scientific thresholds.
LATERAL_MOTION_NORM = 0.08
VERTICAL_MOTION_NORM = 0.08
AREA_GROWTH_NORM = 0.50

WALKING_SCORE_THRESHOLD = 0.35
APPROACHING_SCORE_THRESHOLD = 0.35
CROSSING_PROXY_SCORE_THRESHOLD = 0.55

# Video output settings.
OUTPUT_FOURCC = "mp4v"
BOX_THICKNESS = 2
FONT_SCALE = 0.55
FONT_THICKNESS = 2

# Draw all pedestrian and vehicle boxes in annotated videos.
DRAW_ALL_BOXES_IN_ANNOTATED = True

# In demo videos, keep it cleaner and highlight only the main pedestrian plus cue state.
DRAW_ALL_BOXES_IN_DEMO = False


# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# BASIC HELPERS
# =============================================================================

def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise column names and validate the expected YOLO tracking columns."""
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    required = {
        "yolo-id",
        "x-center",
        "y-center",
        "width",
        "height",
        "unique-id",
        "confidence",
        "frame-count",
    }
    missing = required.difference(set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=list(required))
    df["yolo-id"] = df["yolo-id"].astype(int)
    df["unique-id"] = df["unique-id"].astype(int)
    df["frame-count"] = df["frame-count"].astype(int)
    df["area"] = df["width"] * df["height"]

    return df.sort_values(["frame-count", "unique-id"]).reset_index(drop=True)


def add_box_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """Add x1, y1, x2, y2 coordinates from YOLO centre-size boxes."""
    df = df.copy()
    df["x1"] = df["x-center"] - df["width"] / 2.0
    df["y1"] = df["y-center"] - df["height"] / 2.0
    df["x2"] = df["x-center"] + df["width"] / 2.0
    df["y2"] = df["y-center"] + df["height"] / 2.0
    return df


def looks_normalised(df: pd.DataFrame) -> bool:
    """Return True when bbox coordinates look like normalised YOLO coordinates."""
    cols = ["x-center", "y-center", "width", "height"]
    if df.empty:
        return True
    max_value = float(df[cols].max().max())
    return max_value <= 2.0


def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def row_to_pixel_box(row: pd.Series, frame_width: int, frame_height: int, normalised: bool):
    """Convert one tracking row to pixel bbox coordinates."""
    if normalised:
        x1 = int(round(float(row["x1"]) * frame_width))
        y1 = int(round(float(row["y1"]) * frame_height))
        x2 = int(round(float(row["x2"]) * frame_width))
        y2 = int(round(float(row["y2"]) * frame_height))
    else:
        x1 = int(round(float(row["x1"])))
        y1 = int(round(float(row["y1"])))
        x2 = int(round(float(row["x2"])))
        y2 = int(round(float(row["y2"])))

    x1 = clamp(x1, 0, frame_width - 1)
    y1 = clamp(y1, 0, frame_height - 1)
    x2 = clamp(x2, 0, frame_width - 1)
    y2 = clamp(y2, 0, frame_height - 1)
    return x1, y1, x2, y2


def put_text_with_bg(
    frame: np.ndarray,
    text: str,
    origin: Tuple[int, int],
    text_colour: Tuple[int, int, int] = (255, 255, 255),
    bg_colour: Tuple[int, int, int] = (0, 0, 0),
    alpha: float = 0.65,
) -> None:
    """Draw readable text with a semi-transparent rectangle background."""
    x, y = origin
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, FONT_SCALE, FONT_THICKNESS)
    pad = 5

    x1 = clamp(x - pad, 0, frame.shape[1] - 1)
    y1 = clamp(y - th - baseline - pad, 0, frame.shape[0] - 1)
    x2 = clamp(x + tw + pad, 0, frame.shape[1] - 1)
    y2 = clamp(y + baseline + pad, 0, frame.shape[0] - 1)

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), bg_colour, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.putText(frame, text, (x, y), font, FONT_SCALE, text_colour, FONT_THICKNESS, cv2.LINE_AA)


def draw_panel(frame: np.ndarray, lines: list[str], x: int = 15, y: int = 30) -> None:
    """Draw a compact information panel in the top-left corner."""
    line_height = 25
    max_width = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    for line in lines:
        (tw, _), _ = cv2.getTextSize(line, font, FONT_SCALE, FONT_THICKNESS)
        max_width = max(max_width, tw)

    pad = 10
    panel_h = line_height * len(lines) + pad
    panel_w = max_width + 2 * pad

    overlay = frame.copy()
    cv2.rectangle(overlay, (x - pad, y - 22), (x - pad + panel_w, y - 22 + panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.58, frame, 0.42, 0, frame)

    for i, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (x, y + i * line_height),
            font,
            FONT_SCALE,
            (255, 255, 255),
            FONT_THICKNESS,
            cv2.LINE_AA,
        )


# =============================================================================
# CANDIDATE AND CLIP MATCHING
# =============================================================================

def load_candidates() -> pd.DataFrame:
    """Load top candidates if available, otherwise ranked candidates."""
    if os.path.isfile(TOP_CANDIDATES_CSV):
        path = TOP_CANDIDATES_CSV
    elif os.path.isfile(RANKED_CANDIDATES_CSV):
        path = RANKED_CANDIDATES_CSV
    else:
        raise FileNotFoundError(
            f"Could not find {TOP_CANDIDATES_CSV} or {RANKED_CANDIDATES_CSV}. "
            "Run the clip-mining script first."
        )

    df = pd.read_csv(path)
    if "rank" not in df.columns:
        raise ValueError(f"Candidate CSV has no 'rank' column: {path}")

    df["rank"] = pd.to_numeric(df["rank"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["rank"]).copy()
    df["rank"] = df["rank"].astype(int)
    logger.info("Loaded %d candidate rows from %s", len(df), path)
    return df


def parse_rank_from_clip_name(path: Path) -> Optional[int]:
    """Extract rank from names like candidate_rank_001_xxx_10.00s_to_39.97s.mp4."""
    match = re.search(r"candidate_rank_(\d+)_", path.name)
    if not match:
        logger.warning("Could not parse rank from clip filename: %s", path.name)
        return None
    return int(match.group(1))


def find_remaining_clips() -> list[Path]:
    root = Path(EXTRACTED_CLIPS_DIR)
    if not root.is_dir():
        raise FileNotFoundError(f"Extracted clips directory not found: {root}")

    clips = sorted(root.glob("*.mp4"))
    logger.info("Found %d remaining extracted clips in %s", len(clips), root)
    return clips


def build_clip_jobs(candidates: pd.DataFrame) -> list[Dict[str, object]]:
    """Match remaining extracted MP4 clips to candidate rows by rank."""
    clips = find_remaining_clips()
    by_rank = {int(row["rank"]): row for _, row in candidates.iterrows()}

    jobs = []
    for clip_path in clips:
        rank = parse_rank_from_clip_name(clip_path)
        if rank is None:
            continue

        if rank not in by_rank:
            logger.warning("No candidate row found for rank %s (%s)", rank, clip_path.name)
            continue

        row = by_rank[rank]
        jobs.append({"clip_path": clip_path, "candidate": row})

    logger.info("Prepared %d clip annotation jobs.", len(jobs))
    return jobs


# =============================================================================
# TRIGGERS AND DRAWING
# =============================================================================

def compute_frame_state(frame_rows: pd.DataFrame, main_track_id: int) -> Dict[str, object]:
    """Compute simple trigger state for one video frame."""
    person_rows = frame_rows[frame_rows["yolo-id"].isin(PERSON_CLASS_IDS)]
    vehicle_rows = frame_rows[frame_rows["yolo-id"].isin(VEHICLE_CLASS_IDS)]
    main_rows = person_rows[person_rows["unique-id"] == main_track_id]

    ped_count = int(person_rows["unique-id"].nunique())
    vehicle_count = int(vehicle_rows["unique-id"].nunique())
    traffic_density = ped_count + vehicle_count

    main_visible = not main_rows.empty
    in_roi = False
    main_row = None

    if main_visible:
        main_row = main_rows.iloc[0]
        x_center = float(main_row["x-center"])
        y_center = float(main_row["y-center"])
        in_roi = ROI_X_MIN <= x_center <= ROI_X_MAX and ROI_Y_MIN <= y_center <= ROI_Y_MAX

    monitor_trigger = main_visible
    interaction_trigger = main_visible and in_roi
    clutter_trigger = traffic_density >= CLUTTER_OBJECT_COUNT_THRESHOLD
    risk_proxy_trigger = interaction_trigger and (vehicle_count > 0 if RISK_REQUIRES_VEHICLE_PRESENT else True)

    if risk_proxy_trigger:
        cue_state = "risk"
        cue_text = "EXPLANATION ON: pedestrian-vehicle interaction"
    elif interaction_trigger:
        cue_state = "interaction"
        cue_text = "INTENT CUE: pedestrian in interaction region"
    elif monitor_trigger:
        cue_state = "monitor"
        cue_text = "MONITOR: pedestrian tracked"
    elif clutter_trigger:
        cue_state = "clutter"
        cue_text = "CLUTTER: simplify visual explanation"
    else:
        cue_state = "none"
        cue_text = "No cue"

    return {
        "ped_count": ped_count,
        "vehicle_count": vehicle_count,
        "traffic_density": traffic_density,
        "main_visible": main_visible,
        "in_roi": in_roi,
        "monitor_trigger": monitor_trigger,
        "interaction_trigger": interaction_trigger,
        "clutter_trigger": clutter_trigger,
        "risk_proxy_trigger": risk_proxy_trigger,
        "cue_state": cue_state,
        "cue_text": cue_text,
        "main_row": main_row,
    }


def clip01(series: pd.Series | np.ndarray | float) -> pd.Series | np.ndarray | float:
    """Clamp values to [0, 1]."""
    return np.clip(series, 0.0, 1.0)


def build_proxy_pedestrian_attributes(
    tracking: pd.DataFrame,
    by_frame: Dict[int, pd.DataFrame],
    start_frame_local: int,
    end_frame_local: int,
    main_track_id: int,
) -> pd.DataFrame:
    """
    Build simple, tracker-derived pedestrian attribute proxies.

    This does not run a learned pedestrian-attribute model. It is a first
    feasibility layer that uses the existing read-only bbox CSVs to create
    interpretable signals for the trigger demo.
    """
    records = []

    for frame_local in range(start_frame_local, end_frame_local + 1):
        frame_rows = by_frame.get(frame_local, pd.DataFrame(columns=tracking.columns))
        person_rows = frame_rows[frame_rows["yolo-id"].isin(PERSON_CLASS_IDS)]
        vehicle_rows = frame_rows[frame_rows["yolo-id"].isin(VEHICLE_CLASS_IDS)]
        main_rows = person_rows[person_rows["unique-id"] == main_track_id]

        if main_rows.empty:
            records.append(
                {
                    "frame_local": frame_local,
                    "track_id": main_track_id,
                    "main_visible": 0,
                    "x_center": np.nan,
                    "y_center": np.nan,
                    "area": np.nan,
                    "in_interaction_roi": 0,
                    "pedestrian_count": int(person_rows["unique-id"].nunique()),
                    "vehicle_count": int(vehicle_rows["unique-id"].nunique()),
                }
            )
            continue

        row = main_rows.iloc[0]
        x_center = float(row["x-center"])
        y_center = float(row["y-center"])
        area = float(row["area"])
        in_roi = int(ROI_X_MIN <= x_center <= ROI_X_MAX and ROI_Y_MIN <= y_center <= ROI_Y_MAX)

        records.append(
            {
                "frame_local": frame_local,
                "track_id": main_track_id,
                "main_visible": 1,
                "x_center": x_center,
                "y_center": y_center,
                "area": area,
                "in_interaction_roi": in_roi,
                "pedestrian_count": int(person_rows["unique-id"].nunique()),
                "vehicle_count": int(vehicle_rows["unique-id"].nunique()),
            }
        )

    attrs = pd.DataFrame(records)
    if attrs.empty:
        return attrs

    window = max(1, int(ATTRIBUTE_SMOOTHING_WINDOW_FRAMES))

    attrs["dx_window"] = (attrs["x_center"] - attrs["x_center"].shift(window)).abs()
    attrs["dy_window"] = (attrs["y_center"] - attrs["y_center"].shift(window)).abs()
    attrs["area_growth_window"] = (attrs["area"] / attrs["area"].shift(window)) - 1.0

    attrs["lateral_motion_score"] = clip01(attrs["dx_window"].fillna(0.0) / LATERAL_MOTION_NORM)
    attrs["vertical_motion_score"] = clip01(attrs["dy_window"].fillna(0.0) / VERTICAL_MOTION_NORM)
    attrs["walking_score"] = attrs[["lateral_motion_score", "vertical_motion_score"]].max(axis=1)
    attrs["approaching_score"] = clip01(attrs["area_growth_window"].fillna(0.0).clip(lower=0.0) / AREA_GROWTH_NORM)

    vehicle_present = (attrs["vehicle_count"] > 0).astype(float)
    main_visible = attrs["main_visible"].astype(float)
    in_roi = attrs["in_interaction_roi"].astype(float)

    attrs["crossing_proxy_score_raw"] = (
        0.25 * main_visible
        + 0.25 * in_roi
        + 0.30 * attrs["lateral_motion_score"]
        + 0.20 * vehicle_present
    )
    attrs["crossing_proxy_score"] = (
        attrs["crossing_proxy_score_raw"]
        .rolling(window=window, min_periods=1, center=False)
        .mean()
    )

    attrs["walking_like"] = (attrs["walking_score"] >= WALKING_SCORE_THRESHOLD).astype(int)
    attrs["approaching_like"] = (attrs["approaching_score"] >= APPROACHING_SCORE_THRESHOLD).astype(int)
    attrs["crossing_proxy"] = (attrs["crossing_proxy_score"] >= CROSSING_PROXY_SCORE_THRESHOLD).astype(int)

    attrs["attribute_source"] = "tracker_proxy"
    return attrs


def read_external_attribute_predictions(rank: int, video_id: str, main_track_id: int) -> Optional[pd.DataFrame]:
    """Read external attribute predictions if available, without modifying them."""
    root = Path(EXTERNAL_ATTRIBUTE_DIR)
    if not root.is_dir():
        return None

    candidates = [
        root / f"attributes_rank_{rank:03d}.csv",
        root / f"rank_{rank:03d}.csv",
        root / f"{video_id}.csv",
    ]
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        return None

    try:
        external = pd.read_csv(path)
    except Exception as exc:
        logger.warning("Could not read external attribute CSV %s: %s", path, exc)
        return None

    external = external.copy()
    external.columns = [c.strip().lower() for c in external.columns]

    if "frame_local" not in external.columns:
        logger.warning("External attribute CSV has no frame_local column: %s", path)
        return None

    external["frame_local"] = pd.to_numeric(external["frame_local"], errors="coerce")
    external = external.dropna(subset=["frame_local"]).copy()
    external["frame_local"] = external["frame_local"].astype(int)

    if "track_id" in external.columns:
        external["track_id"] = pd.to_numeric(external["track_id"], errors="coerce")
        external = external[(external["track_id"].isna()) | (external["track_id"] == main_track_id)].copy()

    # Long format: frame_local, track_id, attribute_name, score
    if {"attribute_name", "score"}.issubset(external.columns):
        external["score"] = pd.to_numeric(external["score"], errors="coerce").fillna(0.0)
        wide = external.pivot_table(
            index="frame_local",
            columns="attribute_name",
            values="score",
            aggfunc="max",
        ).reset_index()
        wide.columns = [str(c).strip().lower().replace(" ", "_") for c in wide.columns]
        external = wide

    # Normalise likely column names.
    rename_map = {
        "crossing": "crossing_proxy_score",
        "crossing_score": "crossing_proxy_score",
        "crossing_probability": "crossing_proxy_score",
        "walking": "walking_score",
        "walking_probability": "walking_score",
        "approaching": "approaching_score",
        "approaching_probability": "approaching_score",
        "looking": "looking_score",
        "looking_score": "looking_score",
    }
    external = external.rename(columns={k: v for k, v in rename_map.items() if k in external.columns})

    for col in ["crossing_proxy_score", "walking_score", "approaching_score", "looking_score"]:
        if col in external.columns:
            external[col] = pd.to_numeric(external[col], errors="coerce").fillna(0.0).clip(0.0, 1.0)

    if "crossing_proxy_score" not in external.columns:
        score_cols = [c for c in ["walking_score", "approaching_score", "looking_score"] if c in external.columns]
        if score_cols:
            external["crossing_proxy_score"] = external[score_cols].mean(axis=1)
        else:
            logger.warning("External attribute CSV has no usable score columns: %s", path)
            return None

    if "walking_score" not in external.columns:
        external["walking_score"] = 0.0
    if "approaching_score" not in external.columns:
        external["approaching_score"] = 0.0

    external["crossing_proxy"] = (external["crossing_proxy_score"] >= CROSSING_PROXY_SCORE_THRESHOLD).astype(int)
    external["walking_like"] = (external["walking_score"] >= WALKING_SCORE_THRESHOLD).astype(int)
    external["approaching_like"] = (external["approaching_score"] >= APPROACHING_SCORE_THRESHOLD).astype(int)
    external["attribute_source"] = "external_csv"
    logger.info("Loaded external attribute predictions: %s", path)
    return external


def build_pedestrian_attributes(
    rank: int,
    video_id: str,
    tracking: pd.DataFrame,
    by_frame: Dict[int, pd.DataFrame],
    start_frame_local: int,
    end_frame_local: int,
    main_track_id: int,
) -> pd.DataFrame:
    """Build attribute signals for a clip without modifying source CSV files."""
    if not ENABLE_PEDESTRIAN_ATTRIBUTES:
        return pd.DataFrame()

    if ATTRIBUTE_MODE == "external_csv":
        external = read_external_attribute_predictions(rank, video_id, main_track_id)
        if external is not None:
            return external
        if not FALLBACK_TO_PROXY_ATTRIBUTES:
            return pd.DataFrame()
        logger.warning("External attributes missing for rank %03d; falling back to tracker proxy.", rank)

    return build_proxy_pedestrian_attributes(
        tracking=tracking,
        by_frame=by_frame,
        start_frame_local=start_frame_local,
        end_frame_local=end_frame_local,
        main_track_id=main_track_id,
    )


def merge_attributes_into_state(state: Dict[str, object], attr_row: Optional[Dict[str, object]]) -> Dict[str, object]:
    """Add attribute scores to the current trigger state."""
    state = dict(state)
    state["attribute_trigger"] = False
    state["crossing_proxy_score"] = 0.0
    state["walking_score"] = 0.0
    state["approaching_score"] = 0.0
    state["attribute_source"] = "none"

    if not ENABLE_PEDESTRIAN_ATTRIBUTES or not attr_row:
        return state

    crossing_score = float(attr_row.get("crossing_proxy_score", 0.0) or 0.0)
    walking_score = float(attr_row.get("walking_score", 0.0) or 0.0)
    approaching_score = float(attr_row.get("approaching_score", 0.0) or 0.0)
    attr_source = str(attr_row.get("attribute_source", "unknown"))

    state["crossing_proxy_score"] = crossing_score
    state["walking_score"] = walking_score
    state["approaching_score"] = approaching_score
    state["attribute_source"] = attr_source

    crossing_trigger = crossing_score >= CROSSING_PROXY_SCORE_THRESHOLD
    approaching_trigger = approaching_score >= APPROACHING_SCORE_THRESHOLD

    state["attribute_trigger"] = bool(crossing_trigger or approaching_trigger)

    # Preserve the stronger risk trigger if it is already active. Otherwise allow
    # attribute scores to refine the explanation timing.
    if not state.get("risk_proxy_trigger", False):
        if crossing_trigger and state.get("main_visible", False):
            state["cue_state"] = "attribute_intent"
            state["cue_text"] = f"INTENT CUE: crossing-related motion ({crossing_score:.2f})"
        elif approaching_trigger and state.get("main_visible", False):
            state["cue_state"] = "approaching"
            state["cue_text"] = f"APPROACHING: pedestrian getting closer ({approaching_score:.2f})"

    return state


def draw_attribute_badge(frame: np.ndarray, state: Dict[str, object], normalised: bool) -> None:
    """Draw compact attribute scores above the main pedestrian."""
    main_row = state.get("main_row")
    if main_row is None or not ENABLE_PEDESTRIAN_ATTRIBUTES:
        return

    crossing = float(state.get("crossing_proxy_score", 0.0))
    walking = float(state.get("walking_score", 0.0))
    approaching = float(state.get("approaching_score", 0.0))

    if crossing <= 0 and walking <= 0 and approaching <= 0:
        return

    h, w = frame.shape[:2]
    x1, y1, _, _ = row_to_pixel_box(main_row, w, h, normalised)
    text = f"attr cross={crossing:.2f} walk={walking:.2f} app={approaching:.2f}"
    put_text_with_bg(frame, text, (x1, max(45, y1 - 32)), bg_colour=(70, 70, 180), alpha=0.62)


def draw_box(
    frame: np.ndarray,
    row: pd.Series,
    normalised: bool,
    label: str,
    colour: Tuple[int, int, int],
    thickness: int = BOX_THICKNESS,
) -> None:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = row_to_pixel_box(row, w, h, normalised)
    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, thickness)
    put_text_with_bg(frame, label, (x1, max(20, y1 - 6)), bg_colour=colour, alpha=0.55)


def draw_tracking_boxes(
    frame: np.ndarray,
    frame_rows: pd.DataFrame,
    normalised: bool,
    main_track_id: int,
    draw_all: bool,
) -> None:
    """Draw pedestrian and vehicle boxes."""
    if frame_rows.empty:
        return

    rows_to_draw = frame_rows
    if not draw_all:
        rows_to_draw = frame_rows[
            (frame_rows["yolo-id"].isin(PERSON_CLASS_IDS))
            & (frame_rows["unique-id"] == main_track_id)
        ]

    for _, row in rows_to_draw.iterrows():
        class_id = int(row["yolo-id"])
        track_id = int(row["unique-id"])
        conf = float(row["confidence"])

        if class_id in PERSON_CLASS_IDS:
            if track_id == main_track_id:
                colour = (0, 255, 255)
                label = f"MAIN ped {track_id} {conf:.2f}"
                thickness = BOX_THICKNESS + 1
            else:
                colour = (0, 220, 0)
                label = f"ped {track_id} {conf:.2f}"
                thickness = BOX_THICKNESS
            draw_box(frame, row, normalised, label, colour, thickness)

        elif class_id in VEHICLE_CLASS_IDS:
            colour = (255, 160, 0)
            label = f"veh {track_id} {conf:.2f}"
            draw_box(frame, row, normalised, label, colour, BOX_THICKNESS)


def draw_demo_cue(frame: np.ndarray, state: Dict[str, object]) -> None:
    """Draw a simple proof-of-concept explanation cue."""
    h, w = frame.shape[:2]
    cue_state = str(state["cue_state"])
    cue_text = str(state["cue_text"])

    if cue_state == "none":
        return

    if cue_state == "risk":
        bg_colour = (0, 0, 220)
        title = "RISK TRIGGER"
    elif cue_state == "attribute_intent":
        bg_colour = (0, 90, 230)
        title = "ATTRIBUTE INTENT CUE"
    elif cue_state == "approaching":
        bg_colour = (0, 130, 210)
        title = "APPROACHING CUE"
    elif cue_state == "interaction":
        bg_colour = (0, 150, 255)
        title = "INTENT TRIGGER"
    elif cue_state == "monitor":
        bg_colour = (0, 140, 0)
        title = "MONITORING"
    else:
        bg_colour = (90, 90, 90)
        title = "CONTEXT"

    box_w = min(620, w - 40)
    box_h = 92
    x1 = 20
    y1 = h - box_h - 25
    x2 = x1 + box_w
    y2 = y1 + box_h

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), bg_colour, -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    cv2.putText(frame, title, (x1 + 15, y1 + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, cue_text, (x1 + 15, y1 + 68), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)


def draw_roi(frame: np.ndarray) -> None:
    """Draw the interaction region used by the current trigger logic."""
    h, w = frame.shape[:2]
    x1 = int(ROI_X_MIN * w)
    x2 = int(ROI_X_MAX * w)
    y1 = int(ROI_Y_MIN * h)
    y2 = int(ROI_Y_MAX * h)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 255), 2)
    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
    put_text_with_bg(frame, "interaction ROI", (x1 + 8, max(25, y1 + 22)), bg_colour=(80, 80, 80), alpha=0.45)


# =============================================================================
# VIDEO PROCESSING
# =============================================================================

def make_video_writer(output_path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*OUTPUT_FOURCC)
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise IOError(f"Could not create video writer: {output_path}")
    return writer


def process_clip(job: Dict[str, object]) -> None:
    clip_path = Path(job["clip_path"])
    row = job["candidate"]

    rank = int(row["rank"])
    video_id = str(row["video_id"])
    csv_path = str(row["csv_path"])
    csv_fps = float(row["fps"])
    start_frame_local = int(row["start_frame_local"])
    end_frame_local = int(row["end_frame_local"])
    main_track_id = int(row["main_pedestrian_track_id"])
    abs_start_s = float(row["candidate_start_time_absolute_s"])
    abs_end_s = float(row["candidate_end_time_absolute_s"])

    if not os.path.isfile(csv_path):
        logger.warning("CSV path does not exist for rank %d: %s", rank, csv_path)
        return

    tracking = pd.read_csv(csv_path)
    tracking = ensure_columns(tracking)
    tracking = add_box_coordinates(tracking)
    tracking = tracking[tracking["confidence"] >= MIN_CONFIDENCE].copy()
    tracking = tracking[
        (tracking["frame-count"] >= start_frame_local)
        & (tracking["frame-count"] <= end_frame_local)
    ].copy()

    normalised = looks_normalised(tracking)
    by_frame = {int(k): g.copy() for k, g in tracking.groupby("frame-count")}

    attributes = build_pedestrian_attributes(
        rank=rank,
        video_id=video_id,
        tracking=tracking,
        by_frame=by_frame,
        start_frame_local=start_frame_local,
        end_frame_local=end_frame_local,
        main_track_id=main_track_id,
    )

    attr_by_frame: Dict[int, Dict[str, object]] = {}
    if not attributes.empty and "frame_local" in attributes.columns:
        for _, attr_row in attributes.iterrows():
            try:
                attr_by_frame[int(attr_row["frame_local"])] = attr_row.to_dict()
            except Exception:
                continue

    if SAVE_ATTRIBUTE_OUTPUT_CSV and ENABLE_PEDESTRIAN_ATTRIBUTES and not attributes.empty:
        attr_out_dir = Path(ATTRIBUTE_OUTPUT_DIR)
        attr_out_dir.mkdir(parents=True, exist_ok=True)
        attr_out_path = attr_out_dir / f"pedestrian_attributes_rank_{rank:03d}_{video_id}.csv"
        attributes.to_csv(attr_out_path, index=False)
        logger.info("Saved NEW attribute output CSV: %s", attr_out_path)

    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        logger.warning("Could not open clip: %s", clip_path)
        return

    video_fps = float(cap.get(cv2.CAP_PROP_FPS)) or csv_fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    annotated_path = Path(ANNOTATED_OUTPUT_DIR) / f"annotated_{clip_path.name}"
    demo_path = Path(TRIGGER_DEMO_OUTPUT_DIR) / f"trigger_demo_{clip_path.name}"

    if not OVERWRITE_OUTPUTS:
        if CREATE_ANNOTATED_VIDEO and annotated_path.exists():
            logger.info("Skipping existing annotated output: %s", annotated_path)
            create_annotated = False
        else:
            create_annotated = CREATE_ANNOTATED_VIDEO

        if CREATE_TRIGGER_DEMO_VIDEO and demo_path.exists():
            logger.info("Skipping existing demo output: %s", demo_path)
            create_demo = False
        else:
            create_demo = CREATE_TRIGGER_DEMO_VIDEO
    else:
        create_annotated = CREATE_ANNOTATED_VIDEO
        create_demo = CREATE_TRIGGER_DEMO_VIDEO

    annotated_writer = make_video_writer(annotated_path, video_fps, width, height) if create_annotated else None
    demo_writer = make_video_writer(demo_path, video_fps, width, height) if create_demo else None

    logger.info(
        "Processing rank %03d | %s | %s | frames=%d | video_fps=%.3f | csv_fps=%.3f | alignment=frame_count",
        rank,
        video_id,
        clip_path.name,
        total_frames,
        video_fps,
        csv_fps,
    )

    frame_idx = 0
    progress = tqdm(total=total_frames if total_frames > 0 else None, desc=f"Annotating rank {rank:03d}")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Map frame index in the extracted MP4 back to local frame number in the
        # tracking CSV by frame count, not by FPS conversion. This avoids drift
        # for videos with decimal FPS such as 29.97.
        csv_frame = start_frame_local + frame_idx
        frame_rows = by_frame.get(csv_frame, pd.DataFrame(columns=tracking.columns))
        state = compute_frame_state(frame_rows, main_track_id)
        state = merge_attributes_into_state(state, attr_by_frame.get(csv_frame))

        # Time is display-only. Alignment above is frame-based.
        local_time_s = frame_idx / max(video_fps, 1e-9)
        absolute_time_s = abs_start_s + local_time_s

        if annotated_writer is not None:
            annotated = frame.copy()
            draw_roi(annotated)
            draw_tracking_boxes(
                annotated,
                frame_rows,
                normalised,
                main_track_id,
                draw_all=DRAW_ALL_BOXES_IN_ANNOTATED,
            )
            draw_attribute_badge(annotated, state, normalised)
            panel_lines = [
                f"rank {rank:03d} | {video_id}",
                (
                    f"clip time {local_time_s:05.2f}s | "
                    f"abs {absolute_time_s:07.2f}s"
                ),
                f"csv frame {csv_frame} | main track {main_track_id}",
                (
                    f"ped={state['ped_count']} "
                    f"veh={state['vehicle_count']} "
                    f"density={state['traffic_density']}"
                ),
                (
                    f"monitor={int(state['monitor_trigger'])} "
                    f"interaction={int(state['interaction_trigger'])} "
                    f"risk={int(state['risk_proxy_trigger'])} "
                    f"clutter={int(state['clutter_trigger'])}"
                ),
                (
                    f"attr={int(state['attribute_trigger'])} "
                    f"cross={float(state['crossing_proxy_score']):.2f} "
                    f"walk={float(state['walking_score']):.2f} "
                    f"app={float(state['approaching_score']):.2f}"
                ),
            ]
            draw_panel(annotated, panel_lines)
            annotated_writer.write(annotated)

        if demo_writer is not None:
            demo = frame.copy()
            draw_tracking_boxes(
                demo,
                frame_rows,
                normalised,
                main_track_id,
                draw_all=DRAW_ALL_BOXES_IN_DEMO,
            )
            draw_attribute_badge(demo, state, normalised)
            draw_demo_cue(demo, state)
            panel_lines = [
                (
                    f"time {local_time_s:05.2f}s | "
                    f"abs {absolute_time_s:07.2f}s"
                ),
                f"state: {state['cue_state']}",
                (
                    f"attr cross={float(state['crossing_proxy_score']):.2f} "
                    f"walk={float(state['walking_score']):.2f} "
                    f"app={float(state['approaching_score']):.2f}"
                ),
            ]
            draw_panel(demo, panel_lines)
            demo_writer.write(demo)

        frame_idx += 1
        progress.update(1)

    progress.close()
    cap.release()

    if annotated_writer is not None:
        annotated_writer.release()
        logger.info("Saved annotated video: %s", annotated_path)
    if demo_writer is not None:
        demo_writer.release()
        logger.info("Saved trigger demo video: %s", demo_path)

    logger.info(
        "Finished rank %03d: abs %.2fs to %.2fs | processed %d frames",
        rank,
        abs_start_s,
        abs_end_s,
        frame_idx,
    )


def main() -> None:
    Path(ANNOTATED_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(TRIGGER_DEMO_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    if SAVE_ATTRIBUTE_OUTPUT_CSV and ENABLE_PEDESTRIAN_ATTRIBUTES:
        Path(ATTRIBUTE_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    candidates = load_candidates()
    jobs = build_clip_jobs(candidates)

    if not jobs:
        raise ValueError(
            "No clip jobs were found. Check that extracted MP4s still exist "
            "and that their rank appears in top_clip_candidates.csv."
        )

    for job in jobs:
        process_clip(job)

    print("Done.")
    if CREATE_ANNOTATED_VIDEO:
        print(f"Annotated clips: {Path(ANNOTATED_OUTPUT_DIR).resolve()}")
    if CREATE_TRIGGER_DEMO_VIDEO:
        print(f"Trigger demo clips: {Path(TRIGGER_DEMO_OUTPUT_DIR).resolve()}")
    if SAVE_ATTRIBUTE_OUTPUT_CSV and ENABLE_PEDESTRIAN_ATTRIBUTES:
        print(f"New pedestrian attribute CSVs: {Path(ATTRIBUTE_OUTPUT_DIR).resolve()}")


if __name__ == "__main__":
    main()
