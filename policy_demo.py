"""
Mark requested prototype for the OptiCarVis / CROWD extension.

Goal
----
Build one clean implementation layer for the current task:

1. Read the real project config and CROWD mapping CSV.
2. Read existing current YOLOv11 + BoT SORT tracking CSVs.
3. Automatically find stress test clips with several relevant aspects.
4. Compute frame level event signals for what, when, and how AV explanations
   should be shown.
5. Save a model signal mapping table with the open source models Mark suggested.
6. Save Plotly timelines and optional annotated demo videos.

This is not the final Bayesian optimisation code. It is the implementation layer
that creates the automated event signals that the optimiser can later tune.

No command line parser is used. Edit the CONFIG section and run this file.

Expected tracking CSV columns
-----------------------------
yolo-id, x-center, y-center, width, height, unique-id, confidence, frame-count

Expected tracking CSV filename
------------------------------
{video_id}_{segment_start_time_seconds}_{fps}.csv

Optional external signal CSV
----------------------------
If available, set EXTERNAL_SIGNAL_CSV. The same slot now supports two use cases.

A. Perception or prediction scores:
video_id, segment_start_time_s, frame_count or frame_local,
crossing_probability, crossing_uncertainty, depth_score, ttc_score,
ttc_risk_score, criticality_score, anomaly_score

B. Offline Alpamayo / adaptive prediction "when" signals:
video_id, segment_start_time_s, frame_count or frame_local,
when_start_local_s, when_end_local_s,
reasoning_trace, meta_action, confidence_score, uncertainty_score,
ambiguity_score, trajectory_conflict_score, explanation_needed,
explanation_reason, model_source

The preferred architecture is:
- YOLOv11 + BoT SORT remains the base detector/tracker input.
- Alpamayo or adaptive prediction is run offline before the study.
- Its output decides the explanation-needed moments.
- The current YOLOv11 + BoT SORT trigger remains a fallback if no offline
  model output is available.
"""

from __future__ import annotations

import datetime as dt
import glob
import json
import logging
import math
import os
import re

import common
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from tqdm import tqdm


# =============================================================================
# CONFIG
# =============================================================================

# Project paths are read from the existing project config through common.py.
# Do not read the config file directly here; keep this script consistent with main.py.

# Set these to restrict the search. Leave empty to search all mapped cities.
ONLY_LOCALITIES: set[str] = set()
ONLY_COUNTRIES: set[str] = set()
ONLY_ISO3: set[str] = set()
ONLY_VIDEO_IDS: set[str] = set()

# For quick debugging, set MAX_TRACKING_CSV_FILES to an integer such as 50.
MAX_TRACKING_CSV_FILES: Optional[int] = 300
TRACKING_CSV_GLOB = "*.csv"
RECURSIVE_TRACKING_SEARCH = False

# Keep this as a fast prototype by default. Mark asked to see one clip with the
# relevant automated signals, not to exhaustively score the whole CROWD corpus.
# Set MAX_TRACKING_CSV_FILES=None and MIN_TRACKING_CSV_BYTES=0 only when you
# intentionally want a full dataset scan.
MIN_TRACKING_CSV_BYTES = 5_000
PREFER_LOCAL_SOURCE_VIDEOS = True

# Reuse the ranking CSV from a previous run instead of re-scoring every tracking
# CSV. The ranking does not depend on the policy thresholds below, so this makes
# threshold-tuning reruns start at the per-clip policy phase immediately.
REUSE_EXISTING_RANKING = False

# Candidate windows for the first stress test clip.
CLIP_SECONDS = 30.0
STRIDE_SECONDS = 5.0
TOP_K_CLIPS = 5
MAX_WINDOWS_PER_CSV: Optional[int] = 8

# COCO class ids used by the current YOLOv11 + BoT SORT pipeline.
# Class 0 is a person detection, not necessarily a pedestrian. The role filter
# below separates walking pedestrian candidates from cyclists and motorcyclists
# using person + bicycle/motorcycle track association.
PERSON_CLASS_IDS = {0}
BICYCLE_CLASS_IDS = {1}
CAR_CLASS_IDS = {2}
MOTORCYCLE_CLASS_IDS = {3}
BUS_CLASS_IDS = {5}
TRUCK_CLASS_IDS = {7}
MOTOR_VEHICLE_CLASS_IDS = CAR_CLASS_IDS | MOTORCYCLE_CLASS_IDS | BUS_CLASS_IDS | TRUCK_CLASS_IDS
VEHICLE_CLASS_IDS = BICYCLE_CLASS_IDS | MOTOR_VEHICLE_CLASS_IDS
RIDER_VEHICLE_CLASS_IDS = BICYCLE_CLASS_IDS | MOTORCYCLE_CLASS_IDS
MIN_CONFIDENCE = 0.40
MIN_MAIN_PEDESTRIAN_SECONDS = 3.0

# Lightweight road-user role filter copied in spirit from the crossing validation repo.
# It removes person tracks that are actually riders from the pedestrian-trigger logic.
ENABLE_ROAD_USER_ROLE_FILTER = True
ROLE_MIN_SHARED_FRAMES = 4
ROLE_MIN_CONTINUOUS_SHARED_FRAMES = 12
ROLE_SHARED_RUN_GAP_ALLOW = 2
ROLE_MIN_VEHICLE_WIDTH_RATIO = 0.50
ROLE_MIN_VEHICLE_WIDTH_RATIO_FRAMES = 0.65
ROLE_DISTANCE_RELATIVE_THRESHOLD = 0.80
ROLE_PROXIMITY_RATIO_REQUIRED = 0.70
ROLE_ALPHA_X = 0.75
ROLE_BETA_Y = 0.08
ROLE_GAMMA_Y = 1.40
ROLE_COLOCATION_RATIO_REQUIRED = 0.70
ROLE_MOTION_SIMILARITY_THRESHOLD = 0.40
ROLE_MOTION_SIMILARITY_RATIO_REQUIRED = 0.50
ROLE_MIN_MOTION_STEPS = 3
ROLE_MOTION_COLOCATION_MIN = 0.50
ROLE_SHORT_SHARED_FRAMES = 8
ROLE_SHORT_SIMILARITY_RATIO_REQUIRED = 0.80
ROLE_SHORT_DISPLACEMENT_REQUIRED = 0.12
ROLE_EPS = 1e-9

# Event thresholds. These are deliberately policy parameters, not fixed science.
# Later, BO can tune these parameters.
#
# Important distinction:
# - The first three thresholds below describe diagnostic frame-level signals.
# - The DIRECT_WHEN_* thresholds decide whether an explanation should start.
#   Density and occlusion are context modifiers, not standalone "when" triggers.
CROSSING_TRIGGER_THRESHOLD = 0.58
INTERACTION_TRIGGER_THRESHOLD = 0.55
DENSITY_TRIGGER_THRESHOLD = 6
OCCLUSION_TRIGGER_THRESHOLD = 0.18
RISK_TRIGGER_THRESHOLD = 0.62
ANOMALY_TRIGGER_THRESHOLD = 0.60

DIRECT_WHEN_CROSSING_THRESHOLD = 0.68
DIRECT_WHEN_INTERACTION_THRESHOLD = 0.70
DIRECT_WHEN_RISK_THRESHOLD = 0.68
DIRECT_WHEN_OCCLUSION_CROSSING_THRESHOLD = 0.65
DIRECT_WHEN_OCCLUSION_INTERACTION_THRESHOLD = 0.70
PERSISTENCE_SECONDS = 0.60

# Normalisation constants for tracker based proxy signals.
LATERAL_MOTION_NORM = 0.12
APPROACHING_AREA_GROWTH_NORM = 0.60
PEDESTRIAN_VEHICLE_DISTANCE_NORM = 0.35
OCCLUSION_IOU_THRESHOLD = 0.05
OCCLUSION_FRAME_STEP = 8

# Optional external model signal CSV.
# This can contain either low-level perception scores or offline "when" outputs
# from Alpamayo / adaptive prediction. When available, offline explanation-needed
# rows are preferred over the proxy trigger.
EXTERNAL_SIGNAL_CSV = "C:/Users/localadmin/Desktop/Shadab/alpamayo_outputs/alpamayo_offline_when_TuCsyBF3nHU.csv"
# Offline model "when" configuration.
# Mark's latest direction is: compute WHEN offline, then update UI parameters
# during the study. Therefore this script prefers offline model signals whenever
# they are available, and only falls back to YOLOv11 + BoT SORT proxy signals.
PREFER_OFFLINE_MODEL_WHEN = True
USE_PROXY_WHEN_IF_OFFLINE_MISSING = True
OFFLINE_AMBIGUITY_THRESHOLD = 0.55
OFFLINE_UNCERTAINTY_THRESHOLD = 0.55
OFFLINE_TRAJECTORY_CONFLICT_THRESHOLD = 0.55
OFFLINE_LOW_CONFIDENCE_THRESHOLD = 0.45
OFFLINE_REASONING_MAX_CHARS = 160

# HOW-policy display parameters (BO-tunable like the thresholds above).
STYLE_COMPLEXITY_THRESHOLD = 0.55
CUE_REASON_MAX_CHARS = 72
CUE_ACTION_MAX_CHARS = 48

OFFLINE_MODEL_SIGNAL_TEMPLATE_FILENAME = "offline_model_signal_template.csv"

# Video rendering. The script still creates CSV and Plotly output if videos are absent.
CREATE_ANNOTATED_VIDEO = True
VIDEO_EXTENSIONS = [".mp4", ".mkv", ".mov", ".avi"]
DOWNLOAD_MISSING_SOURCE_VIDEOS = True
FTP_ALIASES = ["tue1", "tue2", "tue3", "tue4"]
FTP_CRAWL_PAGE_LIMIT = 500
FTP_TIMEOUT_SECONDS = 20
OUTPUT_FOURCC = "mp4v"
DRAW_ALL_TRACKS = True

# Exact "when" interval extraction. These outputs are generated inside OUTPUT_DIR
# after frame_level_what_when_how_policy.csv files are created.
FRAME_POLICY_FILENAME = "frame_level_what_when_how_policy.csv"
PER_CLIP_WHEN_INTERVALS_FILENAME = "when_intervals.csv"
WHEN_INTERVALS_SUMMARY_FILENAME = "when_intervals_summary.csv"
WHEN_INTERVALS_READABLE_FILENAME = "when_intervals_readable_summary.txt"

# Ignore very short explanation-active intervals and merge tiny inactive gaps.
# This removes flickery cues and end-of-clip activations that are too short to be useful.
MIN_WHEN_INTERVAL_SECONDS = 1.0
MAX_GAP_TO_MERGE_S = 0.20

FALLBACK_TRIGGER_COLUMNS = [
    "when_trigger",
    "offline_explanation_need_trigger",
    "risk_event_trigger",
    "crossing_interaction_event_trigger",
    "occluded_crossing_event_trigger",
    "anomaly_trigger",
]

REASON_COLUMNS = [
    "offline_explanation_need_trigger",
    "risk_event_trigger",
    "crossing_interaction_event_trigger",
    "occluded_crossing_event_trigger",
    "anomaly_trigger",
]

SCORE_COLUMNS = [
    "risk_score",
    "crossing_probability",
    "interaction_score",
    "occlusion_score",
    "visual_complexity_score",
    "ttc_risk_score",
    "criticality_score",
    "offline_ambiguity_score",
    "offline_uncertainty_score",
    "offline_confidence_score",
    "offline_trajectory_conflict_score",
]




# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIG AND PATHS FROM common.py
# =============================================================================

TRACKING_CSV_DIR = common.get_configs("data")
MAPPING_CSV = common.get_configs("mapping")
SOURCE_VIDEO_DIR = common.get_configs("videos")
VIDEO_BASE_URL = common.get_configs("VIDEO_BASE_URL")
VIDEO_USERNAME = common.get_secrets("ftp_username")
VIDEO_PASSWORD = common.get_secrets("ftp_password")
OUTPUT_DIR = os.path.join(common.get_configs("output_dir"), "mark_when_policy_demo")

if EXTERNAL_SIGNAL_CSV:
    EXTERNAL_SIGNAL_PATH = EXTERNAL_SIGNAL_CSV
else:
    EXTERNAL_SIGNAL_PATH = ""


# =============================================================================
# MODEL MAPPING REQUESTED BY MARK
# =============================================================================

def build_model_signal_mapping() -> pd.DataFrame:
    rows = [
        {
            "policy_part": "Objects, tracks and density",
            "automated_signal": "Pedestrian and vehicle detections, object counts, local traffic density and track continuity",
            "recommended_model_or_repo": "Current YOLOv11 + BoT SORT tracking outputs from the existing pipeline",
            "model_output_used": "Bounding boxes, classes, confidences, track IDs, frame counts and frame level density",
            "how_it_defines_when": "Trigger candidate when a tracked pedestrian becomes interaction relevant or density exceeds a threshold",
            "bo_tunable_parameters": "Detection confidence threshold; density threshold; cue onset; persistence",
            "current_code_status": "Implemented directly from existing YOLOv11 + BoT SORT tracking CSVs",
            "priority": "Core input, always required",
            "source_url": "Internal current pipeline output",
        },
        {
            "policy_part": "Segmentation and interaction region",
            "automated_signal": "Road, crossing, pavement, pedestrian, vehicle and obstacle masks",
            "recommended_model_or_repo": "Meta SAM3",
            "model_output_used": "Scene and object masks over time",
            "how_it_defines_when": "Trigger candidate when pedestrian overlaps a drivable, crossing or occlusion boundary region",
            "bo_tunable_parameters": "Mask overlap threshold; interaction margin; cue placement around mask",
            "current_code_status": "Not implemented yet; current fallback uses bbox derived interaction score",
            "priority": "Core next step",
            "source_url": "https://github.com/facebookresearch/sam3",
        },
        {
            "policy_part": "Depth and approach",
            "automated_signal": "Relative depth, distance gradient, approaching or retreating motion",
            "recommended_model_or_repo": "Depth Anything 3",
            "model_output_used": "Depth score, depth gradient, approximate closing distance",
            "how_it_defines_when": "Trigger candidate when pedestrian becomes close or the depth gradient changes quickly",
            "bo_tunable_parameters": "Depth threshold; TTC proxy threshold; early warning offset; persistence",
            "current_code_status": "Proxy implemented through bbox area growth; external depth columns supported",
            "priority": "Core next step",
            "source_url": "https://github.com/ByteDance-Seed/Depth-Anything-3",
        },
        {
            "policy_part": "Pedestrian attributes",
            "automated_signal": "Looking direction, posture, crossing related pedestrian attributes",
            "recommended_model_or_repo": "Detection Attribute Fields",
            "model_output_used": "Pedestrian attributes and confidence scores",
            "how_it_defines_when": "Trigger candidate when attributes imply attention, preparation to cross or ambiguous intent",
            "bo_tunable_parameters": "Attribute confidence threshold; attribute weights; uncertainty threshold",
            "current_code_status": "External signal slot supported; not run inside this script",
            "priority": "Core next step",
            "source_url": "https://github.com/vita-epfl/detection-attributes-fields",
        },
        {
            "policy_part": "Pedestrian intention",
            "automated_signal": "Crossing probability and uncertainty",
            "recommended_model_or_repo": "EfficientPIE",
            "model_output_used": "Crossing probability and uncertainty per pedestrian or frame",
            "how_it_defines_when": "Trigger candidate when crossing probability exceeds threshold or uncertainty is high near conflict",
            "bo_tunable_parameters": "Crossing probability threshold; uncertainty threshold; cue duration",
            "current_code_status": "External crossing_probability slot supported; YOLOv11 + BoT SORT fallback implemented",
            "priority": "Core",
            "source_url": "https://github.com/heinideyibadiaole/EfficientPIE",
        },
        {
            "policy_part": "Criticality and risk",
            "automated_signal": "TTC, time to react, conflict likelihood, criticality score",
            "recommended_model_or_repo": "CommonRoad-CriMe",
            "model_output_used": "Criticality metrics from trajectories or scenario representation",
            "how_it_defines_when": "Trigger candidate when criticality crosses a risk threshold",
            "bo_tunable_parameters": "TTC threshold; criticality threshold; warning onset; escalation level",
            "current_code_status": "External criticality_score and ttc_score slots supported; bbox proxy implemented",
            "priority": "Core but conditional",
            "source_url": "https://github.com/CommonRoad/commonroad-crime",
        },
        {
            "policy_part": "Trajectory and planning",
            "automated_signal": "Predicted ego and surrounding agent behaviour",
            "recommended_model_or_repo": "NVIDIA Alpamayo / Alpamayo 1.5",
            "model_output_used": "Planning or reasoning output describing possible future driving actions",
            "how_it_defines_when": "Trigger candidate when predicted trajectories conflict or planning uncertainty rises",
            "bo_tunable_parameters": "Planning horizon; risk threshold; explanation detail level",
            "current_code_status": "External anomaly_score slot supported; not run inside this script",
            "priority": "Core offline when source",
            "source_url": "https://github.com/NVlabs/alpamayo",
        },
        {
            "policy_part": "Context descriptor",
            "automated_signal": "City, country, time of day, traffic index, density, occlusion and movement features",
            "recommended_model_or_repo": "CROWD metadata plus fused perception signals",
            "model_output_used": "Compact context vector for each clip and frame window",
            "how_it_defines_when": "Context is fixed for a query; BO conditions on context and tunes display policy only",
            "bo_tunable_parameters": "Feature inclusion; scenario grouping; sampling strategy",
            "current_code_status": "Implemented from mapping.csv and YOLOv11 + BoT SORT tracking CSVs",
            "priority": "Core",
            "source_url": "Internal context layer",
        },
    ]
    return pd.DataFrame(rows)


def build_offline_model_signal_template() -> pd.DataFrame:
    columns = [
        "video_id",
        "segment_start_time_s",
        "frame_count",
        "frame_local",
        "when_start_local_s",
        "when_end_local_s",
        "reasoning_trace",
        "meta_action",
        "confidence_score",
        "uncertainty_score",
        "ambiguity_score",
        "trajectory_conflict_score",
        "explanation_needed",
        "explanation_reason",
        "model_source",
    ]
    return pd.DataFrame(columns=columns)


# =============================================================================
# MAPPING CSV HANDLING
# =============================================================================

def parse_flat_list(value: object) -> List[str]:
    text = str(value).strip()
    if text == "" or text.lower() == "nan":
        return []
    text = text.strip().strip("[]")
    if text == "":
        return []
    return [item.strip().strip("'\"") for item in text.split(",") if item.strip()]


def parse_nested_number_lists(value: object) -> List[List[float]]:
    text = str(value).strip()
    if text == "" or text.lower() == "nan":
        return []

    groups = re.findall(r"\[([^\[\]]*)\]", text)
    if not groups:
        groups = [text.strip().strip("[]")]

    parsed: List[List[float]] = []
    for group in groups:
        values: List[float] = []
        for token in group.split(","):
            cleaned = token.strip().strip("'\"")
            if cleaned == "" or cleaned.lower() == "nan":
                continue
            if re.fullmatch(r"-?\d+(\.\d+)?", cleaned):
                values.append(float(cleaned))
        parsed.append(values)

    return parsed


def first_or_none(values: List[float], index: int) -> Optional[float]:
    if index < len(values):
        return values[index]
    return None


def string_at(values: List[str], index: int, default: str = "") -> str:
    if index < len(values):
        return str(values[index])
    return default


def explode_mapping(mapping_csv: str) -> pd.DataFrame:
    if not os.path.isfile(mapping_csv):
        raise FileNotFoundError(f"Mapping CSV not found: {mapping_csv}")

    mapping = pd.read_csv(mapping_csv)
    required = {"locality", "country", "iso3", "continent", "videos", "start_time", "end_time", "time_of_day"}
    missing = sorted(required.difference(mapping.columns))
    if missing:
        raise ValueError(f"mapping.csv is missing columns: {missing}")

    rows: List[Dict[str, object]] = []
    for _, row in mapping.iterrows():
        videos = parse_flat_list(row.get("videos", ""))
        start_lists = parse_nested_number_lists(row.get("start_time", ""))
        end_lists = parse_nested_number_lists(row.get("end_time", ""))
        time_lists = parse_nested_number_lists(row.get("time_of_day", ""))
        vehicle_types = parse_flat_list(row.get("vehicle_type", ""))
        upload_dates = parse_flat_list(row.get("upload_date", ""))
        channels = parse_flat_list(row.get("channel", ""))

        for video_index, video_id in enumerate(videos):
            starts = start_lists[video_index] if video_index < len(start_lists) else []
            ends = end_lists[video_index] if video_index < len(end_lists) else []
            times = time_lists[video_index] if video_index < len(time_lists) else []

            for segment_index, start_value in enumerate(starts):
                end_value = first_or_none(ends, segment_index)
                time_value = first_or_none(times, segment_index)
                rows.append(
                    {
                        "video_id": str(video_id),
                        "segment_start_time_s": int(round(float(start_value))),
                        "segment_end_time_s": int(round(float(end_value))) if end_value is not None else np.nan,
                        "time_of_day": int(round(float(time_value))) if time_value is not None else np.nan,
                        "locality": row.get("locality", ""),
                        "country": row.get("country", ""),
                        "iso3": row.get("iso3", ""),
                        "continent": row.get("continent", ""),
                        "traffic_index": row.get("traffic_index", np.nan),
                        "vehicle_type": string_at(vehicle_types, video_index),
                        "upload_date": string_at(upload_dates, video_index),
                        "channel": string_at(channels, video_index),
                        "mapping_row_id": row.get("id", ""),
                    }
                )

    exploded = pd.DataFrame(rows)
    if exploded.empty:
        raise ValueError("No segment rows could be extracted from mapping.csv")

    logger.info("Loaded mapping.csv: %d original rows expanded to %d video segments", len(mapping), len(exploded))
    return exploded


def build_mapping_lookup(mapping_segments: pd.DataFrame) -> Dict[Tuple[str, int], Dict[str, object]]:
    lookup: Dict[Tuple[str, int], Dict[str, object]] = {}
    for _, row in mapping_segments.iterrows():
        key = (str(row["video_id"]), int(row["segment_start_time_s"]))
        lookup[key] = row.to_dict()
    return lookup


# =============================================================================
# TRACKING CSV DISCOVERY
# =============================================================================

def parse_tracking_csv_name(path: str) -> Optional[Dict[str, object]]:
    stem = os.path.splitext(os.path.basename(path))[0]
    parts = stem.rsplit("_", 2)
    if len(parts) != 3:
        return None

    video_id, start_text, fps_text = parts
    if not re.fullmatch(r"\d+", start_text):
        return None
    if not re.fullmatch(r"\d+(\.\d+)?", fps_text):
        return None

    return {
        "video_id": video_id,
        "segment_start_time_s": int(start_text),
        "fps": float(fps_text),
        "csv_path": str(path),
    }


def get_local_video_ids() -> set[str]:
    if not os.path.isdir(SOURCE_VIDEO_DIR):
        return set()

    ids: set[str] = set()
    for filename in os.listdir(SOURCE_VIDEO_DIR):
        stem, ext = os.path.splitext(filename)
        if ext.lower() in VIDEO_EXTENSIONS:
            ids.add(stem)
    return ids


def find_tracking_csvs(mapping_lookup: Dict[Tuple[str, int], Dict[str, object]]) -> List[str]:
    if not os.path.isdir(TRACKING_CSV_DIR):
        raise FileNotFoundError(f"Tracking CSV directory not found: {TRACKING_CSV_DIR}")

    if RECURSIVE_TRACKING_SEARCH:
        search_pattern = os.path.join(TRACKING_CSV_DIR, "**", TRACKING_CSV_GLOB)
        candidates = sorted(glob.glob(search_pattern, recursive=True))
    else:
        search_pattern = os.path.join(TRACKING_CSV_DIR, TRACKING_CSV_GLOB)
        candidates = sorted(glob.glob(search_pattern))

    local_video_ids = get_local_video_ids() if PREFER_LOCAL_SOURCE_VIDEOS else set()
    use_local_video_filter = bool(local_video_ids)

    filtered: List[str] = []
    skipped_bad_name = 0
    skipped_not_mapped = 0
    skipped_user_filter = 0
    skipped_no_local_video = 0
    skipped_too_small = 0

    for path in candidates:
        parsed = parse_tracking_csv_name(path)
        if parsed is None:
            skipped_bad_name += 1
            continue

        if MIN_TRACKING_CSV_BYTES > 0 and os.path.getsize(path) < MIN_TRACKING_CSV_BYTES:
            skipped_too_small += 1
            continue

        video_id = str(parsed["video_id"])
        segment_start = int(parsed["segment_start_time_s"])
        meta = mapping_lookup.get((video_id, segment_start))
        if meta is None:
            skipped_not_mapped += 1
            continue

        if use_local_video_filter and video_id not in local_video_ids:
            skipped_no_local_video += 1
            continue

        if ONLY_VIDEO_IDS and video_id not in ONLY_VIDEO_IDS:
            skipped_user_filter += 1
            continue
        if ONLY_LOCALITIES and str(meta.get("locality", "")) not in ONLY_LOCALITIES:
            skipped_user_filter += 1
            continue
        if ONLY_COUNTRIES and str(meta.get("country", "")) not in ONLY_COUNTRIES:
            skipped_user_filter += 1
            continue
        if ONLY_ISO3 and str(meta.get("iso3", "")) not in ONLY_ISO3:
            skipped_user_filter += 1
            continue

        filtered.append(path)

    filtered = sorted(filtered, key=lambda p: os.path.getsize(p), reverse=True)

    before_cap = len(filtered)
    if MAX_TRACKING_CSV_FILES is not None:
        filtered = filtered[:MAX_TRACKING_CSV_FILES]

    logger.info(
        "CSV discovery: found=%d matched=%d selected=%d bad_name=%d not_mapped=%d no_local_video=%d too_small=%d user_filter=%d local_video_filter=%s",
        len(candidates),
        before_cap,
        len(filtered),
        skipped_bad_name,
        skipped_not_mapped,
        skipped_no_local_video,
        skipped_too_small,
        skipped_user_filter,
        str(use_local_video_filter),
    )

    return filtered


# =============================================================================
# TRACKING HELPERS
# =============================================================================

def ensure_tracking_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

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
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Tracking CSV is missing columns: {missing}")

    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=list(required)).copy()
    df["yolo-id"] = df["yolo-id"].astype(int)
    df["unique-id"] = df["unique-id"].astype(int)
    df["frame-count"] = df["frame-count"].astype(int)
    df["area"] = df["width"] * df["height"]
    df["x1"] = df["x-center"] - df["width"] / 2.0
    df["y1"] = df["y-center"] - df["height"] / 2.0
    df["x2"] = df["x-center"] + df["width"] / 2.0
    df["y2"] = df["y-center"] + df["height"] / 2.0

    return df.sort_values(["frame-count", "unique-id"]).reset_index(drop=True)


TRACKING_USECOLS = ["yolo-id", "x-center", "y-center", "width", "height", "unique-id", "confidence", "frame-count"]


def load_filtered_tracking(csv_path: str) -> pd.DataFrame:
    """Read, clean, and confidence-filter one tracking CSV (shared by scoring, policy, and video phases)."""
    tracking = pd.read_csv(csv_path, usecols=TRACKING_USECOLS)
    tracking = ensure_tracking_columns(tracking)
    return tracking[tracking["confidence"] >= MIN_CONFIDENCE].copy()


def looks_normalised(df: pd.DataFrame) -> bool:
    if df.empty:
        return True
    columns = ["x-center", "y-center", "width", "height"]
    return float(df[columns].max().max()) <= 2.0


def clamp_float(value: float, low: float, high: float) -> float:
    return float(max(low, min(high, value)))


def safe_norm(value: float, high: float) -> float:
    if high <= 0:
        return 0.0
    return clamp_float(value / high, 0.0, 1.0)


def distance_score(row_a: pd.Series, rows_b: pd.DataFrame) -> float:
    if rows_b.empty:
        return 0.0

    dx = rows_b["x-center"].to_numpy(dtype=float) - float(row_a["x-center"])
    dy = rows_b["y-center"].to_numpy(dtype=float) - float(row_a["y-center"])
    distances = np.sqrt(dx * dx + dy * dy)
    min_distance = float(np.min(distances))
    return 1.0 - safe_norm(min_distance, PEDESTRIAN_VEHICLE_DISTANCE_NORM)


def estimate_occlusion_for_frame(person_rows: pd.DataFrame, vehicle_rows: pd.DataFrame) -> float:
    if person_rows.empty:
        return 0.0

    person_boxes = person_rows[["x1", "y1", "x2", "y2"]].to_numpy(dtype=float)
    person_ids = person_rows["unique-id"].to_numpy(dtype=int)
    if vehicle_rows.empty:
        all_boxes = person_boxes
        all_ids = person_ids
    else:
        all_boxes = np.vstack([person_boxes, vehicle_rows[["x1", "y1", "x2", "y2"]].to_numpy(dtype=float)])
        all_ids = np.concatenate([person_ids, vehicle_rows["unique-id"].to_numpy(dtype=int)])

    if len(all_boxes) <= 1:
        return 0.0

    x_left = np.maximum(person_boxes[:, None, 0], all_boxes[None, :, 0])
    y_top = np.maximum(person_boxes[:, None, 1], all_boxes[None, :, 1])
    x_right = np.minimum(person_boxes[:, None, 2], all_boxes[None, :, 2])
    y_bottom = np.minimum(person_boxes[:, None, 3], all_boxes[None, :, 3])
    inter_area = np.maximum(0.0, x_right - x_left) * np.maximum(0.0, y_bottom - y_top)

    person_area = np.maximum(0.0, (person_boxes[:, 2] - person_boxes[:, 0]) * (person_boxes[:, 3] - person_boxes[:, 1]))
    all_area = np.maximum(0.0, (all_boxes[:, 2] - all_boxes[:, 0]) * (all_boxes[:, 3] - all_boxes[:, 1]))
    union = person_area[:, None] + all_area[None, :] - inter_area
    iou = np.divide(inter_area, union, out=np.zeros_like(inter_area), where=union > 0)

    # A detection never occludes its own track, including duplicate rows.
    other = person_ids[:, None] != all_ids[None, :]
    other_counts = other.sum(axis=1)
    overlap_hits = ((iou > OCCLUSION_IOU_THRESHOLD) & other).sum(axis=1)
    scores = np.where(other_counts > 0, overlap_hits / np.maximum(other_counts, 1), 0.0)
    return float(np.mean(scores))


def deduplicate_tracking_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    if "confidence" not in df.columns:
        return df.drop_duplicates(subset=["yolo-id", "unique-id", "frame-count"], keep="first").copy()
    return (
        df.sort_values(["yolo-id", "unique-id", "frame-count", "confidence"], ascending=[True, True, True, False])
        .drop_duplicates(subset=["yolo-id", "unique-id", "frame-count"], keep="first")
        .copy()
    )


def longest_frame_run(frames: Iterable[object], gap_allow: int = ROLE_SHARED_RUN_GAP_ALLOW) -> int:
    values = sorted({int(frame) for frame in frames})
    if not values:
        return 0

    max_run = 1
    current_run = 1
    max_gap = max(int(gap_allow), 0) + 1
    previous_frame = values[0]
    for frame in values[1:]:
        if int(frame) - int(previous_frame) <= max_gap:
            current_run += 1
        else:
            max_run = max(max_run, current_run)
            current_run = 1
        previous_frame = frame
    return max(max_run, current_run)


def empty_role_result(person_id: int) -> Dict[str, object]:
    return {
        "unique-id": int(person_id),
        "road_user_role": "pedestrian_candidate",
        "is_vehicle_associated_person": False,
        "rider_type": "",
        "associated_vehicle_id": -1,
        "role_score": 0.0,
        "role_shared_frames": 0,
        "role_longest_shared_run": 0,
        "role_proximity_ratio": 0.0,
        "role_colocation_ratio": 0.0,
        "role_motion_similarity_ratio": 0.0,
    }


def classify_person_role(person_track: Optional[pd.DataFrame], rider_rows: pd.DataFrame, person_id: int) -> Dict[str, object]:
    """Classify one person track from pre-deduplicated rows (see build_person_role_lookup)."""
    result = empty_role_result(person_id)
    if person_track is None or len(person_track) < ROLE_MIN_SHARED_FRAMES:
        return result

    person_track = person_track.sort_values("frame-count").drop_duplicates(subset=["frame-count"], keep="first")
    first_frame = int(person_track["frame-count"].min())
    last_frame = int(person_track["frame-count"].max())

    vehicle_rows = rider_rows[
        (rider_rows["frame-count"] >= first_frame)
        & (rider_rows["frame-count"] <= last_frame)
    ]
    if vehicle_rows.empty:
        return result

    best_result = result.copy()
    for vehicle_id, vehicle_track in vehicle_rows.groupby("unique-id"):
        vehicle_track = vehicle_track.sort_values("frame-count").drop_duplicates(subset=["frame-count"], keep="first")
        if vehicle_track.empty:
            continue

        vehicle_class = int(vehicle_track.iloc[0]["yolo-id"])
        if vehicle_class in BICYCLE_CLASS_IDS:
            rider_type = "bicycle"
            road_user_role = "cyclist_candidate"
        elif vehicle_class in MOTORCYCLE_CLASS_IDS:
            rider_type = "motorcycle"
            road_user_role = "motorcyclist_candidate"
        else:
            continue

        joined = person_track.merge(vehicle_track, on="frame-count", suffixes=("", "_v"), how="inner")
        shared_frames = len(joined)
        if shared_frames < ROLE_MIN_SHARED_FRAMES:
            continue

        longest_shared = longest_frame_run(joined["frame-count"].tolist(), ROLE_SHARED_RUN_GAP_ALLOW)
        if longest_shared < ROLE_MIN_CONTINUOUS_SHARED_FRAMES:
            continue

        person_xy = joined[["x-center", "y-center"]].to_numpy(dtype=float)
        vehicle_xy = joined[["x-center_v", "y-center_v"]].to_numpy(dtype=float)
        person_width = joined["width"].to_numpy(dtype=float)
        person_height = joined["height"].to_numpy(dtype=float)
        vehicle_width = joined["width_v"].to_numpy(dtype=float)

        width_ratio_array = vehicle_width / np.maximum(person_width, ROLE_EPS)
        width_ratio = float(np.median(width_ratio_array))
        width_ratio_pass = float((width_ratio_array >= ROLE_MIN_VEHICLE_WIDTH_RATIO).mean())
        if width_ratio_pass < ROLE_MIN_VEHICLE_WIDTH_RATIO_FRAMES:
            continue

        distance = np.linalg.norm(person_xy - vehicle_xy, axis=1)
        distance_relative = distance / np.maximum(person_height, ROLE_EPS)
        proximity = distance_relative < ROLE_DISTANCE_RELATIVE_THRESHOLD
        proximity_ratio = float(proximity.mean())
        if proximity_ratio < ROLE_PROXIMITY_RATIO_REQUIRED:
            continue

        relative_x = vehicle_xy[:, 0] - person_xy[:, 0]
        relative_y = vehicle_xy[:, 1] - person_xy[:, 1]
        spatial = (
            (np.abs(relative_x) < ROLE_ALPHA_X * person_width)
            & (relative_y > ROLE_BETA_Y * person_height)
            & (relative_y < ROLE_GAMMA_Y * person_height)
        )
        colocated = proximity & spatial
        colocation_ratio = float(colocated.mean())

        person_motion = np.diff(person_xy, axis=0)
        vehicle_motion = np.diff(vehicle_xy, axis=0)
        similarity_ratio = 0.0
        if person_motion.shape[0] > 0:
            person_motion_norm = np.linalg.norm(person_motion, axis=1)
            vehicle_motion_norm = np.linalg.norm(vehicle_motion, axis=1)
            moving = (person_motion_norm > ROLE_EPS) & (vehicle_motion_norm > ROLE_EPS)
            cosine = np.zeros_like(person_motion_norm, dtype=float)
            cosine[moving] = (
                (person_motion[moving] * vehicle_motion[moving]).sum(axis=1)
                / (person_motion_norm[moving] * vehicle_motion_norm[moving])
            )
            proximity_steps = proximity[1:]
            length = min(len(proximity_steps), len(cosine), len(moving))
            denominator_mask = proximity_steps[:length] & moving[:length]
            denominator = int(denominator_mask.sum())
            if denominator >= ROLE_MIN_MOTION_STEPS:
                similarity_ratio = float(((cosine[:length] > ROLE_MOTION_SIMILARITY_THRESHOLD) & denominator_mask).sum() / denominator)

        if shared_frames < ROLE_SHORT_SHARED_FRAMES:
            if shared_frames > 1:
                displacement = float(np.linalg.norm(person_xy[-1] - person_xy[0]))
                displacement_relative = displacement / float(np.maximum(np.mean(person_height), ROLE_EPS))
            else:
                displacement_relative = 0.0
            if not (similarity_ratio >= ROLE_SHORT_SIMILARITY_RATIO_REQUIRED or displacement_relative >= ROLE_SHORT_DISPLACEMENT_REQUIRED):
                continue

        association_ok = (
            colocation_ratio >= ROLE_COLOCATION_RATIO_REQUIRED
            or (similarity_ratio >= ROLE_MOTION_SIMILARITY_RATIO_REQUIRED and colocation_ratio >= ROLE_MOTION_COLOCATION_MIN)
        )
        if not association_ok:
            continue

        score = 0.70 * colocation_ratio + 0.20 * proximity_ratio + 0.10 * similarity_ratio
        if score > float(best_result["role_score"]):
            best_result = {
                "unique-id": int(person_id),
                "road_user_role": road_user_role,
                "is_vehicle_associated_person": True,
                "rider_type": rider_type,
                "associated_vehicle_id": int(vehicle_id),
                "role_score": float(score),
                "role_shared_frames": int(shared_frames),
                "role_longest_shared_run": int(longest_shared),
                "role_proximity_ratio": float(proximity_ratio),
                "role_colocation_ratio": float(colocation_ratio),
                "role_motion_similarity_ratio": float(similarity_ratio),
            }

    return best_result


def build_person_role_lookup(tracking: pd.DataFrame) -> Dict[int, Dict[str, object]]:
    person_ids = sorted(
        tracking[tracking["yolo-id"].isin(PERSON_CLASS_IDS)]["unique-id"].dropna().astype(int).unique().tolist()
    )
    if not ENABLE_ROAD_USER_ROLE_FILTER or tracking.empty or not person_ids:
        return {person_id: empty_role_result(person_id) for person_id in person_ids}

    # Deduplicate and pre-split once for the whole window instead of once per person track.
    work = deduplicate_tracking_rows(tracking)
    person_work = work[work["yolo-id"].isin(PERSON_CLASS_IDS)]
    person_groups = {int(person_id): group for person_id, group in person_work.groupby("unique-id")}
    rider_rows = work[work["yolo-id"].isin(RIDER_VEHICLE_CLASS_IDS)]

    return {
        person_id: classify_person_role(person_groups.get(person_id), rider_rows, person_id)
        for person_id in person_ids
    }


def add_roles_to_person_rows(person_rows: pd.DataFrame, role_lookup: Dict[int, Dict[str, object]]) -> pd.DataFrame:
    if person_rows.empty:
        out = person_rows.copy()
        out["road_user_role"] = []
        return out

    role_rows = []
    for person_id in person_rows["unique-id"].dropna().astype(int).unique().tolist():
        role_rows.append(role_lookup.get(int(person_id), empty_role_result(int(person_id))))
    role_df = pd.DataFrame(role_rows)
    return person_rows.merge(role_df, on="unique-id", how="left")


def rows_with_role(person_rows: pd.DataFrame, role_name: str) -> pd.DataFrame:
    if person_rows.empty or "road_user_role" not in person_rows.columns:
        return person_rows.iloc[0:0].copy()
    return person_rows[person_rows["road_user_role"] == role_name].copy()


# =============================================================================
# MAIN PEDESTRIAN AND WINDOW SCORING
# =============================================================================

def compute_track_stats(person_rows: pd.DataFrame, fps: float) -> pd.DataFrame:
    if person_rows.empty:
        return pd.DataFrame()

    ordered = person_rows.sort_values(["unique-id", "frame-count"], kind="stable")
    grouped = ordered.groupby("unique-id")

    head_means = grouped.head(5).groupby("unique-id")[["x-center", "y-center", "area"]].mean()
    tail_means = grouped.tail(5).groupby("unique-id")[["x-center", "y-center", "area"]].mean()
    first_frame = grouped["frame-count"].min()
    last_frame = grouped["frame-count"].max()

    duration_s = (last_frame - first_frame + 1).astype(float) / fps
    x_start = head_means["x-center"]
    x_end = tail_means["x-center"]
    y_start = head_means["y-center"]
    y_end = tail_means["y-center"]
    area_start = head_means["area"]
    area_end = tail_means["area"]
    area_growth = (area_end - area_start) / np.maximum(area_start, 1e-9)

    median_x = grouped["x-center"].median()
    bottom_centrality = 1.0 - np.minimum((median_x - 0.5).abs() / 0.5, 1.0)
    bottom_proximity = grouped["y-center"].quantile(0.75).clip(0.0, 1.0)
    ego_corridor_score = 0.55 * bottom_centrality + 0.45 * bottom_proximity

    return pd.DataFrame(
        {
            "track_id": first_frame.index.to_numpy(dtype=int),
            "first_frame": first_frame.to_numpy(dtype=int),
            "last_frame": last_frame.to_numpy(dtype=int),
            "duration_s": duration_s.to_numpy(dtype=float),
            "mean_confidence": grouped["confidence"].mean().to_numpy(dtype=float),
            "x_motion": (x_end - x_start).abs().to_numpy(dtype=float),
            "y_motion": (y_end - y_start).abs().to_numpy(dtype=float),
            "area_growth": area_growth.to_numpy(dtype=float),
            "x_start": x_start.to_numpy(dtype=float),
            "x_end": x_end.to_numpy(dtype=float),
            "y_start": y_start.to_numpy(dtype=float),
            "y_end": y_end.to_numpy(dtype=float),
            "median_x": median_x.to_numpy(dtype=float),
            "median_y": grouped["y-center"].median().to_numpy(dtype=float),
            "ego_corridor_score": ego_corridor_score.to_numpy(dtype=float),
            "n_detections": grouped.size().to_numpy(dtype=int),
        }
    )


def choose_main_pedestrian(person_rows: pd.DataFrame, fps: float) -> Optional[Dict[str, object]]:
    stats = compute_track_stats(person_rows, fps)
    if stats.empty:
        return None

    candidates = stats[stats["duration_s"] >= MIN_MAIN_PEDESTRIAN_SECONDS].copy()
    if candidates.empty:
        return None

    candidates["main_score"] = (
        1.60 * candidates["duration_s"].clip(upper=10.0) / 10.0
        + 1.60 * candidates["x_motion"].clip(upper=0.45) / 0.45
        + 1.20 * candidates["ego_corridor_score"].clip(0.0, 1.0)
        + 0.90 * candidates["area_growth"].clip(lower=0.0, upper=1.5) / 1.5
        + 0.50 * candidates["mean_confidence"].clip(0.0, 1.0)
    )
    return candidates.sort_values("main_score", ascending=False).iloc[0].to_dict()


def score_candidate_window(window: pd.DataFrame, fps: float, start_frame: int, end_frame: int) -> Dict[str, object]:
    person_rows_all = window[window["yolo-id"].isin(PERSON_CLASS_IDS)].copy()
    vehicle_rows = window[window["yolo-id"].isin(VEHICLE_CLASS_IDS)].copy()
    role_lookup = build_person_role_lookup(window)
    person_rows_with_role = add_roles_to_person_rows(person_rows_all, role_lookup)
    pedestrian_rows = rows_with_role(person_rows_with_role, "pedestrian_candidate")
    cyclist_rows = rows_with_role(person_rows_with_role, "cyclist_candidate")
    motorcyclist_rows = rows_with_role(person_rows_with_role, "motorcyclist_candidate")

    main = choose_main_pedestrian(pedestrian_rows, fps)
    if main is None:
        return {
            "start_frame_local": start_frame,
            "end_frame_local": end_frame,
            "duration_s": round((end_frame - start_frame + 1) / fps, 2),
            "stress_test_score": 0.0,
            "reject_reason": "no sufficiently long pedestrian candidate after cyclist/motorcyclist filtering",
        }

    main_role = role_lookup.get(int(main["track_id"]), empty_role_result(int(main["track_id"])))
    frames = np.arange(start_frame, end_frame + 1)
    ped_density = pedestrian_rows.groupby("frame-count")["unique-id"].nunique().reindex(frames, fill_value=0)
    person_density = person_rows_all.groupby("frame-count")["unique-id"].nunique().reindex(frames, fill_value=0)
    cyclist_density = cyclist_rows.groupby("frame-count")["unique-id"].nunique().reindex(frames, fill_value=0)
    motorcyclist_density = motorcyclist_rows.groupby("frame-count")["unique-id"].nunique().reindex(frames, fill_value=0)
    veh_density = vehicle_rows.groupby("frame-count")["unique-id"].nunique().reindex(frames, fill_value=0)

    occlusion_frames = frames[::max(1, OCCLUSION_FRAME_STEP)]
    ped_by_frame = {int(frame): group for frame, group in pedestrian_rows.groupby("frame-count")}
    veh_by_frame = {int(frame): group for frame, group in vehicle_rows.groupby("frame-count")}
    empty_ped = pedestrian_rows.iloc[0:0]
    empty_veh = vehicle_rows.iloc[0:0]
    occlusion_scores: List[float] = [
        estimate_occlusion_for_frame(
            ped_by_frame.get(int(frame), empty_ped),
            veh_by_frame.get(int(frame), empty_veh),
        )
        for frame in occlusion_frames
    ]

    vehicle_presence = float((veh_density > 0).mean())
    avg_density = float((ped_density + veh_density).mean())
    occlusion_score = float(np.mean(occlusion_scores)) if occlusion_scores else 0.0

    crossing_proxy = safe_norm(float(main["x_motion"]), LATERAL_MOTION_NORM)
    approach_proxy = safe_norm(max(float(main["area_growth"]), 0.0), APPROACHING_AREA_GROWTH_NORM)
    interaction_score = 0.45 * float(main["ego_corridor_score"]) + 0.25 * crossing_proxy + 0.30 * vehicle_presence

    stress_score = (
        1.40 * safe_norm(float(main["duration_s"]), 10.0)
        + 1.50 * crossing_proxy
        + 1.40 * interaction_score
        + 1.10 * vehicle_presence
        + 0.90 * safe_norm(avg_density, 8.0)
        + 0.80 * occlusion_score
        + 0.70 * approach_proxy
    )

    return {
        "start_frame_local": int(start_frame),
        "end_frame_local": int(end_frame),
        "duration_s": round((end_frame - start_frame + 1) / fps, 2),
        "stress_test_score": round(float(stress_score), 4),
        "main_pedestrian_track_id": int(main["track_id"]),
        "main_road_user_role": str(main_role.get("road_user_role", "pedestrian_candidate")),
        "main_rider_type": str(main_role.get("rider_type", "")),
        "main_associated_vehicle_id": int(main_role.get("associated_vehicle_id", -1)),
        "main_role_score": round(float(main_role.get("role_score", 0.0)), 4),
        "main_track_duration_s": round(float(main["duration_s"]), 2),
        "main_x_motion": round(float(main["x_motion"]), 4),
        "main_area_growth_pct": round(float(main["area_growth"] * 100.0), 2),
        "ego_corridor_score": round(float(main["ego_corridor_score"]), 4),
        "crossing_proxy": round(float(crossing_proxy), 4),
        "approach_proxy": round(float(approach_proxy), 4),
        "interaction_score": round(float(interaction_score), 4),
        "avg_pedestrian_density": round(float(ped_density.mean()), 4),
        "avg_person_class_density": round(float(person_density.mean()), 4),
        "avg_cyclist_candidate_density": round(float(cyclist_density.mean()), 4),
        "avg_motorcyclist_candidate_density": round(float(motorcyclist_density.mean()), 4),
        "avg_vehicle_density": round(float(veh_density.mean()), 4),
        "avg_total_density": round(avg_density, 4),
        "vehicle_presence_fraction": round(vehicle_presence, 4),
        "occlusion_proxy": round(occlusion_score, 4),
        "reject_reason": "",
    }


def score_tracking_csv(csv_path: str, mapping_lookup: Dict[Tuple[str, int], Dict[str, object]]) -> List[Dict[str, object]]:
    parsed = parse_tracking_csv_name(csv_path)
    if parsed is None:
        return []

    video_id = str(parsed["video_id"])
    segment_start = int(parsed["segment_start_time_s"])
    fps = float(parsed["fps"])
    meta = mapping_lookup.get((video_id, segment_start), {})

    tracking = load_filtered_tracking(csv_path)
    if tracking.empty:
        return []

    max_frame = int(tracking["frame-count"].max())
    clip_frames = max(1, int(round(CLIP_SECONDS * fps)))
    stride_frames = max(1, int(round(STRIDE_SECONDS * fps)))

    if max_frame + 1 <= clip_frames:
        starts = [0]
    else:
        starts = list(range(0, max_frame - clip_frames + 2, stride_frames))

    if MAX_WINDOWS_PER_CSV is not None and len(starts) > MAX_WINDOWS_PER_CSV:
        indices = np.linspace(0, len(starts) - 1, MAX_WINDOWS_PER_CSV, dtype=int)
        starts = [starts[i] for i in sorted(set(indices.tolist()))]

    results: List[Dict[str, object]] = []
    for start_frame in starts:
        end_frame = min(start_frame + clip_frames - 1, max_frame)
        window = tracking[(tracking["frame-count"] >= start_frame) & (tracking["frame-count"] <= end_frame)].copy()
        if window.empty:
            continue

        score = score_candidate_window(window, fps, start_frame, end_frame)
        score.update(
            {
                "csv_file": os.path.basename(csv_path),
                "csv_path": str(csv_path),
                "video_id": video_id,
                "fps": fps,
                "segment_start_time_s": segment_start,
                "candidate_start_time_local_s": round(start_frame / fps, 2),
                "candidate_end_time_local_s": round(end_frame / fps, 2),
                "candidate_start_time_absolute_s": round(segment_start + start_frame / fps, 2),
                "candidate_end_time_absolute_s": round(segment_start + end_frame / fps, 2),
                "locality": meta.get("locality", ""),
                "country": meta.get("country", ""),
                "iso3": meta.get("iso3", ""),
                "continent": meta.get("continent", ""),
                "time_of_day": meta.get("time_of_day", np.nan),
                "traffic_index": meta.get("traffic_index", np.nan),
                "upload_date": meta.get("upload_date", ""),
                "channel": meta.get("channel", ""),
            }
        )
        results.append(score)

    return results


# =============================================================================
# EXTERNAL SIGNALS
# =============================================================================

def normalise_external_column_name(column: object) -> str:
    return str(column).strip().lower().replace(" ", "-").replace("_", "-")


def load_external_signals(path: str) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    if not os.path.isfile(path):
        logger.warning(
            "EXTERNAL_SIGNAL_CSV is set but the file does not exist: %s "
            "— all clips will fall back to YOLOv11 + BoT SORT proxy WHEN triggers",
            path,
        )
        return pd.DataFrame()

    signals = pd.read_csv(path)
    signals = signals.copy()
    signals.columns = [normalise_external_column_name(column) for column in signals.columns]
    signals = signals.rename(
        columns={
            "frame": "frame-count",
            "frame-local": "frame-local",
            "frame-count": "frame-count",
            "segment-start-time-s": "segment-start-time-s",
            "video-id": "video-id",
        }
    )
    logger.info(f"Loaded external offline/perception signal CSV: {path} | rows={len(signals)}")
    return signals


def external_has_column(external: pd.DataFrame, names: List[str]) -> bool:
    for name in names:
        if normalise_external_column_name(name) in external.columns:
            return True
    return False


def get_external_value(row: pd.Series, names: List[str], default_value: object) -> object:
    for name in names:
        column = normalise_external_column_name(name)
        if column in row.index and pd.notna(row[column]):
            return row[column]
    return default_value


def get_external_text(row: pd.Series, names: List[str], default_value: str = "") -> str:
    value = get_external_value(row, names, default_value)
    if pd.isna(value):
        return default_value
    return str(value)


def get_external_number(row: pd.Series, names: List[str], default_value: float = 0.0) -> float:
    value = get_external_value(row, names, default_value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default_value)
    if math.isnan(number):
        return float(default_value)
    return number


def get_external_bool(row: pd.Series, names: List[str], default_value: bool = False) -> bool:
    value = get_external_value(row, names, default_value)
    if isinstance(value, bool):
        return bool(value)

    try:
        number = float(value)
    except (TypeError, ValueError):
        number = float("nan")
    if not math.isnan(number):
        return bool(number > 0.0)

    text_value = str(value).strip().lower()
    if text_value in {"true", "yes", "y", "1", "needed", "explain", "explanation-needed"}:
        return True
    if text_value in {"false", "no", "n", "0", "not-needed", "none"}:
        return False
    return bool(default_value)


EXTERNAL_TIME_COLUMNS = [
    "frame-count",
    "frame-local",
    "when-start-frame-local",
    "when-end-frame-local",
    "when-start-local-s",
    "when-end-local-s",
    "when-start-absolute-s",
    "when-end-absolute-s",
]


def prepare_external_for_clip(
    external: pd.DataFrame,
    video_id: str,
    segment_start: int,
) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
    """Apply the per-clip filters and numeric conversions once, instead of once per frame."""
    if external.empty:
        return external, {}

    candidates = external
    if "video-id" in candidates.columns:
        candidates = candidates[candidates["video-id"].astype(str) == str(video_id)]

    if "segment-start-time-s" in candidates.columns:
        start_times = pd.to_numeric(candidates["segment-start-time-s"], errors="coerce")
        candidates = candidates[start_times == int(segment_start)]

    if not external.empty and candidates.empty and PREFER_OFFLINE_MODEL_WHEN:
        logger.info(
            "External signal CSV has no rows for video=%s segment=%s; falling back to proxy WHEN triggers for this clip",
            video_id,
            segment_start,
        )

    numeric = {
        column: pd.to_numeric(candidates[column], errors="coerce")
        for column in EXTERNAL_TIME_COLUMNS
        if column in candidates.columns
    }
    return candidates, numeric


def filter_external_rows_for_frame(
    candidates: pd.DataFrame,
    numeric: Dict[str, pd.Series],
    segment_start: int,
    frame_count: int,
    time_local_s: float,
) -> pd.DataFrame:
    if candidates.empty:
        return candidates

    if "frame-count" in numeric:
        matched = candidates[numeric["frame-count"] == int(frame_count)]
        if not matched.empty:
            return matched

    if "frame-local" in numeric:
        matched = candidates[numeric["frame-local"] == int(frame_count)]
        if not matched.empty:
            return matched

    has_frame_interval = "when-start-frame-local" in numeric and "when-end-frame-local" in numeric
    if has_frame_interval:
        matched = candidates[
            (numeric["when-start-frame-local"] <= int(frame_count))
            & (numeric["when-end-frame-local"] >= int(frame_count))
        ]
        if not matched.empty:
            return matched

    # Note: time_local_s is measured from the CANDIDATE CLIP start (frame - start_frame),
    # so when-start/end-local-s values are interpreted relative to the selected clip.
    has_time_interval = "when-start-local-s" in numeric and "when-end-local-s" in numeric
    if has_time_interval:
        matched = candidates[
            (numeric["when-start-local-s"] <= float(time_local_s))
            & (numeric["when-end-local-s"] >= float(time_local_s))
        ]
        if not matched.empty:
            return matched

    has_absolute_interval = "when-start-absolute-s" in numeric and "when-end-absolute-s" in numeric
    if has_absolute_interval:
        absolute_time = float(segment_start) + float(time_local_s)
        matched = candidates[
            (numeric["when-start-absolute-s"] <= absolute_time)
            & (numeric["when-end-absolute-s"] >= absolute_time)
        ]
        if not matched.empty:
            return matched

    has_any_time_column = (
        "frame-count" in numeric
        or "frame-local" in numeric
        or has_frame_interval
        or has_time_interval
        or has_absolute_interval
    )
    if not has_any_time_column and len(candidates) == 1:
        return candidates

    return candidates.iloc[0:0]


def external_row_for_frame(
    external_clip: Tuple[pd.DataFrame, Dict[str, pd.Series]],
    segment_start: int,
    frame_count: int,
    time_local_s: float,
) -> Dict[str, object]:
    clip_rows, numeric = external_clip
    if clip_rows.empty:
        return {}

    candidates = filter_external_rows_for_frame(clip_rows, numeric, segment_start, frame_count, time_local_s)
    if candidates.empty:
        return {}

    row = candidates.iloc[0]
    values: Dict[str, object] = {}

    for source_names, target in [
        (["crossing-probability"], "crossing_probability"),
        (["crossing-uncertainty"], "crossing_uncertainty"),
        (["depth-score"], "depth_score"),
        (["ttc-score"], "ttc_score"),
        (["ttc-risk-score"], "ttc_risk_score"),
        (["criticality-score"], "criticality_score"),
        (["anomaly-score"], "anomaly_score"),
    ]:
        if external_has_column(candidates, source_names):
            values[target] = get_external_number(row, source_names, 0.0)

    offline_columns = [
        "reasoning-trace",
        "meta-action",
        "confidence-score",
        "uncertainty-score",
        "ambiguity-score",
        "trajectory-conflict-score",
        "explanation-needed",
        "explanation-reason",
        "model-source",
    ]
    has_offline_data = external_has_column(candidates, offline_columns)
    if has_offline_data:
        ambiguity_score = get_external_number(row, ["ambiguity-score", "ambiguous-score"], 0.0)
        uncertainty_score = get_external_number(row, ["uncertainty-score", "prediction-uncertainty"], 0.0)
        confidence_score = get_external_number(row, ["confidence-score", "model-confidence"], 1.0)
        conflict_score = get_external_number(row, ["trajectory-conflict-score", "conflict-score"], 0.0)
        explanation_needed_flag = get_external_bool(row, ["explanation-needed", "explain", "show-explanation"], False)

        values["offline_model_available"] = True
        values["offline_ambiguity_score"] = ambiguity_score
        values["offline_uncertainty_score"] = uncertainty_score
        values["offline_confidence_score"] = confidence_score
        values["offline_trajectory_conflict_score"] = conflict_score
        values["offline_explanation_needed_flag"] = explanation_needed_flag
        values["offline_reasoning_trace"] = get_external_text(row, ["reasoning-trace", "chain-of-causation"], "")
        values["offline_meta_action"] = get_external_text(row, ["meta-action", "action"], "")
        values["offline_explanation_reason"] = get_external_text(row, ["explanation-reason", "reason"], "")
        values["offline_model_source"] = get_external_text(row, ["model-source", "source"], "offline_model")

    return values


def compact_text(value: object, max_chars: int = OFFLINE_REASONING_MAX_CHARS) -> str:
    text_value = str(value or "").replace("\n", " ").strip()
    if len(text_value) <= max_chars:
        return text_value
    return text_value[:max_chars - 3].rstrip() + "..."


# =============================================================================
# FRAME LEVEL POLICY
# =============================================================================

def compute_main_track_features(
    track_frames: np.ndarray,
    track_x: np.ndarray,
    track_y: np.ndarray,
    track_area: np.ndarray,
    current_frame: int,
    window_frames: int,
) -> Dict[str, float]:
    """Movement features for the main track up to current_frame.

    The track arrays must be sorted by frame (ensure_tracking_columns guarantees this).
    Head/tail rows of the recent window are found with searchsorted instead of
    filtering and copying the history DataFrame for every frame.
    """
    tail_idx = int(np.searchsorted(track_frames, current_frame, side="right")) - 1
    if tail_idx < 0:
        return {
            "lateral_motion_score": 0.0,
            "approach_score": 0.0,
            "ego_corridor_score": 0.0,
        }

    head_idx = int(np.searchsorted(track_frames, current_frame - window_frames, side="left"))
    if head_idx > tail_idx:
        head_idx = tail_idx

    dx = abs(float(track_x[tail_idx]) - float(track_x[head_idx]))
    area_growth = (float(track_area[tail_idx]) - float(track_area[head_idx])) / max(float(track_area[head_idx]), 1e-9)

    x_center = float(track_x[tail_idx])
    y_center = float(track_y[tail_idx])
    centrality = 1.0 - min(abs(x_center - 0.5) / 0.5, 1.0)
    bottom_proximity = clamp_float(y_center, 0.0, 1.0)
    ego_corridor_score = 0.55 * centrality + 0.45 * bottom_proximity

    return {
        "lateral_motion_score": safe_norm(dx, LATERAL_MOTION_NORM),
        "approach_score": safe_norm(max(area_growth, 0.0), APPROACHING_AREA_GROWTH_NORM),
        "ego_corridor_score": float(ego_corridor_score),
    }


def compute_policy_for_clip(
    candidate: pd.Series,
    external_signals: pd.DataFrame,
    tracking: Optional[pd.DataFrame] = None,
    role_lookup: Optional[Dict[int, Dict[str, object]]] = None,
) -> pd.DataFrame:
    csv_path = str(candidate["csv_path"])
    if tracking is None:
        tracking = load_filtered_tracking(csv_path)

    start_frame = int(candidate["start_frame_local"])
    end_frame = int(candidate["end_frame_local"])
    fps = float(candidate["fps"])
    video_id = str(candidate["video_id"])
    segment_start = int(candidate["segment_start_time_s"])
    main_track_id = int(candidate["main_pedestrian_track_id"])

    clip_rows = tracking[(tracking["frame-count"] >= start_frame) & (tracking["frame-count"] <= end_frame)].copy()
    if role_lookup is None:
        role_lookup = build_person_role_lookup(clip_rows)
    main_role = role_lookup.get(main_track_id, empty_role_result(main_track_id))
    main_road_user_role = str(main_role.get("road_user_role", "pedestrian_candidate"))
    main_is_pedestrian_candidate = bool(main_road_user_role == "pedestrian_candidate")

    # Roles are fixed per track for the whole clip: resolve them once per unique-id
    # instead of building and merging a role DataFrame for every frame.
    role_names = {
        int(person_id): str(role.get("road_user_role", "pedestrian_candidate"))
        for person_id, role in role_lookup.items()
    }
    by_frame = {int(frame): group for frame, group in clip_rows.groupby("frame-count")}
    empty_frame = clip_rows.iloc[0:0]
    main_track_all = clip_rows[
        (clip_rows["yolo-id"].isin(PERSON_CLASS_IDS))
        & (clip_rows["unique-id"] == main_track_id)
    ]
    main_frames = main_track_all["frame-count"].to_numpy()
    main_x = main_track_all["x-center"].to_numpy(dtype=float)
    main_y = main_track_all["y-center"].to_numpy(dtype=float)
    main_area = main_track_all["area"].to_numpy(dtype=float)
    movement_window_frames = max(1, int(round(0.75 * fps)))

    external_clip = prepare_external_for_clip(external_signals, video_id, segment_start)

    persistence_frames = max(1, int(round(PERSISTENCE_SECONDS * fps)))
    active_until_frame = -1
    last_active_what = "no_explanation"
    last_active_cue = "No explanation"
    records: List[Dict[str, object]] = []

    frame_iterator = tqdm(
        range(start_frame, end_frame + 1),
        desc=f"Policy frames {video_id}",
        unit="frame",
        leave=False,
    )
    for frame in frame_iterator:
        frame_rows = by_frame.get(frame, empty_frame)
        person_rows_all = frame_rows[frame_rows["yolo-id"].isin(PERSON_CLASS_IDS)]
        person_roles = person_rows_all["unique-id"].map(role_names).fillna("pedestrian_candidate")
        pedestrian_rows = person_rows_all[person_roles == "pedestrian_candidate"]
        vehicle_rows = frame_rows[frame_rows["yolo-id"].isin(VEHICLE_CLASS_IDS)]
        main_now = pedestrian_rows[pedestrian_rows["unique-id"] == main_track_id]

        person_class_count = int(person_rows_all["unique-id"].nunique())
        ped_count = int(pedestrian_rows["unique-id"].nunique())
        cyclist_count = int(person_rows_all.loc[person_roles == "cyclist_candidate", "unique-id"].nunique())
        motorcyclist_count = int(person_rows_all.loc[person_roles == "motorcyclist_candidate", "unique-id"].nunique())
        vehicle_count = int(vehicle_rows["unique-id"].nunique())
        density = ped_count + vehicle_count
        main_visible = int((not main_now.empty) and main_is_pedestrian_candidate)

        if main_visible:
            main_row = main_now.iloc[0]
            movement = compute_main_track_features(main_frames, main_x, main_y, main_area, frame, movement_window_frames)
            proximity_score = distance_score(main_row, vehicle_rows)
        else:
            movement = {"lateral_motion_score": 0.0, "approach_score": 0.0, "ego_corridor_score": 0.0}
            proximity_score = 0.0

        occlusion_score = estimate_occlusion_for_frame(person_rows_all, vehicle_rows)
        density_score = safe_norm(float(density), float(DENSITY_TRIGGER_THRESHOLD + 2))
        vehicle_present = 1.0 if vehicle_count > 0 else 0.0

        time_local_s = (frame - start_frame) / fps
        external = external_row_for_frame(external_clip, segment_start, frame, time_local_s)

        offline_model_available = bool(external.get("offline_model_available", False))
        offline_ambiguity_score = float(external.get("offline_ambiguity_score", 0.0))
        offline_uncertainty_score = float(external.get("offline_uncertainty_score", 0.0))
        offline_confidence_score = float(external.get("offline_confidence_score", 1.0))
        offline_trajectory_conflict_score = float(external.get("offline_trajectory_conflict_score", 0.0))
        offline_explanation_needed_flag = bool(external.get("offline_explanation_needed_flag", False))
        offline_reasoning_trace = str(external.get("offline_reasoning_trace", ""))
        offline_meta_action = str(external.get("offline_meta_action", ""))
        offline_explanation_reason = str(external.get("offline_explanation_reason", ""))
        offline_model_source = str(external.get("offline_model_source", ""))

        offline_explanation_need_trigger = bool(
            offline_model_available
            and (
                offline_explanation_needed_flag
                or offline_ambiguity_score >= OFFLINE_AMBIGUITY_THRESHOLD
                or offline_uncertainty_score >= OFFLINE_UNCERTAINTY_THRESHOLD
                or offline_trajectory_conflict_score >= OFFLINE_TRAJECTORY_CONFLICT_THRESHOLD
                or (0.0 < offline_confidence_score <= OFFLINE_LOW_CONFIDENCE_THRESHOLD)
            )
        )

        crossing_probability = external.get(
            "crossing_probability",
            clamp_float(
                0.45 * movement["lateral_motion_score"]
                + 0.25 * movement["ego_corridor_score"]
                + 0.20 * vehicle_present
                + 0.10 * proximity_score,
                0.0,
                1.0,
            ),
        )
        crossing_uncertainty = external.get("crossing_uncertainty", 0.0)
        depth_score = external.get("depth_score", movement["approach_score"])
        ttc_risk_score = external.get(
            "ttc_risk_score",
            external.get("ttc_score", 0.50 * proximity_score + 0.50 * movement["approach_score"]),
        )
        criticality_score = external.get(
            "criticality_score",
            clamp_float(0.45 * crossing_probability + 0.35 * ttc_risk_score + 0.20 * proximity_score, 0.0, 1.0),
        )
        anomaly_score = external.get("anomaly_score", 0.0)

        interaction_score = clamp_float(
            0.35 * movement["ego_corridor_score"]
            + 0.25 * proximity_score
            + 0.25 * crossing_probability
            + 0.15 * vehicle_present,
            0.0,
            1.0,
        )
        visual_complexity_score = clamp_float(0.60 * density_score + 0.40 * occlusion_score, 0.0, 1.0)
        risk_score = clamp_float(
            0.35 * criticality_score
            + 0.25 * ttc_risk_score
            + 0.20 * crossing_probability
            + 0.10 * visual_complexity_score
            + 0.10 * anomaly_score,
            0.0,
            1.0,
        )

        crossing_trigger = bool(crossing_probability >= CROSSING_TRIGGER_THRESHOLD and main_visible)
        interaction_trigger = bool(interaction_score >= INTERACTION_TRIGGER_THRESHOLD and main_visible)
        density_trigger = bool(density >= DENSITY_TRIGGER_THRESHOLD)
        occlusion_trigger = bool(occlusion_score >= OCCLUSION_TRIGGER_THRESHOLD)
        risk_trigger = bool(risk_score >= RISK_TRIGGER_THRESHOLD and main_visible)
        anomaly_trigger = bool(anomaly_score >= ANOMALY_TRIGGER_THRESHOLD)

        risk_event_trigger = bool(
            main_visible
            and risk_score >= DIRECT_WHEN_RISK_THRESHOLD
            and interaction_score >= DIRECT_WHEN_INTERACTION_THRESHOLD
        )
        crossing_interaction_event_trigger = bool(
            main_visible
            and crossing_probability >= DIRECT_WHEN_CROSSING_THRESHOLD
            and interaction_score >= DIRECT_WHEN_INTERACTION_THRESHOLD
        )
        occluded_crossing_event_trigger = bool(
            main_visible
            and occlusion_trigger
            and crossing_probability >= DIRECT_WHEN_OCCLUSION_CROSSING_THRESHOLD
            and interaction_score >= DIRECT_WHEN_OCCLUSION_INTERACTION_THRESHOLD
        )
        context_modifier_trigger = bool(density_trigger or occlusion_trigger)

        proxy_when_trigger = bool(
            risk_event_trigger
            or crossing_interaction_event_trigger
            or occluded_crossing_event_trigger
            or anomaly_trigger
        )

        use_offline_when = bool(PREFER_OFFLINE_MODEL_WHEN and offline_model_available)
        if use_offline_when:
            raw_trigger = bool(offline_explanation_need_trigger)
            when_source = "offline_model"
        elif USE_PROXY_WHEN_IF_OFFLINE_MISSING:
            raw_trigger = bool(proxy_when_trigger)
            when_source = "yolov11_botsort_proxy"
        else:
            raw_trigger = False
            when_source = "disabled_no_offline_model"

        if use_offline_when and offline_explanation_need_trigger:
            current_what_to_show = "model_reasoning_explanation_cue"
            if offline_explanation_reason:
                current_cue_text = compact_text(offline_explanation_reason, CUE_REASON_MAX_CHARS)
            elif offline_meta_action:
                current_cue_text = f"Explain AV action: {compact_text(offline_meta_action, CUE_ACTION_MAX_CHARS)}"
            else:
                current_cue_text = "Explain model reasoning / uncertainty"
        elif risk_event_trigger:
            current_what_to_show = "risk_marker_and_short_warning"
            current_cue_text = "Risk: pedestrian vehicle interaction"
        elif crossing_interaction_event_trigger:
            current_what_to_show = "pedestrian_intent_cue"
            current_cue_text = "Intent: possible crossing"
        elif occluded_crossing_event_trigger:
            current_what_to_show = "occlusion_aware_intent_cue"
            current_cue_text = "Intent cue: crossing under occlusion or clutter"
        elif anomaly_trigger:
            current_what_to_show = "anomaly_cue"
            current_cue_text = "Anomaly: model uncertainty"
        else:
            current_what_to_show = "no_explanation"
            current_cue_text = "No explanation"

        if raw_trigger:
            active_until_frame = max(active_until_frame, frame + persistence_frames)
            last_active_what = current_what_to_show
            last_active_cue = current_cue_text

        explanation_active = bool(frame <= active_until_frame)
        if explanation_active:
            what_to_show = last_active_what
            cue_text = last_active_cue
        else:
            what_to_show = "no_explanation"
            cue_text = "No explanation"

        saliency = clamp_float(0.25 + 0.55 * risk_score + 0.20 * visual_complexity_score, 0.0, 1.0)
        opacity = clamp_float(0.30 + 0.55 * saliency, 0.0, 1.0)
        cue_size = clamp_float(0.40 + 0.45 * saliency, 0.0, 1.0)
        if visual_complexity_score >= STYLE_COMPLEXITY_THRESHOLD or context_modifier_trigger:
            explanation_style = "short_label"
        else:
            explanation_style = "icon_plus_short_label"

        records.append(
            {
                "video_id": video_id,
                "segment_start_time_s": segment_start,
                "frame_local": frame,
                "time_local_s": round((frame - start_frame) / fps, 3),
                "time_absolute_s": round(segment_start + frame / fps, 3),
                "main_pedestrian_track_id": main_track_id,
                "main_road_user_role": main_road_user_role,
                "main_rider_type": str(main_role.get("rider_type", "")),
                "main_associated_vehicle_id": int(main_role.get("associated_vehicle_id", -1)),
                "main_role_score": round(float(main_role.get("role_score", 0.0)), 4),
                "main_visible": main_visible,
                "person_class_count": person_class_count,
                "pedestrian_count": ped_count,
                "cyclist_candidate_count": cyclist_count,
                "motorcyclist_candidate_count": motorcyclist_count,
                "vehicle_count": vehicle_count,
                "object_density": density,
                "lateral_motion_score": round(movement["lateral_motion_score"], 4),
                "approach_score": round(movement["approach_score"], 4),
                "ego_corridor_score": round(movement["ego_corridor_score"], 4),
                "pedestrian_vehicle_proximity_score": round(proximity_score, 4),
                "crossing_probability": round(float(crossing_probability), 4),
                "crossing_uncertainty": round(float(crossing_uncertainty), 4),
                "depth_score": round(float(depth_score), 4),
                "ttc_risk_score": round(float(ttc_risk_score), 4),
                "criticality_score": round(float(criticality_score), 4),
                "anomaly_score": round(float(anomaly_score), 4),
                "interaction_score": round(float(interaction_score), 4),
                "occlusion_score": round(float(occlusion_score), 4),
                "visual_complexity_score": round(float(visual_complexity_score), 4),
                "risk_score": round(float(risk_score), 4),
                "offline_model_available": int(offline_model_available),
                "offline_explanation_need_trigger": int(offline_explanation_need_trigger),
                "offline_explanation_needed_flag": int(offline_explanation_needed_flag),
                "offline_ambiguity_score": round(float(offline_ambiguity_score), 4),
                "offline_uncertainty_score": round(float(offline_uncertainty_score), 4),
                "offline_confidence_score": round(float(offline_confidence_score), 4),
                "offline_trajectory_conflict_score": round(float(offline_trajectory_conflict_score), 4),
                "offline_meta_action": offline_meta_action,
                "offline_explanation_reason": offline_explanation_reason,
                "offline_reasoning_trace": offline_reasoning_trace,
                "offline_model_source": offline_model_source,
                "when_source": when_source,
                "proxy_when_trigger": int(proxy_when_trigger),
                "crossing_trigger": int(crossing_trigger),
                "interaction_trigger": int(interaction_trigger),
                "density_trigger": int(density_trigger),
                "occlusion_trigger": int(occlusion_trigger),
                "risk_trigger": int(risk_trigger),
                "anomaly_trigger": int(anomaly_trigger),
                "risk_event_trigger": int(risk_event_trigger),
                "crossing_interaction_event_trigger": int(crossing_interaction_event_trigger),
                "occluded_crossing_event_trigger": int(occluded_crossing_event_trigger),
                "context_modifier_trigger": int(context_modifier_trigger),
                "when_trigger": int(raw_trigger),
                "explanation_active": int(explanation_active),
                "what_to_show": what_to_show,
                "cue_text": cue_text,
                "how_saliency": round(saliency, 4),
                "how_opacity": round(opacity, 4),
                "how_size": round(cue_size, 4),
                "explanation_style": explanation_style,
            }
        )

    return pd.DataFrame(records)


# =============================================================================
# PLOTLY TIMELINE
# =============================================================================

def save_policy_timeline(policy: pd.DataFrame, output_html: str, title: str) -> None:
    output_parent = os.path.dirname(output_html)
    if output_parent:
        os.makedirs(output_parent, exist_ok=True)

    fig = go.Figure()
    traces = [
        ("crossing_probability", "crossing probability"),
        ("interaction_score", "interaction score"),
        ("risk_score", "risk score"),
        ("visual_complexity_score", "visual complexity"),
        ("occlusion_score", "occlusion"),
        ("explanation_active", "explanation active"),
    ]

    for col, name in traces:
        fig.add_trace(
            go.Scatter(
                x=policy["time_local_s"],
                y=policy[col],
                mode="lines",
                name=name,
                customdata=policy[["pedestrian_count", "cyclist_candidate_count", "motorcyclist_candidate_count", "vehicle_count", "main_road_user_role", "when_source", "offline_meta_action", "what_to_show", "cue_text"]],
                hovertemplate=(
                    "time=%{x:.2f}s<br>"
                    + name
                    + "=%{y:.2f}<br>ped=%{customdata[0]}<br>cyclist=%{customdata[1]}<br>motorcyclist=%{customdata[2]}<br>veh=%{customdata[3]}<br>main_role=%{customdata[4]}<br>when_source=%{customdata[5]}<br>meta_action=%{customdata[6]}<br>what=%{customdata[7]}<br>%{customdata[8]}<extra></extra>"
                ),
            )
        )

    trigger_rows = policy[policy["explanation_active"] == 1]
    if not trigger_rows.empty:
        fig.add_trace(
            go.Scatter(
                x=trigger_rows["time_local_s"],
                y=np.ones(len(trigger_rows)) * 1.05,
                mode="markers",
                name="active frames",
                text=trigger_rows["cue_text"],
                hovertemplate="time=%{x:.2f}s<br>%{text}<extra></extra>",
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Time in selected clip (s)",
        yaxis_title="Signal value",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=55, r=25, t=95, b=55),
    )
    fig.write_html(output_html, include_plotlyjs="cdn", full_html=True)



# =============================================================================
# EXACT "WHEN" INTERVAL EXTRACTION
# =============================================================================

def find_time_column(policy_df: pd.DataFrame) -> str:
    candidates = [
        "time_local_s",
        "local_time_s",
        "clip_time_s",
        "time_s",
        "time_absolute_s",
        "absolute_time_s",
    ]
    for column in candidates:
        if column in policy_df.columns:
            return column

    if "frame_local" in policy_df.columns and "fps" in policy_df.columns:
        policy_df["time_local_s"] = pd.to_numeric(policy_df["frame_local"], errors="coerce") / pd.to_numeric(policy_df["fps"], errors="coerce").replace(0, np.nan)
        return "time_local_s"

    raise ValueError("No usable time column found in frame-level policy data.")


def ensure_explanation_active_column(policy_df: pd.DataFrame) -> pd.DataFrame:
    policy_df = policy_df.copy()

    if "explanation_active" in policy_df.columns:
        policy_df["explanation_active"] = pd.to_numeric(policy_df["explanation_active"], errors="coerce").fillna(0).astype(int)
        return policy_df

    existing_triggers = [column for column in FALLBACK_TRIGGER_COLUMNS if column in policy_df.columns]
    if not existing_triggers:
        raise ValueError("No explanation_active column and no fallback trigger columns were found.")

    active = pd.Series(0, index=policy_df.index, dtype=int)
    for column in existing_triggers:
        active = np.maximum(active, pd.to_numeric(policy_df[column], errors="coerce").fillna(0).astype(int))

    policy_df["explanation_active"] = active.astype(int)
    return policy_df


def get_mode_text(policy_slice: pd.DataFrame, column: str, default_value: str) -> str:
    if column not in policy_slice.columns:
        return default_value

    values = policy_slice[column].dropna().astype(str)
    values = values[values.str.len() > 0]
    if values.empty:
        return default_value

    return str(values.mode().iloc[0])


def get_max_score(policy_slice: pd.DataFrame, column: str) -> float:
    if column not in policy_slice.columns:
        return 0.0
    return float(pd.to_numeric(policy_slice[column], errors="coerce").fillna(0.0).max())


def get_mean_score(policy_slice: pd.DataFrame, column: str) -> float:
    if column not in policy_slice.columns:
        return 0.0
    return float(pd.to_numeric(policy_slice[column], errors="coerce").fillna(0.0).mean())


def get_trigger_fraction(policy_slice: pd.DataFrame, column: str) -> float:
    if column not in policy_slice.columns:
        return 0.0
    return float(pd.to_numeric(policy_slice[column], errors="coerce").fillna(0).astype(int).mean())


def dominant_reason(policy_slice: pd.DataFrame) -> str:
    trigger_scores: Dict[str, float] = {}
    for column in REASON_COLUMNS:
        trigger_scores[column] = get_trigger_fraction(policy_slice, column)

    best_column = max(trigger_scores, key=trigger_scores.get)
    best_value = trigger_scores[best_column]

    if best_value > 0:
        return best_column.replace("_trigger", "")

    score_candidates = {
        "risk": get_mean_score(policy_slice, "risk_score"),
        "crossing_interaction": 0.50 * get_mean_score(policy_slice, "crossing_probability") + 0.50 * get_mean_score(policy_slice, "interaction_score"),
        "anomaly": get_mean_score(policy_slice, "anomaly_score"),
    }
    return max(score_candidates, key=score_candidates.get)


def raw_active_intervals(policy_df: pd.DataFrame) -> List[Tuple[int, int]]:
    active_values = policy_df["explanation_active"].astype(int).tolist()
    intervals: List[Tuple[int, int]] = []
    inside = False
    start_idx = 0

    for idx, active in enumerate(active_values):
        if active == 1 and not inside:
            inside = True
            start_idx = idx
        if active == 0 and inside:
            intervals.append((start_idx, idx - 1))
            inside = False

    if inside:
        intervals.append((start_idx, len(active_values) - 1))

    return intervals


def merge_active_intervals(policy_df: pd.DataFrame, intervals: List[Tuple[int, int]], time_column: str) -> List[Tuple[int, int]]:
    if not intervals:
        return []

    merged = [intervals[0]]
    for start_idx, end_idx in intervals[1:]:
        previous_start, previous_end = merged[-1]
        gap_s = float(policy_df.loc[start_idx, time_column]) - float(policy_df.loc[previous_end, time_column])
        if gap_s <= MAX_GAP_TO_MERGE_S:
            merged[-1] = (previous_start, end_idx)
        else:
            merged.append((start_idx, end_idx))

    return merged


def extract_when_intervals_for_policy(candidate: pd.Series, policy: pd.DataFrame) -> pd.DataFrame:
    policy_df = ensure_explanation_active_column(policy)
    time_column = find_time_column(policy_df)
    intervals = raw_active_intervals(policy_df)
    intervals = merge_active_intervals(policy_df, intervals, time_column)

    records: List[Dict[str, object]] = []
    for interval_number, interval in enumerate(intervals, start=1):
        start_idx, end_idx = interval
        start_time_s = float(policy_df.loc[start_idx, time_column])
        end_time_s = float(policy_df.loc[end_idx, time_column])
        duration_s = end_time_s - start_time_s

        if duration_s < MIN_WHEN_INTERVAL_SECONDS:
            continue

        interval_df = policy_df.iloc[start_idx:end_idx + 1].copy()
        record: Dict[str, object] = {
            "rank": int(candidate["rank"]),
            "video_id": str(candidate["video_id"]),
            "locality": candidate.get("locality", ""),
            "country": candidate.get("country", ""),
            "clip_absolute_start_s": round(float(candidate["candidate_start_time_absolute_s"]), 2),
            "clip_absolute_end_s": round(float(candidate["candidate_end_time_absolute_s"]), 2),
            "interval_number": interval_number,
            "when_start_local_s": round(start_time_s, 2),
            "when_end_local_s": round(end_time_s, 2),
            "when_duration_s": round(duration_s, 2),
            "when_start_absolute_s": round(float(candidate["candidate_start_time_absolute_s"]) + start_time_s, 2),
            "when_end_absolute_s": round(float(candidate["candidate_start_time_absolute_s"]) + end_time_s, 2),
            "start_frame_row": int(start_idx),
            "end_frame_row": int(end_idx),
            "dominant_reason": dominant_reason(interval_df),
            "what_to_show": get_mode_text(interval_df, "what_to_show", "explanation"),
            "cue_text": get_mode_text(interval_df, "cue_text", "explanation active"),
            "explanation_style": get_mode_text(interval_df, "explanation_style", "visual cue"),
            "when_source": get_mode_text(interval_df, "when_source", ""),
            "offline_meta_action": get_mode_text(interval_df, "offline_meta_action", ""),
            "offline_explanation_reason": get_mode_text(interval_df, "offline_explanation_reason", ""),
            "offline_model_source": get_mode_text(interval_df, "offline_model_source", ""),
            "offline_reasoning_trace": get_mode_text(interval_df, "offline_reasoning_trace", ""),
        }

        for score_column in SCORE_COLUMNS:
            record[f"max_{score_column}"] = round(get_max_score(interval_df, score_column), 4)
            record[f"mean_{score_column}"] = round(get_mean_score(interval_df, score_column), 4)

        for reason_column in REASON_COLUMNS:
            record[f"fraction_{reason_column}"] = round(get_trigger_fraction(interval_df, reason_column), 4)

        records.append(record)

    return pd.DataFrame(records)


def write_readable_when_summary(intervals: pd.DataFrame, output_txt: str) -> None:
    lines: List[str] = []
    lines.append("When intervals summary")
    lines.append("======================")
    lines.append("")

    if intervals.empty:
        lines.append("No stable explanation-active intervals found.")
    else:
        for _, row in intervals.iterrows():
            lines.append(f"Rank {int(row['rank']):03d} | {row['video_id']} | {row['locality']}, {row['country']}")
            lines.append(
                f"  WHEN: local {float(row['when_start_local_s']):.2f}s to {float(row['when_end_local_s']):.2f}s "
                f"(absolute {float(row['when_start_absolute_s']):.2f}s to {float(row['when_end_absolute_s']):.2f}s)"
            )
            lines.append(f"  WHY: {row['dominant_reason']}")
            when_source = str(row.get("when_source", ""))
            if when_source:
                lines.append(f"  SOURCE: {when_source}")
            meta_action = str(row.get("offline_meta_action", ""))
            if meta_action:
                lines.append(f"  META ACTION: {meta_action}")
            offline_reason = str(row.get("offline_explanation_reason", ""))
            if offline_reason:
                lines.append(f"  OFFLINE REASON: {compact_text(offline_reason, 120)}")
            lines.append(f"  WHAT: {row['what_to_show']}")
            lines.append(f"  CUE: {row['cue_text']}")
            lines.append(f"  STYLE: {row['explanation_style']}")
            lines.append(
                f"  SCORES: risk={float(row.get('max_risk_score', 0.0)):.2f}, "
                f"crossing={float(row.get('max_crossing_probability', 0.0)):.2f}, "
                f"interaction={float(row.get('max_interaction_score', 0.0)):.2f}, "
                f"occlusion={float(row.get('max_occlusion_score', 0.0)):.2f}, "
                f"complexity={float(row.get('max_visual_complexity_score', 0.0)):.2f}, "
                f"ambiguity={float(row.get('max_offline_ambiguity_score', 0.0)):.2f}, "
                f"uncertainty={float(row.get('max_offline_uncertainty_score', 0.0)):.2f}"
            )
            lines.append("")

    output_parent = os.path.dirname(output_txt)
    if output_parent:
        os.makedirs(output_parent, exist_ok=True)

    with open(output_txt, "w", encoding="utf-8") as file_handle:
        file_handle.write("\n".join(lines))

# =============================================================================
# VIDEO OUTPUT
# =============================================================================

def get_video_resolution_label(local_path: str) -> str:
    cap = cv2.VideoCapture(local_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if width > 0 and height > 0:
        return f"{width}x{height}"
    return "unknown"


def get_video_fps(local_path: str) -> float:
    cap = cv2.VideoCapture(local_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    if fps > 0:
        return fps
    return 0.0


def save_video_response(response: requests.Response, filename_with_ext: str, source_url: str) -> Optional[str]:
    os.makedirs(SOURCE_VIDEO_DIR, exist_ok=True)
    local_path = os.path.join(SOURCE_VIDEO_DIR, filename_with_ext)

    if os.path.isfile(local_path) and os.path.getsize(local_path) > 0:
        logger.info("Using already downloaded source video: %s", local_path)
        response.close()
        return local_path

    total_text = response.headers.get("content-length", "0")
    total = int(total_text) if str(total_text).isdigit() and int(total_text) > 0 else None
    written = 0

    # Stream to a .part file and rename on success, so an interrupted download can
    # never leave a truncated video that later runs silently treat as complete.
    part_path = local_path + ".part"
    try:
        with open(part_path, "wb") as file_handle, tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=f"Downloading source video {filename_with_ext}",
        ) as bar:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file_handle.write(chunk)
                    written += len(chunk)
                    if total:
                        bar.update(len(chunk))
    except Exception as exc:
        logger.warning("Download interrupted for %s: %s (removing partial file)", filename_with_ext, exc)
        if os.path.isfile(part_path):
            os.remove(part_path)
        return None
    finally:
        response.close()

    if total is not None and written != total:
        logger.warning(
            "Download incomplete for %s: got %d of %d bytes (discarding partial file)",
            filename_with_ext,
            written,
            total,
        )
        if os.path.isfile(part_path):
            os.remove(part_path)
        return None

    os.replace(part_path, local_path)

    if os.path.isfile(local_path) and os.path.getsize(local_path) > 0:
        resolution = get_video_resolution_label(local_path)
        fps = get_video_fps(local_path)
        logger.info(
            "Downloaded source video from %s to %s | bytes=%d | resolution=%s | fps=%.3f",
            source_url,
            local_path,
            written,
            resolution,
            fps,
        )
        return local_path

    logger.warning("Download produced no usable file for %s", filename_with_ext)
    return None


def fetch_url(session: requests.Session, url: str, stream: bool) -> Optional[requests.Response]:
    try:
        response = session.get(url, timeout=FTP_TIMEOUT_SECONDS, stream=stream)
    except requests.RequestException as exc:
        logger.warning("Request failed for %s: %s", url, exc)
        return None
    logger.debug("GET %s -> %d", url, response.status_code)
    if response.status_code == 200:
        return response
    if response.status_code == 401:
        logger.warning("Authentication failed for %s", url)
    response.close()
    return None


def crawl_for_video_url(session: requests.Session, filename_with_ext: str, start_url: str) -> Optional[str]:
    filename_lower = filename_with_ext.lower()
    stack = [start_url]
    visited: set[str] = set()
    pages_seen = 0

    while stack:
        current_url = stack.pop()
        if current_url in visited:
            continue

        visited.add(current_url)
        pages_seen += 1
        if pages_seen > FTP_CRAWL_PAGE_LIMIT:
            logger.warning("Stopped FTP crawl after %d pages", FTP_CRAWL_PAGE_LIMIT)
            return None

        response = fetch_url(session, current_url, stream=False)
        if response is None:
            continue

        soup = BeautifulSoup(response.text, "html.parser")
        for link in soup.find_all("a"):
            href = str(link.get("href") or "").strip()
            if not href:
                continue

            full_url = urljoin(current_url, href)
            link_text = str(link.text or "").strip().lower()
            url_tail = os.path.basename(urlparse(full_url).path).lower()

            if "/files/" in href and (link_text == filename_lower or url_tail == filename_lower):
                return full_url

            if href.startswith("/v/") and "/browse" in href:
                stack.append(full_url)

    return None


def download_source_video_from_ftp(video_id: str) -> Optional[str]:
    if not DOWNLOAD_MISSING_SOURCE_VIDEOS:
        return None

    if not VIDEO_BASE_URL:
        logger.warning("VIDEO_BASE_URL is missing in common.get_configs('VIDEO_BASE_URL')")
        return None

    filename_with_ext = video_id if video_id.lower().endswith(".mp4") else f"{video_id}.mp4"
    base_url = VIDEO_BASE_URL if VIDEO_BASE_URL.endswith("/") else VIDEO_BASE_URL + "/"

    logger.info("Source video missing locally. Trying FTP download for %s", filename_with_ext)
    session = requests.Session()
    if VIDEO_USERNAME and VIDEO_PASSWORD:
        session.auth = (VIDEO_USERNAME, VIDEO_PASSWORD)
    session.headers.update({"User-Agent": "opticarvis-policy-demo-downloader/1.0"})

    for alias in FTP_ALIASES:
        direct_url = urljoin(base_url, f"v/{alias}/files/{filename_with_ext}")
        response = fetch_url(session, direct_url, stream=True)
        if response is not None:
            return save_video_response(response, filename_with_ext, direct_url)

    for alias in FTP_ALIASES:
        browse_url = urljoin(base_url, f"v/{alias}/browse")
        found_url = crawl_for_video_url(session, filename_with_ext, browse_url)
        if found_url is None:
            continue

        response = fetch_url(session, found_url, stream=True)
        if response is not None:
            return save_video_response(response, filename_with_ext, found_url)

    logger.warning("Source video %s was not found on the FTP server", filename_with_ext)
    return None


def find_source_video(video_id: str) -> Optional[str]:
    for ext in VIDEO_EXTENSIONS:
        path = os.path.join(SOURCE_VIDEO_DIR, f"{video_id}{ext}")
        if os.path.isfile(path):
            return path

    return download_source_video_from_ftp(video_id)


def row_to_pixel_box(row: pd.Series, frame_width: int, frame_height: int, normalised: bool) -> Tuple[int, int, int, int]:
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

    x1 = max(0, min(frame_width - 1, x1))
    y1 = max(0, min(frame_height - 1, y1))
    x2 = max(0, min(frame_width - 1, x2))
    y2 = max(0, min(frame_height - 1, y2))
    return x1, y1, x2, y2


def put_text(frame: np.ndarray, text: str, x: int, y: int, scale: float = 0.55, thickness: int = 2) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    pad = 5
    x1 = max(0, x - pad)
    y1 = max(0, y - th - baseline - pad)
    x2 = min(frame.shape[1] - 1, x + tw + pad)
    y2 = min(frame.shape[0] - 1, y + baseline + pad)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.60, frame, 0.40, 0, frame)
    cv2.putText(frame, text, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def draw_panel(frame: np.ndarray, lines: List[str]) -> None:
    y = 28
    for line in lines:
        put_text(frame, line, 15, y, scale=0.52, thickness=2)
        y += 25


def draw_tracks(
    frame: np.ndarray,
    rows: pd.DataFrame,
    normalised: bool,
    main_track_id: int,
    role_lookup: Optional[Dict[int, Dict[str, object]]] = None,
) -> None:
    h, w = frame.shape[:2]
    for _, row in rows.iterrows():
        class_id = int(row["yolo-id"])
        track_id = int(row["unique-id"])
        if class_id not in PERSON_CLASS_IDS and class_id not in VEHICLE_CLASS_IDS:
            continue
        if not DRAW_ALL_TRACKS and track_id != main_track_id:
            continue

        x1, y1, x2, y2 = row_to_pixel_box(row, w, h, normalised)
        if class_id in PERSON_CLASS_IDS:
            role = "pedestrian_candidate"
            if role_lookup is not None:
                role = str(role_lookup.get(track_id, empty_role_result(track_id)).get("road_user_role", "pedestrian_candidate"))
            if track_id == main_track_id:
                colour = (0, 255, 255)
                label = f"MAIN {role} {track_id}"
            elif role == "cyclist_candidate":
                colour = (255, 0, 255)
                label = f"cyclist {track_id}"
            elif role == "motorcyclist_candidate":
                colour = (0, 128, 255)
                label = f"motorcyclist {track_id}"
            else:
                colour = (0, 200, 0)
                label = f"ped {track_id}"
        elif class_id in BICYCLE_CLASS_IDS:
            colour = (255, 0, 255)
            label = f"bike {track_id}"
        elif class_id in MOTORCYCLE_CLASS_IDS:
            colour = (0, 128, 255)
            label = f"moto {track_id}"
        else:
            colour = (255, 160, 0)
            label = f"veh {track_id}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
        put_text(frame, label, x1, max(20, y1 - 5), scale=0.45, thickness=1)


def draw_cue(frame: np.ndarray, policy_row: pd.Series) -> None:
    if int(policy_row["explanation_active"]) != 1:
        return

    h, w = frame.shape[:2]
    risk = float(policy_row["risk_score"])
    saliency = float(policy_row["how_saliency"])
    cue_text = str(policy_row["cue_text"])
    style = str(policy_row["explanation_style"])

    box_w = min(720, w - 40)
    box_h = 88
    x1 = 20
    y1 = h - box_h - 25
    x2 = x1 + box_w
    y2 = y1 + box_h

    intensity = int(90 + 120 * saliency)
    colour = (0, max(0, 180 - intensity // 2), intensity) if risk >= RISK_TRIGGER_THRESHOLD else (0, intensity, 180)

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), colour, -1)
    cv2.addWeighted(overlay, float(policy_row["how_opacity"]), frame, 1.0 - float(policy_row["how_opacity"]), 0, frame)

    cv2.putText(frame, "AV EXPLANATION POLICY", (x1 + 15, y1 + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, cue_text, (x1 + 15, y1 + 62), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, style, (x2 - 230, y1 + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)


def create_annotated_video(
    candidate: pd.Series,
    policy: pd.DataFrame,
    output_path: str,
    tracking: Optional[pd.DataFrame] = None,
    role_lookup: Optional[Dict[int, Dict[str, object]]] = None,
) -> bool:
    source_video = find_source_video(str(candidate["video_id"]))
    if source_video is None:
        logger.warning("Source video not found for %s in %s", candidate["video_id"], SOURCE_VIDEO_DIR)
        return False

    if tracking is None:
        tracking = load_filtered_tracking(str(candidate["csv_path"]))
    normalised = looks_normalised(tracking)

    start_frame = int(candidate["start_frame_local"])
    end_frame = int(candidate["end_frame_local"])
    segment_start = int(candidate["segment_start_time_s"])
    main_track_id = int(candidate["main_pedestrian_track_id"])

    clip_rows = tracking[(tracking["frame-count"] >= start_frame) & (tracking["frame-count"] <= end_frame)]
    if role_lookup is None:
        role_lookup = build_person_role_lookup(clip_rows)
    by_frame = {int(frame): group for frame, group in clip_rows.groupby("frame-count")}
    empty_frame = clip_rows.iloc[0:0]
    policy_by_frame = {int(row["frame_local"]): row for _, row in policy.iterrows()}

    cap = cv2.VideoCapture(source_video)
    if not cap.isOpened():
        logger.warning("Could not open source video: %s", source_video)
        return False

    source_fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if source_fps <= 0 or width <= 0 or height <= 0:
        cap.release()
        logger.warning("Invalid video metadata for %s", source_video)
        return False

    csv_fps = float(candidate["fps"])
    if abs(source_fps - csv_fps) > 0.1:
        logger.warning(
            "FPS mismatch for %s: tracking CSV fps=%.2f, video fps=%.2f — overlays may desync",
            source_video,
            csv_fps,
            source_fps,
        )

    absolute_start_frame = int(round(segment_start * source_fps)) + start_frame
    n_frames = end_frame - start_frame + 1
    absolute_end_frame = absolute_start_frame + n_frames - 1
    if total_frames > 0:
        absolute_start_frame = max(0, min(absolute_start_frame, total_frames - 1))
        absolute_end_frame = max(absolute_start_frame, min(absolute_end_frame, total_frames - 1))

    output_parent = os.path.dirname(output_path)
    if output_parent:
        os.makedirs(output_parent, exist_ok=True)
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*OUTPUT_FOURCC), source_fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise IOError(f"Could not create video writer: {output_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, absolute_start_frame)
    current_source_frame = absolute_start_frame
    local_frame = start_frame

    progress_total = max(0, absolute_end_frame - absolute_start_frame + 1)
    progress = tqdm(total=progress_total, desc="Rendering annotated policy video")
    while current_source_frame <= absolute_end_frame and local_frame <= end_frame:
        ok, frame = cap.read()
        if not ok:
            break

        frame_rows = by_frame.get(local_frame, empty_frame)
        policy_row = policy_by_frame.get(local_frame)
        draw_tracks(frame, frame_rows, normalised, main_track_id, role_lookup=role_lookup)

        if policy_row is not None:
            draw_cue(frame, policy_row)
            panel_lines = [
                f"{candidate['video_id']} | {candidate['locality']}, {candidate['country']}",
                f"clip {float(policy_row['time_local_s']):05.2f}s | frame {local_frame}",
                f"main role: {policy_row['main_road_user_role']}",
                f"what: {policy_row['what_to_show']}",
                f"risk={float(policy_row['risk_score']):.2f} cross={float(policy_row['crossing_probability']):.2f} int={float(policy_row['interaction_score']):.2f}",
                f"ped={int(policy_row['pedestrian_count'])} cyc={int(policy_row['cyclist_candidate_count'])} moto={int(policy_row['motorcyclist_candidate_count'])} active={int(policy_row['explanation_active'])}",
            ]
            draw_panel(frame, panel_lines)

        writer.write(frame)
        current_source_frame += 1
        local_frame += 1
        progress.update(1)

    progress.close()
    cap.release()
    writer.release()
    return os.path.exists(output_path) and os.path.getsize(output_path) > 0


# =============================================================================
# MAIN
# =============================================================================

def write_run_metadata(output_dir: str) -> None:
    """Record the CONFIG values that produced this run, so outputs stay comparable across tuning runs."""
    metadata = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "external_signal_csv": EXTERNAL_SIGNAL_PATH,
        "prefer_offline_model_when": PREFER_OFFLINE_MODEL_WHEN,
        "use_proxy_when_if_offline_missing": USE_PROXY_WHEN_IF_OFFLINE_MISSING,
        "reuse_existing_ranking": REUSE_EXISTING_RANKING,
        "filters": {
            "only_localities": sorted(ONLY_LOCALITIES),
            "only_countries": sorted(ONLY_COUNTRIES),
            "only_iso3": sorted(ONLY_ISO3),
            "only_video_ids": sorted(ONLY_VIDEO_IDS),
        },
        "discovery": {
            "max_tracking_csv_files": MAX_TRACKING_CSV_FILES,
            "min_tracking_csv_bytes": MIN_TRACKING_CSV_BYTES,
            "prefer_local_source_videos": PREFER_LOCAL_SOURCE_VIDEOS,
            "clip_seconds": CLIP_SECONDS,
            "stride_seconds": STRIDE_SECONDS,
            "top_k_clips": TOP_K_CLIPS,
            "max_windows_per_csv": MAX_WINDOWS_PER_CSV,
            "min_confidence": MIN_CONFIDENCE,
            "min_main_pedestrian_seconds": MIN_MAIN_PEDESTRIAN_SECONDS,
        },
        "thresholds": {
            "crossing_trigger": CROSSING_TRIGGER_THRESHOLD,
            "interaction_trigger": INTERACTION_TRIGGER_THRESHOLD,
            "density_trigger": DENSITY_TRIGGER_THRESHOLD,
            "occlusion_trigger": OCCLUSION_TRIGGER_THRESHOLD,
            "risk_trigger": RISK_TRIGGER_THRESHOLD,
            "anomaly_trigger": ANOMALY_TRIGGER_THRESHOLD,
            "direct_when_crossing": DIRECT_WHEN_CROSSING_THRESHOLD,
            "direct_when_interaction": DIRECT_WHEN_INTERACTION_THRESHOLD,
            "direct_when_risk": DIRECT_WHEN_RISK_THRESHOLD,
            "direct_when_occlusion_crossing": DIRECT_WHEN_OCCLUSION_CROSSING_THRESHOLD,
            "direct_when_occlusion_interaction": DIRECT_WHEN_OCCLUSION_INTERACTION_THRESHOLD,
            "persistence_seconds": PERSISTENCE_SECONDS,
            "offline_ambiguity": OFFLINE_AMBIGUITY_THRESHOLD,
            "offline_uncertainty": OFFLINE_UNCERTAINTY_THRESHOLD,
            "offline_trajectory_conflict": OFFLINE_TRAJECTORY_CONFLICT_THRESHOLD,
            "offline_low_confidence": OFFLINE_LOW_CONFIDENCE_THRESHOLD,
            "style_complexity": STYLE_COMPLEXITY_THRESHOLD,
            "min_when_interval_seconds": MIN_WHEN_INTERVAL_SECONDS,
            "max_gap_to_merge_s": MAX_GAP_TO_MERGE_S,
        },
    }
    metadata_path = os.path.join(output_dir, "run_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as file_handle:
        json.dump(metadata, file_handle, indent=2)
    logger.info("Run metadata saved to %s", metadata_path)


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    write_run_metadata(OUTPUT_DIR)

    model_mapping = build_model_signal_mapping()
    model_mapping_path = os.path.join(OUTPUT_DIR, "mark_model_signal_mapping.csv")
    model_mapping.to_csv(model_mapping_path, index=False)

    offline_template = build_offline_model_signal_template()
    offline_template_path = os.path.join(OUTPUT_DIR, OFFLINE_MODEL_SIGNAL_TEMPLATE_FILENAME)
    offline_template.to_csv(offline_template_path, index=False)

    mapping_segments = explode_mapping(MAPPING_CSV)
    mapping_segments_path = os.path.join(OUTPUT_DIR, "crowd_mapping_segments_exploded.csv")
    mapping_segments.to_csv(mapping_segments_path, index=False)
    mapping_lookup = build_mapping_lookup(mapping_segments)

    logger.info("Tracking CSV directory: %s", TRACKING_CSV_DIR)

    ranked_path = os.path.join(OUTPUT_DIR, "ranked_stress_test_clips.csv")
    if REUSE_EXISTING_RANKING and os.path.isfile(ranked_path):
        logger.info("REUSE_EXISTING_RANKING=True: reusing cached ranking from %s (skipping CSV scoring)", ranked_path)
        candidates = pd.read_csv(ranked_path)
        candidates["video_id"] = candidates["video_id"].astype(str)
    else:
        tracking_csvs = find_tracking_csvs(mapping_lookup)
        if not tracking_csvs:
            raise FileNotFoundError(
                "No YOLOv11 + BoT SORT tracking CSV files matched mapping.csv. Check config['data'], CSV filenames, and mapping.csv."
            )

        logger.info("Scoring %d selected YOLOv11 + BoT SORT tracking CSV files", len(tracking_csvs))
        all_scores: List[Dict[str, object]] = []
        skipped_csvs = 0
        for csv_path in tqdm(tracking_csvs, desc="Scoring stress test clips"):
            try:
                all_scores.extend(score_tracking_csv(csv_path, mapping_lookup))
            except Exception as exc:
                skipped_csvs += 1
                logger.warning("Skipping unreadable tracking CSV %s: %s", csv_path, exc)
        if skipped_csvs:
            logger.warning("Skipped %d tracking CSV files due to read/parse errors", skipped_csvs)

        if not all_scores:
            raise ValueError("No candidate windows were scored. Check YOLOv11 + BoT SORT tracking CSV content and confidence threshold.")

        candidates = pd.DataFrame(all_scores)
        candidates = candidates[candidates["stress_test_score"] > 0].copy()
        candidates = candidates.sort_values("stress_test_score", ascending=False).reset_index(drop=True)
        candidates.insert(0, "rank", np.arange(1, len(candidates) + 1))

        candidates.to_csv(ranked_path, index=False)

    top = candidates.head(TOP_K_CLIPS).copy()
    top_path = os.path.join(OUTPUT_DIR, "selected_stress_test_clips.csv")
    top.to_csv(top_path, index=False)

    external_signals = load_external_signals(EXTERNAL_SIGNAL_PATH)
    all_when_intervals: List[pd.DataFrame] = []

    expected_clip_dirs = {f"rank_{int(row['rank']):03d}_{row['video_id']}" for _, row in top.iterrows()}
    stale_clip_dirs = [
        path for path in sorted(glob.glob(os.path.join(OUTPUT_DIR, "rank_*")))
        if os.path.isdir(path) and os.path.basename(path) not in expected_clip_dirs
    ]
    if stale_clip_dirs:
        logger.warning(
            "OUTPUT_DIR contains %d rank_* directories from a previous run that this run will not refresh: %s",
            len(stale_clip_dirs),
            ", ".join(os.path.basename(path) for path in stale_clip_dirs),
        )

    for clip_index, (_, candidate) in enumerate(top.iterrows(), start=1):
        rank = int(candidate["rank"])
        logger.info("Processing clip %d/%d: rank %03d %s", clip_index, len(top), rank, candidate["video_id"])
        clip_dir = os.path.join(OUTPUT_DIR, f"rank_{rank:03d}_{candidate['video_id']}")
        os.makedirs(clip_dir, exist_ok=True)

        # Load and role-classify the clip once; the policy and video phases share it.
        tracking = load_filtered_tracking(str(candidate["csv_path"]))
        start_frame = int(candidate["start_frame_local"])
        end_frame = int(candidate["end_frame_local"])
        clip_rows = tracking[(tracking["frame-count"] >= start_frame) & (tracking["frame-count"] <= end_frame)].copy()
        role_lookup = build_person_role_lookup(clip_rows)

        policy = compute_policy_for_clip(candidate, external_signals, tracking=tracking, role_lookup=role_lookup)
        policy_csv = os.path.join(clip_dir, FRAME_POLICY_FILENAME)
        policy.to_csv(policy_csv, index=False)

        when_intervals = extract_when_intervals_for_policy(candidate, policy)
        when_intervals_csv = os.path.join(clip_dir, PER_CLIP_WHEN_INTERVALS_FILENAME)
        when_intervals.to_csv(when_intervals_csv, index=False)
        if not when_intervals.empty:
            all_when_intervals.append(when_intervals)

        title = (
            f"Rank {rank}: {candidate['video_id']} | "
            f"{candidate.get('locality', '')}, {candidate.get('country', '')} | "
            f"abs {candidate['candidate_start_time_absolute_s']}s to {candidate['candidate_end_time_absolute_s']}s"
        )
        timeline_html = os.path.join(clip_dir, "policy_timeline.html")
        save_policy_timeline(policy, timeline_html, title)

        if CREATE_ANNOTATED_VIDEO:
            video_path = os.path.join(clip_dir, "annotated_policy_demo.mp4")
            try:
                created = create_annotated_video(candidate, policy, video_path, tracking=tracking, role_lookup=role_lookup)
            except Exception as exc:
                created = False
                logger.warning("Annotated video failed for rank %03d (%s): %s", rank, candidate["video_id"], exc)
            if created:
                logger.info("Saved annotated demo video: %s", video_path)

    if all_when_intervals:
        when_summary = pd.concat(all_when_intervals, ignore_index=True)
    else:
        when_summary = pd.DataFrame()

    when_summary_path = os.path.join(OUTPUT_DIR, WHEN_INTERVALS_SUMMARY_FILENAME)
    when_summary.to_csv(when_summary_path, index=False)
    when_readable_path = os.path.join(OUTPUT_DIR, WHEN_INTERVALS_READABLE_FILENAME)
    write_readable_when_summary(when_summary, when_readable_path)

    print("Done.")
    print(f"Model signal mapping: {model_mapping_path}")
    print(f"Exploded CROWD mapping: {mapping_segments_path}")
    print(f"Ranked stress test clips: {ranked_path}")
    print(f"Selected stress test clips: {top_path}")
    print(f"Per clip outputs: {os.path.abspath(OUTPUT_DIR)}")
    print(f"When intervals CSV: {when_summary_path}")
    print(f"Readable when summary: {when_readable_path}")
    if when_summary.empty:
        print("No stable explanation-active intervals found.")
    else:
        when_display_cols = [
            "rank",
            "video_id",
            "when_start_local_s",
            "when_end_local_s",
            "when_start_absolute_s",
            "when_end_absolute_s",
            "dominant_reason",
            "when_source",
            "what_to_show",
        ]
        print("\nDetected when intervals:")
        print(when_summary[when_display_cols].to_string(index=False))
    print("\nTop selected clips:")
    display_cols = [
        "rank",
        "video_id",
        "locality",
        "country",
        "candidate_start_time_absolute_s",
        "candidate_end_time_absolute_s",
        "stress_test_score",
        "main_pedestrian_track_id",
        "main_road_user_role",
        "crossing_proxy",
        "interaction_score",
        "avg_total_density",
        "occlusion_proxy",
    ]
    print(top[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
