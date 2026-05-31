"""
Select candidate CROWD clips from a directory of YOLOv11 + BoT SORT CSV files.

This version matches the existing processing pipeline:
- main.py writes tracking CSVs to data_path/bbox.
- File names look like: {video_id}_{segment_start_time}_{fps}.csv
- The CSV frame-count values are local to the trimmed segment.
- To extract from the original video, absolute start time is:
      source frame = round(segment_start_time * source_fps) + local candidate frame

No command-line parser is used. Edit the CONFIG section and run the file.

Expected bbox CSV columns
-------------------------
yolo-id, x-center, y-center, width, height, unique-id, confidence, frame-count

Main outputs
------------
OUTPUT_DIR/ranked_clip_candidates.csv
OUTPUT_DIR/timeline_rank_001.html
OUTPUT_DIR/timeline_rank_001.csv
OUTPUT_DIR/extracted_clips/candidate_rank_001_....mp4

Frame-rate note
---------------
The bbox CSV filename may contain decimal FPS values such as 29.97. This script
preserves FPS as a float. Window selection is stored using integer frame counts,
and clip extraction uses source-video frame indices rather than ffmpeg time
seeking wherever possible.
"""

from __future__ import annotations

import ast
import logging
import os
import pathlib
import subprocess
import common
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


# =============================================================================
# CONFIG
# =============================================================================

# Directory that contains the tracking CSV files.
TRACKING_CSV_DIR = common.get_configs("data")
CSV_GLOB = "*.csv"
RECURSIVE_CSV_SEARCH = False

# Mapping file used by your existing pipeline.
# This is needed to recover metadata and absolute video time.
MAPPING_CSV = common.get_configs("mapping")

OUTPUT_DIR = common.get_configs("output_dir")

# Clip mining settings. For Mark's request, 30 seconds is a good first target.
CLIP_SECONDS = 30
STRIDE_SECONDS = 5
TOP_K_OUTPUTS = 10
MAX_CSV_FILES: Optional[int] = None  # set to an int for quick debugging, otherwise None

# Keep only CSV files that are present in the mapping CSV.
# This prevents old/unrelated bbox CSVs from being scored.
FILTER_TO_MAPPING = True
SORT_CSV_FILES_BY = "size_desc"  # options: "size_desc", "name"

# Optional narrow filters. Leave empty to disable.
# Examples:
# SCORE_ONLY_VIDEO_IDS = {"3ai7SUaPoHM"}
# SCORE_ONLY_COUNTRIES = {"India", "Japan"}
SCORE_ONLY_LOCALITIES = {"Tokyo"}
# SCORE_ONLY_ISO3 = {"IND", "JPN"}
SCORE_ONLY_VIDEO_IDS: set[str] = set()
SCORE_ONLY_COUNTRIES: set[str] = set()
# SCORE_ONLY_LOCALITIES: set[str] = set()
SCORE_ONLY_ISO3: set[str] = set()

# Speed controls. These keep each individual CSV manageable.
# MAX_WINDOWS_PER_CSV uniformly samples sliding windows inside long segments.
# OCCLUSION_FRAME_STEP computes the occlusion proxy on every nth frame only.
MAX_WINDOWS_PER_CSV: Optional[int] = 20
OCCLUSION_FRAME_STEP = 10

# COCO-style class IDs. Change if your YOLO model uses different ids.
PERSON_CLASS_IDS = {0}
VEHICLE_CLASS_IDS = {2, 3, 5, 7}

MIN_CONFIDENCE = 0.40
MIN_PERSON_TRACK_SECONDS = 3.00

# Interaction region for normalised coordinates. Adjust after inspecting timelines.
ROI_X_MIN = 0.25
ROI_X_MAX = 0.75
ROI_Y_MIN = 0.35
ROI_Y_MAX = 1.00

# Scoring weights. These are for clip mining only, not a final scientific metric.
WEIGHTS = {
    "track_duration": 2.00,
    "crossing_motion": 2.00,
    "central_presence": 1.50,
    "area_growth": 1.00,
    "vehicle_presence": 1.25,
    "traffic_density": 0.75,
    "occlusion": 0.75,
}

# ----------------------------- Video extraction -----------------------------
# If True, the script will extract MP4 candidate clips for top ranked windows.
EXTRACT_TOP_CLIPS = True
FFMPEG_BINARY = "ffmpeg"

# First, the script tries to find {video_id}.mp4 in these local directories.
VIDEO_SEARCH_DIRS = [
    "path/to/videos",
]

# If the video is not found locally, the script can download it from the file server.
DOWNLOAD_MISSING_VIDEOS = True
VIDEO_BASE_URL = common.get_configs("VIDEO_BASE_URL")
VIDEO_DOWNLOAD_DIR = common.get_configs("videos")
VIDEO_USERNAME: Optional[str] = common.get_secrets("ftp_username")
VIDEO_PASSWORD: Optional[str] = common.get_secrets("55bEezkFaBUbRDg")
VIDEO_TOKEN: Optional[str] = None

# If True, only videos downloaded by this script during this run are deleted
# after all requested snippets have been extracted. Local videos found in
# VIDEO_SEARCH_DIRS are never deleted.
DELETE_DOWNLOADED_VIDEOS_AFTER_EXTRACTION = False

# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# MAPPING AND FILE DISCOVERY
# =============================================================================

def parse_video_ids(value: object) -> List[str]:
    """Parse the mapping 'videos' column, which is stored like [id1,id2,...]."""
    text = str(value).strip()
    if not text or text == "nan":
        return []
    text = text.strip("[]")
    return [x.strip().strip("'\"") for x in text.split(",") if x.strip()]


def safe_literal_list(value: object) -> list:
    """Safely parse columns such as start_time, end_time, and time_of_day."""
    try:
        parsed = ast.literal_eval(str(value))
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []


def load_mapping_lookup(mapping_csv: str) -> Dict[Tuple[str, int], Dict[str, object]]:
    """
    Build lookup: (video_id, segment_start_time) -> metadata.

    The existing main.py loops through mapping rows, parses videos, start_time,
    end_time and time_of_day, then processes each segment. This mirrors that logic.
    """
    if not mapping_csv or not os.path.isfile(mapping_csv):
        logger.warning("Mapping CSV not found. Segment end time and metadata will be unavailable.")
        return {}

    mapping = pd.read_csv(mapping_csv)
    lookup: Dict[Tuple[str, int], Dict[str, object]] = {}

    for _, row in mapping.iterrows():
        videos = parse_video_ids(row.get("videos", ""))
        start_times = safe_literal_list(row.get("start_time", "[]"))
        end_times = safe_literal_list(row.get("end_time", "[]"))
        time_of_day = safe_literal_list(row.get("time_of_day", "[]"))

        for video_idx, vid in enumerate(videos):
            st_list = start_times[video_idx] if video_idx < len(start_times) else []
            et_list = end_times[video_idx] if video_idx < len(end_times) else []
            tod_list = time_of_day[video_idx] if video_idx < len(time_of_day) else []

            if not isinstance(st_list, list):
                st_list = [st_list]
            if not isinstance(et_list, list):
                et_list = [et_list]
            if not isinstance(tod_list, list):
                tod_list = [tod_list]

            for seg_idx, st in enumerate(st_list):
                try:
                    st_int = int(st)
                except Exception:
                    continue

                et_value = None
                tod_value = None
                if seg_idx < len(et_list):
                    try:
                        et_value = int(et_list[seg_idx])
                    except Exception:
                        et_value = None
                if seg_idx < len(tod_list):
                    tod_value = tod_list[seg_idx]

                lookup[(vid, st_int)] = {
                    "video_id": vid,
                    "segment_start_time_s": st_int,
                    "segment_end_time_s": et_value,
                    "time_of_day": tod_value,
                    "locality": row.get("locality", ""),
                    "country": row.get("country", ""),
                    "iso3": row.get("iso3", ""),
                    "continent": row.get("continent", ""),
                    "traffic_index": row.get("traffic_index", np.nan),
                    "vehicle_type": row.get("vehicle_type", ""),
                    "upload_date": row.get("upload_date", ""),
                    "channel": row.get("channel", ""),
                }

    logger.info("Loaded %d mapping segment entries.", len(lookup))
    return lookup


def parse_tracking_csv_name(csv_path: Path) -> Optional[Dict[str, object]]:
    """
    Parse {video_id}_{segment_start}_{fps}.csv.

    Uses rsplit so video IDs containing underscores still work.
    """
    stem = csv_path.stem
    parts = stem.rsplit("_", 2)
    if len(parts) != 3:
        logger.warning("Skipping CSV with unexpected name: %s", csv_path.name)
        return None

    video_id, start_text, fps_text = parts
    try:
        segment_start_time_s = int(start_text)
        fps = float(fps_text)
    except Exception:
        logger.warning("Skipping CSV with unparseable start/fps: %s", csv_path.name)
        return None

    return {
        "video_id": video_id,
        "segment_start_time_s": segment_start_time_s,
        "fps": fps,
        "csv_path": str(csv_path),
    }


def find_tracking_csvs(mapping_lookup: Optional[Dict[Tuple[str, int], Dict[str, object]]] = None) -> List[Path]:
    """
    Find tracking CSVs and apply cheap filename-level filters before scoring.

    This is intentionally conservative because the bbox directory can contain
    tens of thousands of CSVs. Scoring all of them is unnecessary for the first
    stress-test clip and can take many hours.
    """
    root = Path(TRACKING_CSV_DIR)
    pattern = f"**/{CSV_GLOB}" if RECURSIVE_CSV_SEARCH else CSV_GLOB
    all_csvs = sorted(root.glob(pattern))
    logger.info("Found %d tracking CSV files in %s", len(all_csvs), root)

    filtered: List[Path] = []
    skipped_not_in_mapping = 0
    skipped_by_user_filter = 0
    skipped_bad_name = 0

    for csv_path in all_csvs:
        parsed = parse_tracking_csv_name(csv_path)
        if parsed is None:
            skipped_bad_name += 1
            continue

        video_id = str(parsed["video_id"])
        segment_start = int(parsed["segment_start_time_s"])
        key = (video_id, segment_start)
        meta = mapping_lookup.get(key, {}) if mapping_lookup else {}

        if FILTER_TO_MAPPING and mapping_lookup is not None and key not in mapping_lookup:
            skipped_not_in_mapping += 1
            continue

        if SCORE_ONLY_VIDEO_IDS and video_id not in SCORE_ONLY_VIDEO_IDS:
            skipped_by_user_filter += 1
            continue
        if SCORE_ONLY_COUNTRIES and str(meta.get("country", "")) not in SCORE_ONLY_COUNTRIES:
            skipped_by_user_filter += 1
            continue
        if SCORE_ONLY_LOCALITIES and str(meta.get("locality", "")) not in SCORE_ONLY_LOCALITIES:
            skipped_by_user_filter += 1
            continue
        if SCORE_ONLY_ISO3 and str(meta.get("iso3", "")) not in SCORE_ONLY_ISO3:
            skipped_by_user_filter += 1
            continue

        filtered.append(csv_path)

    if SORT_CSV_FILES_BY == "size_desc":
        filtered = sorted(filtered, key=lambda p: p.stat().st_size if p.exists() else 0, reverse=True)
    elif SORT_CSV_FILES_BY == "name":
        filtered = sorted(filtered)
    else:
        logger.warning("Unknown SORT_CSV_FILES_BY=%s. Keeping filename order.", SORT_CSV_FILES_BY)

    if MAX_CSV_FILES is not None and len(filtered) > MAX_CSV_FILES:
        logger.info(
            "Keeping only %d CSV files for scoring. Increase MAX_CSV_FILES or add filters if needed.",
            MAX_CSV_FILES,
        )
        filtered = filtered[:MAX_CSV_FILES]

    logger.info(
        "CSV filter summary: scoring=%d | bad_name=%d | not_in_mapping=%d | user_filter=%d",
        len(filtered),
        skipped_bad_name,
        skipped_not_in_mapping,
        skipped_by_user_filter,
    )

    return filtered


# =============================================================================
# VIDEO DOWNLOADER AND EXTRACTION
# =============================================================================

class VideoDownloader:
    """Standalone version of your FTP / HTTP file-server downloader."""

    @staticmethod
    def get_video_fps(local_path: str) -> float:
        try:
            import cv2

            cap = cv2.VideoCapture(local_path)
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            cap.release()
            return fps if fps > 0 else 0.0
        except Exception:
            return 0.0

    @staticmethod
    def get_video_resolution_label(local_path: str) -> str:
        try:
            import cv2

            cap = cv2.VideoCapture(local_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            if width > 0 and height > 0:
                return f"{width}x{height}"
        except Exception:
            pass
        return "unknown"

    def download_videos_from_ftp(
        self,
        filename: str,
        base_url: Optional[str] = None,
        out_dir: str = ".",
        username: Optional[str] = None,
        password: Optional[str] = None,
        token: Optional[str] = None,
        timeout: int = 20,
        debug: bool = True,
        max_pages: int = 500,
    ) -> Optional[Tuple[str, str, str, float]]:
        if not base_url:
            logger.error("Base URL is missing.")
            return None

        base = base_url if base_url.endswith("/") else base_url + "/"

        if username == "":
            username = None
        if password == "":
            password = None

        filename_with_ext = filename if filename.lower().endswith(".mp4") else f"{filename}.mp4"
        filename_lower = filename_with_ext.lower()
        aliases = ["tue1", "tue2", "tue3", "tue4"]
        req_params = {"token": token} if token else None

        logger.info("Starting download for '%s'", filename_with_ext)

        with requests.Session() as session:
            if username and password:
                session.auth = (username, password)
            session.headers.update({"User-Agent": "multi-fileserver-downloader/1.0"})

            def fetch(url: str, stream: bool = False) -> Optional[requests.Response]:
                try:
                    r = session.get(url, timeout=timeout, params=req_params, stream=stream)
                    logger.debug("GET %s -> %s", url, r.status_code)
                    if r.status_code == 401:
                        logger.error("Authentication failed for %s", url)
                    r.raise_for_status()
                    return r
                except requests.RequestException as e:
                    logger.warning("Request failed [%s]: %s", url, e)
                    return None

            def save_response(response: requests.Response, source_url: str) -> Optional[str]:
                os.makedirs(out_dir, exist_ok=True)
                local_path = os.path.join(out_dir, filename_with_ext)

                if os.path.exists(local_path):
                    stem, suffix = os.path.splitext(local_path)
                    i = 1
                    while os.path.exists(f"{stem} ({i}){suffix}"):
                        i += 1
                    local_path = f"{stem} ({i}){suffix}"

                total = int(response.headers.get("content-length", 0)) or None
                try:
                    written = 0
                    with open(local_path, "wb") as f, tqdm(
                        total=total,
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                        desc=f"Downloading {filename_with_ext}",
                    ) as bar:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                f.write(chunk)
                                written += len(chunk)
                                if total:
                                    bar.update(len(chunk))
                    logger.info("Downloaded %s from %s (%d bytes)", local_path, source_url, written)
                    return local_path
                except Exception as e:
                    logger.error("Download failed for %s: %s", filename_with_ext, e)
                    return None

            # 1. Try direct /files paths.
            for alias in aliases:
                direct_url = urljoin(base, f"v/{alias}/files/{filename_with_ext}")
                r = fetch(direct_url, stream=True)
                if r is None:
                    continue

                local_path = save_response(r, direct_url)
                if not local_path:
                    return None
                resolution = self.get_video_resolution_label(local_path)
                fps = float(self.get_video_fps(local_path))
                return local_path, filename, resolution, fps

            # 2. Crawl /browse fallback.
            visited: Set[str] = set()

            def is_dir_link(href: str) -> bool:
                return href.startswith("/v/") and "/browse" in href

            def is_file_link(href: str) -> bool:
                return "/files/" in href

            def crawl(start_url: str) -> Optional[str]:
                stack = [start_url]
                pages_seen = 0

                while stack:
                    url = stack.pop()
                    if url in visited:
                        continue
                    visited.add(url)
                    pages_seen += 1
                    if pages_seen > max_pages:
                        logger.warning("Crawl aborted after %d pages.", max_pages)
                        return None

                    resp = fetch(url)
                    if resp is None:
                        continue

                    try:
                        soup = BeautifulSoup(resp.text, "html.parser")
                    except Exception as e:
                        logger.warning("HTML parse failed at %s: %s", url, e)
                        continue

                    for a in soup.find_all("a"):
                        href = (a.get("href") or "").strip()
                        if not href:
                            continue
                        full = urljoin(url, href)

                        if is_file_link(href):
                            anchor_text = (a.text or "").strip().lower()
                            tail = pathlib.PurePosixPath(urlparse(full).path).name.lower()
                            if anchor_text == filename_lower or tail == filename_lower:
                                return full

                        if is_dir_link(href):
                            stack.append(full)

                return None

            for alias in aliases:
                start_url = urljoin(base, f"v/{alias}/browse")
                found = crawl(start_url)
                if not found:
                    continue
                r = fetch(found, stream=True)
                if not r:
                    continue

                local_path = save_response(r, found)
                if not local_path:
                    return None
                resolution = self.get_video_resolution_label(local_path)
                fps = float(self.get_video_fps(local_path))
                return local_path, filename, resolution, fps

            logger.warning("File '%s' not found in any alias.", filename_with_ext)
            return None


def find_local_video(video_id: str) -> Optional[str]:
    for directory in VIDEO_SEARCH_DIRS:
        if not directory:
            continue
        candidate = Path(directory) / f"{video_id}.mp4"
        if candidate.is_file():
            return str(candidate)
    return None


def resolve_video_path(video_id: str) -> Optional[Tuple[str, bool]]:
    """
    Resolve the source video path for a video ID.

    Returns
    -------
    Optional[Tuple[str, bool]]
        (video_path, downloaded_by_this_script). The second value is True only
        when this function downloaded the video during the current run. Local
        videos found in VIDEO_SEARCH_DIRS are marked False and are never deleted
        by the cleanup flag.
    """
    local = find_local_video(video_id)
    if local:
        return local, False

    if not DOWNLOAD_MISSING_VIDEOS:
        return None

    downloader = VideoDownloader()
    result = downloader.download_videos_from_ftp(
        filename=video_id,
        base_url=VIDEO_BASE_URL,
        out_dir=VIDEO_DOWNLOAD_DIR,
        username=VIDEO_USERNAME,
        password=VIDEO_PASSWORD,
        token=VIDEO_TOKEN,
        timeout=20,
        debug=True,
        max_pages=500,
    )
    if not result:
        return None

    local_path, _, _, _ = result
    return local_path, True


def delete_downloaded_video_if_requested(video_path: str, extraction_succeeded: bool) -> None:
    """Delete a downloaded source video when cleanup is enabled and safe."""
    if not DELETE_DOWNLOADED_VIDEOS_AFTER_EXTRACTION:
        return

    if not extraction_succeeded:
        logger.warning(
            "Keeping downloaded video because at least one extraction failed: %s",
            video_path,
        )
        return

    try:
        os.remove(video_path)
        logger.info("Deleted downloaded source video after extraction: %s", video_path)
    except FileNotFoundError:
        logger.info("Downloaded source video was already removed: %s", video_path)
    except Exception as e:
        logger.warning("Could not delete downloaded source video %s: %s", video_path, e)


def get_video_fps_from_file(video_path: str) -> float:
    """Read the source video FPS as a float, for example 29.97."""
    try:
        import cv2

        cap = cv2.VideoCapture(video_path)
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        cap.release()
        return fps if fps > 0 else 0.0
    except Exception as e:
        logger.warning("Could not read FPS for %s: %s", video_path, e)
        return 0.0


def extract_clip_by_frame_count(
    source_video: str,
    segment_start_time_s: float,
    start_frame_local: int,
    end_frame_local: int,
    output_path: Path,
) -> bool:
    """
    Extract a candidate clip using frame counts instead of ffmpeg time seeking.

    The tracking CSV frame-count is local to the trimmed source segment. To map
    it back to the original source video we estimate the absolute source frame as:

        round(segment_start_time_s * source_video_fps) + local_frame

    This preserves the exact number of tracker frames in the extracted snippet.
    The output has no audio, which is acceptable for the visual feasibility demo.
    """
    try:
        import cv2
    except Exception as e:
        logger.warning("OpenCV is unavailable; cannot extract by frame count: %s", e)
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(source_video)
    if not cap.isOpened():
        logger.warning("Could not open video for frame-based extraction: %s", source_video)
        return False

    source_fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if source_fps <= 0 or width <= 0 or height <= 0:
        cap.release()
        logger.warning("Invalid source video metadata for %s", source_video)
        return False

    absolute_start_frame = int(round(segment_start_time_s * source_fps)) + int(start_frame_local)
    n_frames = int(end_frame_local) - int(start_frame_local) + 1
    absolute_end_frame = absolute_start_frame + n_frames - 1

    if total_frames > 0:
        absolute_start_frame = max(0, min(absolute_start_frame, total_frames - 1))
        absolute_end_frame = max(absolute_start_frame, min(absolute_end_frame, total_frames - 1))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, source_fps, (width, height))
    if not writer.isOpened():
        cap.release()
        logger.warning("Could not open video writer for %s", output_path)
        return False

    cap.set(cv2.CAP_PROP_POS_FRAMES, absolute_start_frame)

    written = 0
    current_frame = absolute_start_frame
    while current_frame <= absolute_end_frame:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        written += 1
        current_frame += 1

    cap.release()
    writer.release()

    if written > 0 and output_path.exists() and output_path.stat().st_size > 0:
        logger.info(
            "Extracted %s using frame counts: source frames %d-%d, written=%d, fps=%.3f",
            output_path,
            absolute_start_frame,
            absolute_end_frame,
            written,
            source_fps,
        )
        return True

    logger.warning("Frame-based extraction wrote no frames for %s", output_path)
    return False


def extract_clip_with_ffmpeg(source_video: str, start_time_s: float, duration_s: float, output_path: Path) -> bool:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd_copy = [
        FFMPEG_BINARY,
        "-y",
        "-ss",
        f"{start_time_s:.3f}",
        "-i",
        source_video,
        "-t",
        f"{duration_s:.3f}",
        "-c",
        "copy",
        "-avoid_negative_ts",
        "make_zero",
        str(output_path),
    ]
    result = subprocess.run(cmd_copy, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0:
        return True

    logger.warning("ffmpeg stream copy failed for %s. Retrying with re-encoding.", output_path.name)
    cmd_reencode = [
        FFMPEG_BINARY,
        "-y",
        "-ss",
        f"{start_time_s:.3f}",
        "-i",
        source_video,
        "-t",
        f"{duration_s:.3f}",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-c:a",
        "aac",
        str(output_path),
    ]
    result = subprocess.run(cmd_reencode, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0:
        return True

    logger.error("ffmpeg failed for %s: %s", output_path, result.stderr[-1000:])
    return False


# =============================================================================
# TRACKING CSV HELPERS
# =============================================================================

def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
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
    df = df.copy()
    df["x1"] = df["x-center"] - df["width"] / 2.0
    df["y1"] = df["y-center"] - df["height"] / 2.0
    df["x2"] = df["x-center"] + df["width"] / 2.0
    df["y2"] = df["y-center"] + df["height"] / 2.0
    return df


def iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x_left = np.maximum(box[0], boxes[:, 0])
    y_top = np.maximum(box[1], boxes[:, 1])
    x_right = np.minimum(box[2], boxes[:, 2])
    y_bottom = np.minimum(box[3], boxes[:, 3])

    inter_w = np.maximum(0.0, x_right - x_left)
    inter_h = np.maximum(0.0, y_bottom - y_top)
    inter_area = inter_w * inter_h

    box_area = max(0.0, (box[2] - box[0]) * (box[3] - box[1]))
    boxes_area = np.maximum(0.0, (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))
    union_area = box_area + boxes_area - inter_area

    return np.divide(inter_area, union_area, out=np.zeros_like(inter_area), where=union_area > 0)


def safe_norm(value: float, high: float) -> float:
    if high <= 0:
        return 0.0
    return float(np.clip(value / high, 0.0, 1.0))


# =============================================================================
# FEATURE COMPUTATION AND SCORING
# =============================================================================

def compute_track_stats(rows: pd.DataFrame, fps: float) -> pd.DataFrame:
    stats = []
    for track_id, g in rows.groupby("unique-id"):
        g = g.sort_values("frame-count")
        first = g.iloc[0]

        duration_frames = int(g["frame-count"].max() - g["frame-count"].min() + 1)
        duration_s = duration_frames / fps

        head = g.head(min(5, len(g)))
        tail = g.tail(min(5, len(g)))

        x_start = float(head["x-center"].mean())
        x_end = float(tail["x-center"].mean())
        y_start = float(head["y-center"].mean())
        y_end = float(tail["y-center"].mean())
        area_start = float(head["area"].mean())
        area_end = float(tail["area"].mean())

        area_growth = (area_end - area_start) / max(area_start, 1e-9)
        x_motion = abs(x_end - x_start)
        y_motion = abs(y_end - y_start)

        in_roi = (
            (g["x-center"] >= ROI_X_MIN)
            & (g["x-center"] <= ROI_X_MAX)
            & (g["y-center"] >= ROI_Y_MIN)
            & (g["y-center"] <= ROI_Y_MAX)
        )

        stats.append(
            {
                "unique_id": int(track_id),
                "class_id": int(first["yolo-id"]),
                "first_frame": int(g["frame-count"].min()),
                "last_frame": int(g["frame-count"].max()),
                "duration_s": float(duration_s),
                "mean_confidence": float(g["confidence"].mean()),
                "x_start": x_start,
                "x_end": x_end,
                "y_start": y_start,
                "y_end": y_end,
                "x_motion": x_motion,
                "y_motion": y_motion,
                "area_start": area_start,
                "area_end": area_end,
                "area_growth": area_growth,
                "roi_fraction": float(in_roi.mean()),
                "n_detections": int(len(g)),
            }
        )
    return pd.DataFrame(stats)


def estimate_occlusion_score(person_rows: pd.DataFrame, vehicle_rows: pd.DataFrame) -> float:
    if person_rows.empty:
        return 0.0

    frame_scores = []
    grouped_frames = list(person_rows.groupby("frame-count"))
    if OCCLUSION_FRAME_STEP > 1:
        grouped_frames = grouped_frames[::OCCLUSION_FRAME_STEP]

    for frame, p_frame in grouped_frames:
        v_frame = vehicle_rows[vehicle_rows["frame-count"] == frame]
        all_other = pd.concat([p_frame, v_frame], ignore_index=True)

        boxes_all = all_other[["x1", "y1", "x2", "y2"]].to_numpy(dtype=float)
        ids_all = all_other["unique-id"].to_numpy(dtype=int)

        if len(boxes_all) <= 1:
            frame_scores.append(0.0)
            continue

        overlaps = []
        for _, ped in p_frame.iterrows():
            ped_box = ped[["x1", "y1", "x2", "y2"]].to_numpy(dtype=float)
            mask = ids_all != int(ped["unique-id"])
            other_boxes = boxes_all[mask]
            if len(other_boxes) == 0:
                overlaps.append(0.0)
            else:
                overlaps.append(float((iou_one_to_many(ped_box, other_boxes) > 0.05).mean()))

        frame_scores.append(float(np.mean(overlaps)))

    return float(np.mean(frame_scores)) if frame_scores else 0.0


def pick_main_pedestrian(person_stats: pd.DataFrame) -> Optional[Dict[str, float]]:
    if person_stats.empty:
        return None

    candidates = person_stats[person_stats["duration_s"] >= MIN_PERSON_TRACK_SECONDS].copy()
    if candidates.empty:
        return None

    candidates["track_score"] = (
        1.50 * candidates["duration_s"].clip(upper=10) / 10
        + 2.00 * candidates["x_motion"].clip(upper=0.50) / 0.50
        + 1.50 * candidates["roi_fraction"]
        + 1.00 * candidates["area_growth"].clip(lower=0, upper=2) / 2
        + 0.50 * candidates["mean_confidence"].clip(lower=0, upper=1)
    )
    best = candidates.sort_values("track_score", ascending=False).iloc[0]
    return best.to_dict()


def score_window(window: pd.DataFrame, start_frame: int, end_frame: int, fps: float) -> Dict[str, object]:
    person_rows = window[window["yolo-id"].isin(PERSON_CLASS_IDS)].copy()
    vehicle_rows = window[window["yolo-id"].isin(VEHICLE_CLASS_IDS)].copy()

    person_stats = compute_track_stats(person_rows, fps) if not person_rows.empty else pd.DataFrame()
    main_ped = pick_main_pedestrian(person_stats)

    if main_ped is None:
        return {
            "start_frame_local": start_frame,
            "end_frame_local": end_frame,
            "candidate_start_time_local_s": round(start_frame / fps, 2),
            "candidate_end_time_local_s": round(end_frame / fps, 2),
            "duration_s": round((end_frame - start_frame + 1) / fps, 2),
            "quality_score": 0.0,
            "reject_reason": "no pedestrian track long enough",
        }

    frames = np.arange(start_frame, end_frame + 1)
    ped_density = person_rows.groupby("frame-count")["unique-id"].nunique().reindex(frames, fill_value=0)
    veh_density = vehicle_rows.groupby("frame-count")["unique-id"].nunique().reindex(frames, fill_value=0)

    avg_ped_density = float(ped_density.mean())
    avg_vehicle_density = float(veh_density.mean())
    vehicle_presence_fraction = float((veh_density > 0).mean())
    occlusion_score = estimate_occlusion_score(person_rows, vehicle_rows)

    quality_score = (
        WEIGHTS["track_duration"] * safe_norm(float(main_ped["duration_s"]), 10.0)
        + WEIGHTS["crossing_motion"] * safe_norm(float(main_ped["x_motion"]), 0.40)
        + WEIGHTS["central_presence"] * float(main_ped["roi_fraction"])
        + WEIGHTS["area_growth"] * safe_norm(max(float(main_ped["area_growth"]), 0.0), 1.50)
        + WEIGHTS["vehicle_presence"] * vehicle_presence_fraction
        + WEIGHTS["traffic_density"] * safe_norm(avg_vehicle_density + avg_ped_density, 8.0)
        + WEIGHTS["occlusion"] * occlusion_score
    )

    return {
        "start_frame_local": start_frame,
        "end_frame_local": end_frame,
        "candidate_start_time_local_s": round(start_frame / fps, 2),
        "candidate_end_time_local_s": round(end_frame / fps, 2),
        "duration_s": round((end_frame - start_frame + 1) / fps, 2),
        "quality_score": round(float(quality_score), 4),
        "main_pedestrian_track_id": int(main_ped["unique_id"]),
        "main_track_duration_s": round(float(main_ped["duration_s"]), 2),
        "main_x_motion": round(float(main_ped["x_motion"]), 4),
        "main_y_motion": round(float(main_ped["y_motion"]), 4),
        "main_area_growth_pct": round(float(main_ped["area_growth"] * 100), 2),
        "main_roi_fraction": round(float(main_ped["roi_fraction"]), 3),
        "main_mean_confidence": round(float(main_ped["mean_confidence"]), 3),
        "avg_pedestrian_density": round(avg_ped_density, 3),
        "avg_vehicle_density": round(avg_vehicle_density, 3),
        "vehicle_presence_fraction": round(vehicle_presence_fraction, 3),
        "occlusion_proxy": round(occlusion_score, 3),
        "reject_reason": "",
    }


def score_csv_file(csv_path: Path, mapping_lookup: Dict[Tuple[str, int],
                                                        Dict[str, object]]) -> List[Dict[str, object]]:
    parsed = parse_tracking_csv_name(csv_path)
    if parsed is None:
        return []

    video_id = str(parsed["video_id"])
    segment_start = int(parsed["segment_start_time_s"])
    fps = float(parsed["fps"])
    meta = mapping_lookup.get((video_id, segment_start), {})

    try:
        df = pd.read_csv(csv_path)
        df = ensure_columns(df)
        df = add_box_coordinates(df)
        df = df[df["confidence"] >= MIN_CONFIDENCE].copy()
    except Exception as e:
        logger.warning("Skipping %s due to read/format error: %s", csv_path.name, e)
        return []

    if df.empty:
        return []

    # Use the actual frame-count range present in the CSV as the authority.
    # The video FPS may be 29.97, 59.94, etc., and older filenames may also round
    # FPS. The safest unit for mining candidate windows is therefore the frame
    # index written by the tracker.
    max_frame = int(df["frame-count"].max())

    clip_frames = max(1, int(round(CLIP_SECONDS * fps)))
    stride_frames = max(1, int(round(STRIDE_SECONDS * fps)))

    if max_frame + 1 <= clip_frames:
        starts = [0]
    else:
        starts = list(range(0, max_frame - clip_frames + 2, stride_frames))

    if MAX_WINDOWS_PER_CSV is not None and len(starts) > MAX_WINDOWS_PER_CSV:
        sampled_indices = np.linspace(0, len(starts) - 1, MAX_WINDOWS_PER_CSV, dtype=int)
        starts = [starts[i] for i in sorted(set(sampled_indices.tolist()))]

    results = []
    for start_frame in starts:
        end_frame = min(start_frame + clip_frames - 1, max_frame)
        window = df[(df["frame-count"] >= start_frame) & (df["frame-count"] <= end_frame)].copy()
        if window.empty:
            continue

        score = score_window(window, start_frame, end_frame, fps)
        absolute_start = segment_start + float(score["candidate_start_time_local_s"])
        absolute_end = segment_start + float(score["candidate_end_time_local_s"])

        row = {
            "csv_file": csv_path.name,
            "csv_path": str(csv_path),
            "video_id": video_id,
            "fps": fps,
            "segment_start_time_s": segment_start,
            "segment_end_time_s": meta.get("segment_end_time_s", np.nan),
            "candidate_start_time_absolute_s": round(absolute_start, 2),
            "candidate_end_time_absolute_s": round(absolute_end, 2),
            "candidate_start_frame_local": int(start_frame),
            "candidate_end_frame_local": int(end_frame),
            "candidate_frame_count": int(end_frame - start_frame + 1),
            "locality": meta.get("locality", ""),
            "country": meta.get("country", ""),
            "iso3": meta.get("iso3", ""),
            "continent": meta.get("continent", ""),
            "time_of_day": meta.get("time_of_day", ""),
        }
        row.update(score)
        results.append(row)

    return results


# =============================================================================
# TIMELINE PLOTS
# =============================================================================

def build_trigger_timeline(csv_path: str, start_frame: int, end_frame: int, main_track_id: int) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = ensure_columns(df)
    df = add_box_coordinates(df)
    df = df[df["confidence"] >= MIN_CONFIDENCE].copy()
    df = df[(df["frame-count"] >= start_frame) & (df["frame-count"] <= end_frame)].copy()

    frames = list(range(start_frame, end_frame + 1))
    person_rows = df[df["yolo-id"].isin(PERSON_CLASS_IDS)].copy()
    vehicle_rows = df[df["yolo-id"].isin(VEHICLE_CLASS_IDS)].copy()
    main_rows = person_rows[person_rows["unique-id"] == main_track_id].copy()

    records = []
    previous_area = None
    previous_x = None

    for frame in frames:
        p = person_rows[person_rows["frame-count"] == frame]
        v = vehicle_rows[vehicle_rows["frame-count"] == frame]
        m = main_rows[main_rows["frame-count"] == frame]

        ped_count = p["unique-id"].nunique()
        vehicle_count = v["unique-id"].nunique()
        traffic_density = ped_count + vehicle_count

        if m.empty:
            main_visible = 0
            x_center = np.nan
            y_center = np.nan
            area = np.nan
            in_roi = 0
            area_growth_signal = 0.0
            crossing_motion_signal = 0.0
        else:
            row = m.iloc[0]
            main_visible = 1
            x_center = float(row["x-center"])
            y_center = float(row["y-center"])
            area = float(row["area"])
            in_roi = int(ROI_X_MIN <= x_center <= ROI_X_MAX and ROI_Y_MIN <= y_center <= ROI_Y_MAX)

            area_growth_signal = 0.0 if previous_area is None else (area - previous_area) / max(previous_area, 1e-9)
            crossing_motion_signal = 0.0 if previous_x is None else abs(x_center - previous_x)
            previous_area = area
            previous_x = x_center

        records.append(
            {
                "frame_local": frame,
                "main_visible": main_visible,
                "main_x_center": x_center,
                "main_y_center": y_center,
                "main_area": area,
                "pedestrian_count": int(ped_count),
                "vehicle_count": int(vehicle_count),
                "traffic_density": int(traffic_density),
                "in_interaction_roi": in_roi,
                "area_growth_signal": area_growth_signal,
                "crossing_motion_signal": crossing_motion_signal,
                "monitor_trigger": int(main_visible == 1),
                "interaction_trigger": int(main_visible == 1 and in_roi == 1),
                "clutter_trigger": int(traffic_density >= 5),
                "risk_proxy_trigger": int(main_visible == 1 and in_roi == 1 and vehicle_count > 0),
            }
        )

    return pd.DataFrame(records)


def plot_timeline(timeline: pd.DataFrame, output_path: Path, fps: float, title: str) -> None:
    if timeline.empty:
        return

    timeline = timeline.copy()
    timeline["time_local_s"] = timeline["frame_local"] / fps
    max_density = max(float(timeline["traffic_density"].max()), 1.0)
    timeline["traffic_density_scaled"] = timeline["traffic_density"] / max_density

    fig = go.Figure()
    traces = [
        ("main_visible", "main pedestrian visible"),
        ("in_interaction_roi", "in interaction ROI"),
        ("risk_proxy_trigger", "risk proxy trigger"),
        ("clutter_trigger", "clutter trigger"),
        ("traffic_density_scaled", "traffic density scaled"),
    ]

    for y_col, name in traces:
        fig.add_trace(
            go.Scatter(
                x=timeline["time_local_s"],
                y=timeline[y_col],
                mode="lines",
                name=name,
                customdata=timeline[["pedestrian_count", "vehicle_count", "traffic_density"]],
                hovertemplate=(
                    "local time=%{x:.2f}s<br>"
                    f"{name}=%{{y:.2f}}<br>"
                    "pedestrians=%{customdata[0]}<br>"
                    "vehicles=%{customdata[1]}<br>"
                    "total objects=%{customdata[2]}"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Local time inside tracked segment (s)",
        yaxis_title="Signal value",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=50, r=30, t=95, b=50),
    )
    fig.write_html(output_path, include_plotlyjs="cdn", full_html=True)


# =============================================================================
# MAIN RUN
# =============================================================================

def main() -> None:
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    mapping_lookup = load_mapping_lookup(MAPPING_CSV)
    csv_files = find_tracking_csvs(mapping_lookup)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {TRACKING_CSV_DIR}")

    all_results: List[Dict[str, object]] = []
    for csv_path in tqdm(csv_files, desc="Scoring tracking CSVs"):
        all_results.extend(score_csv_file(csv_path, mapping_lookup))

    if not all_results:
        raise ValueError("No candidate windows could be scored.")

    ranked = pd.DataFrame(all_results)
    ranked = ranked.sort_values("quality_score", ascending=False).reset_index(drop=True)
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))

    ranked_path = output_dir / "ranked_clip_candidates.csv"
    ranked.to_csv(ranked_path, index=False)

    top = ranked[ranked["quality_score"] > 0].head(TOP_K_OUTPUTS).copy()
    top_path = output_dir / "top_clip_candidates.csv"
    top.to_csv(top_path, index=False)

    clips_dir = output_dir / "extracted_clips"

    # Cache video resolution within one run. This prevents repeated downloads when
    # several top-ranked snippets come from the same source video.
    resolved_video_cache: Dict[str, Tuple[str, bool]] = {}
    downloaded_video_success: Dict[str, bool] = {}

    for _, row in top.iterrows():
        rank = int(row["rank"])
        fps = float(row["fps"])
        video_id = str(row["video_id"])
        start_frame = int(row["start_frame_local"])
        end_frame = int(row["end_frame_local"])
        main_track_id = int(row["main_pedestrian_track_id"])
        csv_path = str(row["csv_path"])

        timeline = build_trigger_timeline(csv_path, start_frame, end_frame, main_track_id)
        timeline["time_absolute_s"] = float(row["segment_start_time_s"]) + (timeline["frame_local"] / fps)

        timeline_csv = output_dir / f"timeline_rank_{rank:03d}.csv"
        timeline_html = output_dir / f"timeline_rank_{rank:03d}.html"
        timeline.to_csv(timeline_csv, index=False)

        title = (
            f"Rank {rank}: {video_id} | "
            f"abs {row['candidate_start_time_absolute_s']}s to {row['candidate_end_time_absolute_s']}s | "
            f"{row.get('locality', '')}, {row.get('country', '')}"
        )
        plot_timeline(timeline, timeline_html, fps, title)

        if EXTRACT_TOP_CLIPS:
            if video_id not in resolved_video_cache:
                resolved = resolve_video_path(video_id)
                if resolved is None:
                    logger.warning("Video not found for %s. Timeline saved, but MP4 not extracted.", video_id)
                    continue
                resolved_video_cache[video_id] = resolved

            source_video, downloaded_by_this_script = resolved_video_cache[video_id]
            if downloaded_by_this_script:
                downloaded_video_success.setdefault(source_video, True)

            out_name = (
                f"candidate_rank_{rank:03d}_{video_id}_"
                f"{float(row['candidate_start_time_absolute_s']):.2f}s_to_"
                f"{float(row['candidate_end_time_absolute_s']):.2f}s.mp4"
            )
            out_path = clips_dir / out_name
            extracted_ok = extract_clip_by_frame_count(
                source_video=source_video,
                segment_start_time_s=float(row["segment_start_time_s"]),
                start_frame_local=int(row["start_frame_local"]),
                end_frame_local=int(row["end_frame_local"]),
                output_path=out_path,
            )

            if not extracted_ok:
                # Fallback for environments where OpenCV cannot read/write the video.
                # This is time based and may be slightly less exact for 29.97 FPS videos.
                extracted_ok = extract_clip_with_ffmpeg(
                    source_video=source_video,
                    start_time_s=float(row["candidate_start_time_absolute_s"]),
                    duration_s=float(row["duration_s"]),
                    output_path=out_path,
                )

            if downloaded_by_this_script and not extracted_ok:
                downloaded_video_success[source_video] = False

    for downloaded_video, extraction_succeeded in downloaded_video_success.items():
        delete_downloaded_video_if_requested(downloaded_video, extraction_succeeded)

    print("Done.")
    print(f"Ranked candidates: {ranked_path}")
    print(f"Top candidates: {top_path}")
    print(f"Timelines: {output_dir.resolve()}")
    if EXTRACT_TOP_CLIPS:
        print(f"Extracted clips: {clips_dir.resolve()}")
    print("\nTop candidates:")
    display_cols = [
        "rank",
        "video_id",
        "locality",
        "country",
        "segment_start_time_s",
        "candidate_start_time_absolute_s",
        "candidate_end_time_absolute_s",
        "quality_score",
        "main_pedestrian_track_id",
        "main_track_duration_s",
        "main_x_motion",
        "main_area_growth_pct",
        "avg_vehicle_density",
        "occlusion_proxy",
    ]
    print(ranked.head(10)[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
