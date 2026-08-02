r"""Batch entry point for the 100 city OptiCarVis analysis.

This version uses the same FTP credential mechanism as the earlier OptiCarVis
scripts and runs Alpamayo through infer_crowd_clip.py --clips:

    VIDEO_BASE_URL  = common.get_configs("VIDEO_BASE_URL")
    SOURCE_VIDEO_DIR = common.get_configs("videos")
    VIDEO_USERNAME = common.get_secrets("ftp_username")
    VIDEO_PASSWORD = common.get_secrets("ftp_password")

It does not require setting FTP credentials manually in the terminal.

Usage:
    python batch_corrected_pipeline.py
    python batch_corrected_pipeline.py 20
    python batch_corrected_pipeline.py 20 100

The second form runs only 20 jobs.
The third form starts at job index 100 and runs 20 jobs.
"""

import csv
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# common.py and the config files are in the parent opticarvis folder.
# This file is inside opticarvis/src, so add the parent folder before importing common.
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
OPTICARVIS_DIR = os.path.dirname(SRC_DIR)

if OPTICARVIS_DIR not in sys.path:
    sys.path.insert(0, OPTICARVIS_DIR)

import common

from pipeline_common import (
    PROJECT_ROOT,
    WORKFLOW_OUTPUTS,
    ensure_dir,
    append_jsonl,
    ffmpeg_path,
)


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


CLIP_JOBS_JSONL = os.environ.get(
    "OPTICARVIS_CLIP_JOBS",
    WORKFLOW_OUTPUTS + "/clip_jobs.jsonl",
)

MASTER_INDEX_JSONL = os.environ.get(
    "OPTICARVIS_MASTER_CLIP_INDEX",
    WORKFLOW_OUTPUTS + "/master_clip_index.jsonl",
)

PIPELINE_SCRIPT = os.environ.get(
    "OPTICARVIS_SINGLE_PIPELINE",
    SRC_DIR + "/run_corrected_pipeline.py",
)

ALPAMAYO_REPO = os.environ.get(
    "OPTICARVIS_ALPAMAYO_REPO",
    PROJECT_ROOT + "/oom-free-alpamayo",
)

ALPAMAYO_SCRIPT = os.environ.get(
    "OPTICARVIS_ALPAMAYO_SCRIPT",
    ALPAMAYO_REPO + "/scripts/infer_crowd_clip.py",
)

ALPAMAYO_CONFIG = os.environ.get("OPTICARVIS_ALPAMAYO_CONFIG", "config_5080_16gb.json")
ALPAMAYO_JSON_DIR = os.environ.get(
    "OPTICARVIS_ALPAMAYO_JSON_DIR",
    PROJECT_ROOT + "/alpamayo_outputs/alpamayo_json",
)

EXTRACT_CLIPS = os.environ.get("OPTICARVIS_EXTRACT_CLIPS", "1") == "1"
RUN_ALPAMAYO = os.environ.get("OPTICARVIS_RUN_ALPAMAYO", "1") == "1"
SKIP_EXISTING_STATE = os.environ.get("OPTICARVIS_SKIP_EXISTING_STATE", "0") == "1"

VIDEO_EXTENSIONS = [".mp4", ".mkv", ".mov", ".avi"]
DOWNLOAD_MISSING_SOURCE_VIDEOS = True
FTP_ALIASES = ["tue4", "tue5"]
FTP_CRAWL_PAGE_LIMIT = 500
FTP_TIMEOUT_SECONDS = 20
WHEN_START_LOCAL_S = 12.67
WHEN_END_LOCAL_S = 15.60

def as_project_path(path_value):
    """Resolve relative config paths from the parent opticarvis folder."""
    if not path_value:
        return path_value

    path_text = str(path_value)

    if os.path.isabs(path_text):
        return path_text

    return os.path.abspath(os.path.join(OPTICARVIS_DIR, path_text)).replace("\\", "/")


SOURCE_VIDEO_DIR = as_project_path(common.get_configs("videos"))
VIDEO_BASE_URL = common.get_configs("VIDEO_BASE_URL")
VIDEO_USERNAME = common.get_secrets("ftp_username")
VIDEO_PASSWORD = common.get_secrets("ftp_password")


def read_jobs(path):
    if not os.path.isfile(path):
        print("Missing clip jobs file:")
        print(path)
        print("")
        print("Run first:")
        print("python clip_job_builder.py")
        raise SystemExit(1)

    jobs = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                jobs.append(json.loads(line))

    return jobs


def state_json_for(job):
    return (
        PROJECT_ROOT
        + "/workflow_outputs/"
        + job["video_id"]
        + "_"
        + str(int(float(job["segment_start_time_s"])))
        + "_workflow_state.json"
    )


def clip_tag(job):
    return job["video_id"] + "_" + str(int(float(job["segment_start_time_s"])))


def alpamayo_raw_json_path(job):
    return (
        ALPAMAYO_JSON_DIR
        + "/alpamayo_inference_output_"
        + clip_tag(job)
        + ".json"
    )


def get_video_resolution_label(local_path):
    try:
        import cv2
    except ImportError:
        return "unknown"

    cap = cv2.VideoCapture(local_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if width > 0 and height > 0:
        return str(width) + "x" + str(height)

    return "unknown"


def get_video_fps(local_path):
    try:
        import cv2
    except ImportError:
        return 0.0

    cap = cv2.VideoCapture(local_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()

    if fps > 0:
        return fps

    return 0.0


def save_video_response(response, filename_with_ext, source_url):
    os.makedirs(SOURCE_VIDEO_DIR, exist_ok=True)
    local_path = os.path.join(SOURCE_VIDEO_DIR, filename_with_ext)

    if os.path.isfile(local_path) and os.path.getsize(local_path) > 0:
        logger.info("Using already downloaded source video: %s", local_path)
        return local_path

    total_text = response.headers.get("content-length", "0")
    total = int(total_text) if str(total_text).isdigit() and int(total_text) > 0 else None
    written = 0

    with open(local_path, "wb") as file_handle, tqdm(
        total=total,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc="Downloading source video " + filename_with_ext,
    ) as bar:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                file_handle.write(chunk)
                written += len(chunk)
                if total:
                    bar.update(len(chunk))

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


def fetch_url(session, url, stream):
    response = session.get(url, timeout=FTP_TIMEOUT_SECONDS, stream=stream)
    logger.debug("GET %s -> %d", url, response.status_code)

    if response.status_code == 200:
        return response

    if response.status_code == 401:
        logger.warning("Authentication failed for %s", url)

    return None


def crawl_for_video_url(session, filename_with_ext, start_url):
    filename_lower = filename_with_ext.lower()
    stack = [start_url]
    visited = set()
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


def download_source_video_from_ftp(video_id):
    if not DOWNLOAD_MISSING_SOURCE_VIDEOS:
        return None

    if not VIDEO_BASE_URL:
        logger.warning("VIDEO_BASE_URL is missing in common.get_configs('VIDEO_BASE_URL')")
        return None

    filename_with_ext = video_id if video_id.lower().endswith(".mp4") else video_id + ".mp4"
    base_url = VIDEO_BASE_URL if VIDEO_BASE_URL.endswith("/") else VIDEO_BASE_URL + "/"

    logger.info("Source video missing locally. Trying FTP download for %s", filename_with_ext)

    session = requests.Session()

    if VIDEO_USERNAME and VIDEO_PASSWORD:
        session.auth = (VIDEO_USERNAME, VIDEO_PASSWORD)

    session.headers.update({"User-Agent": "opticarvis-batch-downloader/1.0"})

    for alias in FTP_ALIASES:
        direct_url = urljoin(base_url, "v/" + alias + "/files/" + filename_with_ext)
        response = fetch_url(session, direct_url, stream=True)

        if response is not None:
            return save_video_response(response, filename_with_ext, direct_url)

    for alias in FTP_ALIASES:
        browse_url = urljoin(base_url, "v/" + alias + "/browse")
        found_url = crawl_for_video_url(session, filename_with_ext, browse_url)

        if found_url is None:
            continue

        response = fetch_url(session, found_url, stream=True)

        if response is not None:
            return save_video_response(response, filename_with_ext, found_url)

    logger.warning("Source video %s was not found on the FTP server", filename_with_ext)

    return None


def find_source_video(video_id, preferred_path):
    if preferred_path and os.path.isfile(preferred_path):
        return preferred_path

    for ext in VIDEO_EXTENSIONS:
        path = os.path.join(SOURCE_VIDEO_DIR, video_id + ext)

        if os.path.isfile(path):
            return path

    return download_source_video_from_ftp(video_id)


def extract_clip(job):
    if os.path.isfile(job["clip_video"]):
        return True, "clip_already_exists"

    if not EXTRACT_CLIPS:
        return False, "clip_missing_extraction_disabled"

    source_video = find_source_video(job["video_id"], job.get("source_video", ""))

    if source_video is None or not os.path.isfile(source_video):
        return False, "missing_source_video"

    ffmpeg = ffmpeg_path()

    if ffmpeg is None:
        return False, "missing_ffmpeg"

    ensure_dir(os.path.dirname(job["clip_video"]))

    command = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-ss",
        str(job["segment_start_time_s"]),
        "-i",
        source_video,
        "-t",
        str(job["clip_length_s"]),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-an",
        job["clip_video"],
    ]

    completed = subprocess.run(command, capture_output=True, text=True)

    if completed.returncode != 0 or not os.path.isfile(job["clip_video"]):
        message = completed.stderr.strip() if completed.stderr else "ffmpeg_failed"
        return False, message[:500]

    job["source_video"] = source_video

    return True, "clip_extracted"


def write_alpamayo_manifest(jobs, start_index):
    ensure_dir(WORKFLOW_OUTPUTS)
    manifest_path = (
        WORKFLOW_OUTPUTS
        + "/alpamayo_manifest_"
        + str(start_index)
        + "_"
        + str(len(jobs))
        + ".csv"
    )

    with open(manifest_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "video_id",
                "segment_start_time_s",
                "clip_video",
                "when_start_local_s",
                "when_end_local_s",
            ],
        )
        writer.writeheader()

        for job in jobs:
            writer.writerow(
                {
                    "video_id": clip_tag(job),
                    "segment_start_time_s": job["segment_start_time_s"],
                    "clip_video": job["clip_video"],
                    "when_start_local_s": WHEN_START_LOCAL_S,
                    "when_end_local_s": WHEN_END_LOCAL_S,
                }
            )

    return manifest_path


def copy_alpamayo_json_to_expected_path(job):
    raw_json = alpamayo_raw_json_path(job)

    if not os.path.isfile(raw_json):
        return False

    ensure_dir(os.path.dirname(job["alpamayo_json"]))
    shutil.copyfile(raw_json, job["alpamayo_json"])

    return os.path.isfile(job["alpamayo_json"])


def run_alpamayo_for_ready_jobs(jobs, start_index):
    missing_jobs = [job for job in jobs if not os.path.isfile(job["alpamayo_json"])]

    if not missing_jobs:
        print("")
        print("Alpamayo JSON already exists for all ready jobs.")
        return

    if not RUN_ALPAMAYO:
        print("")
        print("Alpamayo run disabled by OPTICARVIS_RUN_ALPAMAYO=0")
        return

    if not os.path.isfile(ALPAMAYO_SCRIPT):
        print("")
        print("Missing Alpamayo script:")
        print(ALPAMAYO_SCRIPT)
        return

    ensure_dir(ALPAMAYO_JSON_DIR)
    manifest_path = write_alpamayo_manifest(missing_jobs, start_index)

    print("")
    print("Running Alpamayo batch")
    print("======================")
    print("clips:", len(missing_jobs))
    print("manifest:", manifest_path)
    print("output_dir:", ALPAMAYO_JSON_DIR)

    command = [
        sys.executable,
        ALPAMAYO_SCRIPT,
        "--clips",
        manifest_path,
        "--output-dir",
        ALPAMAYO_JSON_DIR,
        "--config",
        ALPAMAYO_CONFIG,
    ]

    completed = subprocess.run(command, cwd=ALPAMAYO_REPO)

    if completed.returncode != 0:
        print("")
        print("Alpamayo batch returned non zero code:", completed.returncode)
        return

    for job in missing_jobs:
        copied = copy_alpamayo_json_to_expected_path(job)

        if copied:
            print("Alpamayo JSON ready:", job["alpamayo_json"])
        else:
            print("Missing Alpamayo output after batch:", alpamayo_raw_json_path(job))


def job_environment(job):
    env = os.environ.copy()

    env["OPTICARVIS_PROJECT_ROOT"] = PROJECT_ROOT
    env["OPTICARVIS_JOB_ID"] = job["job_id"]
    env["OPTICARVIS_VIDEO_ID"] = job["video_id"]
    env["OPTICARVIS_SEGMENT_START_S"] = str(job["segment_start_time_s"])
    env["OPTICARVIS_CLIP_LENGTH_S"] = str(job["clip_length_s"])
    env["OPTICARVIS_SOURCE_VIDEO"] = job["source_video"]
    env["OPTICARVIS_CLIP_VIDEO"] = job["clip_video"]
    env["OPTICARVIS_ALPAMAYO_JSON"] = job["alpamayo_json"]
    env["OPTICARVIS_LOCALITY"] = job.get("locality", "")
    env["OPTICARVIS_COUNTRY"] = job.get("country", "")
    env["OPTICARVIS_CONTINENT"] = job.get("continent", "")

    return env


def load_state_summary(job):
    state_json = state_json_for(job)

    if not os.path.isfile(state_json):
        return {
            "state_json": state_json,
            "state_available": False,
        }

    with open(state_json, "r", encoding="utf-8") as handle:
        state = json.load(handle)

    explanation = state.get("explanation", {})
    decision = state.get("decision", {})
    outputs = state.get("outputs", {})

    return {
        "state_json": state_json,
        "state_available": True,
        "current_stage": state.get("current_stage", ""),
        "proper_time_to_explain": bool(explanation.get("needed", False)),
        "explanation_status": explanation.get("status", ""),
        "decision_reason": explanation.get("decision_reason", ""),
        "alpamayo_action": decision.get("alpamayo_action", ""),
        "scene_cause": decision.get("scene_cause", ""),
        "uncertainty_score": decision.get("uncertainty_score", None),
        "display_target": decision.get("display_plan", {}).get("display_target", ""),
        "gemma_gate_json": outputs.get("gemma_gate_json", ""),
        "segmentation_json": outputs.get("segmentation_json", ""),
        "depth_json": outputs.get("depth_json", ""),
        "mirage_effect_plan_json": outputs.get("mirage_effect_plan_json", ""),
        "rendered_video": (
            outputs.get("roadline_v3_final_preview_video")
            or outputs.get("clean_final_preview_video")
            or outputs.get("final_preview_video")
            or ""
        ),
    }


def append_master(job, status, message, elapsed_s):
    record = dict(job)
    record["batch_status"] = status
    record["batch_message"] = message
    record["elapsed_s"] = round(elapsed_s, 3)
    record.update(load_state_summary(job))
    append_jsonl(MASTER_INDEX_JSONL, record)


def prepare_one_job(job, index, total):
    print("")
    print("=" * 80)
    print("Preparing job %d/%d: %s" % (index + 1, total, job["job_id"]))
    print("=" * 80)

    started = time.time()
    existing_state = state_json_for(job)

    if SKIP_EXISTING_STATE and os.path.isfile(existing_state):
        elapsed = time.time() - started
        append_master(job, "skipped_existing_state", existing_state, elapsed)
        print("Skipped existing state:", existing_state)
        return False

    ok, message = extract_clip(job)

    if not ok:
        elapsed = time.time() - started
        append_master(job, "skipped_" + message, message, elapsed)
        print("Skipped:", message)
        return False

    print("Clip ready:", message)
    return True


def run_pipeline_one_job(job, index, total):
    print("")
    print("=" * 80)
    print("Pipeline job %d/%d: %s" % (index + 1, total, job["job_id"]))
    print("=" * 80)

    started = time.time()

    if not os.path.isfile(job["alpamayo_json"]):
        elapsed = time.time() - started
        append_master(job, "skipped_missing_alpamayo_json", "missing_alpamayo_json", elapsed)
        print("Skipped: missing_alpamayo_json")
        print("Expected Alpamayo JSON:", job["alpamayo_json"])
        return

    env = job_environment(job)

    completed = subprocess.run(
        [sys.executable, PIPELINE_SCRIPT],
        cwd=SRC_DIR,
        env=env,
    )

    elapsed = time.time() - started

    if completed.returncode != 0:
        append_master(job, "failed_pipeline", "return_code_" + str(completed.returncode), elapsed)
        raise SystemExit(completed.returncode)

    append_master(job, "complete", "pipeline_complete", elapsed)


def main():
    max_jobs = int(sys.argv[1]) if len(sys.argv) > 1 else None
    start_index = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    jobs = read_jobs(CLIP_JOBS_JSONL)

    selected = jobs[start_index:]

    if max_jobs is not None:
        selected = selected[:max_jobs]

    ensure_dir(os.path.dirname(MASTER_INDEX_JSONL))

    print("")
    print("OptiCarVis 100 city batch pipeline")
    print("==================================")
    print("jobs_file:", CLIP_JOBS_JSONL)
    print("master_index:", MASTER_INDEX_JSONL)
    print("available_jobs:", len(jobs))
    print("start_index:", start_index)
    print("jobs_to_run:", len(selected))
    print("extract_clips:", EXTRACT_CLIPS)
    print("run_alpamayo:", RUN_ALPAMAYO)
    print("opticarvis_dir:", OPTICARVIS_DIR)
    print("source_video_dir:", SOURCE_VIDEO_DIR)
    print("video_base_url:", "configured" if VIDEO_BASE_URL else "missing")
    print("ftp_credentials:", "configured" if VIDEO_USERNAME and VIDEO_PASSWORD else "missing")
    print("alpamayo_script:", ALPAMAYO_SCRIPT)
    print("alpamayo_config:", ALPAMAYO_CONFIG)
    print("alpamayo_json_dir:", ALPAMAYO_JSON_DIR)

    ready_jobs = []

    for offset, job in enumerate(selected):
        ready = prepare_one_job(job, start_index + offset, len(jobs))

        if ready:
            ready_jobs.append(job)

    run_alpamayo_for_ready_jobs(ready_jobs, start_index)

    for offset, job in enumerate(ready_jobs):
        run_pipeline_one_job(job, start_index + offset, len(jobs))

    print("")
    print("Batch complete.")
    print("Master index:", MASTER_INDEX_JSONL)


if __name__ == "__main__":
    main()