r"""Batch entry point for the 100 city OptiCarVis analysis.

This version assumes the consolidated OptiCarVis folder layout:

    opticarvis/
        src/
        videos/
        mapping.csv
        alpamayo_outputs/
        workflow_outputs/
        external/
            alpamayo/
            oom-free-alpamayo/
            UFLDv2/

It uses the existing OptiCarVis credential mechanism:

    VIDEO_BASE_URL   = common.get_configs("VIDEO_BASE_URL")

SOURCE_VIDEO_DIR = common.get_configs("videos")
    VIDEO_USERNAME   = common.get_secrets("ftp_username")
    VIDEO_PASSWORD   = common.get_secrets("ftp_password")

It runs Alpamayo through:

    external/oom-free-alpamayo/scripts/infer_crowd_clip.py --clips

Usage:
    python batch_corrected_pipeline.py
    python batch_corrected_pipeline.py 20
    python batch_corrected_pipeline.py 20 100

The second form runs only 20 jobs.
The third form starts at zero based job index 100 and runs 20 jobs.
"""

import cv2
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

SRC_DIR = os.path.dirname(os.path.abspath(__file__))

from pipeline_common import (  # noqa: E402
    PROJECT_ROOT,
    WORKFLOW_OUTPUTS,
    ALPAMAYO_OUTPUTS,
    ALPAMAYO_JSON_DIR,
    ALPAMAYO_MODEL,
    OOM_FREE_ALPAMAYO_REPO,
    VIDEOS_DIR,
    alpamayo_extra_args,
    alpamayo_python,
    ensure_dir,
    append_jsonl,
    ffmpeg_path,
    normalise_path,
    transcode_h264,
)

# common.py and the config files are in the main opticarvis folder.
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import common  # noqa: E402


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


CLIP_JOBS_JSONL = normalise_path(
    os.environ.get(
        "OPTICARVIS_CLIP_JOBS",
        os.path.join(WORKFLOW_OUTPUTS, "clip_jobs.jsonl"),
    )
)

MASTER_INDEX_JSONL = normalise_path(
    os.environ.get(
        "OPTICARVIS_MASTER_CLIP_INDEX",
        os.path.join(WORKFLOW_OUTPUTS, "master_clip_index.jsonl"),
    )
)

PIPELINE_SCRIPT = normalise_path(
    os.environ.get(
        "OPTICARVIS_SINGLE_PIPELINE",
        os.path.join(SRC_DIR, "run_corrected_pipeline.py"),
    )
)

OOM_ALPAMAYO_REPO = normalise_path(
    os.environ.get(
        "OPTICARVIS_OOM_FREE_ALPAMAYO_REPO",
        os.environ.get("OPTICARVIS_ALPAMAYO_REPO", OOM_FREE_ALPAMAYO_REPO),
    )
)

ALPAMAYO_SCRIPT = normalise_path(
    os.environ.get(
        "OPTICARVIS_ALPAMAYO_SCRIPT",
        os.path.join(OOM_ALPAMAYO_REPO, "scripts", "infer_crowd_clip.py"),
    )
)

ALPAMAYO_CONFIG = os.environ.get("OPTICARVIS_ALPAMAYO_CONFIG", "config_5080_16gb.json")

EXTRACT_CLIPS = os.environ.get("OPTICARVIS_EXTRACT_CLIPS", "1") == "1"
RUN_ALPAMAYO = os.environ.get("OPTICARVIS_RUN_ALPAMAYO", "1") == "1"
SKIP_EXISTING_STATE = os.environ.get("OPTICARVIS_SKIP_EXISTING_STATE", "0") == "1"

# Default off: a 100 city batch should record a failed clip and carry on. Set it
# while debugging a single job, when the first traceback is the thing you want.
STOP_ON_JOB_FAILURE = os.environ.get("OPTICARVIS_STOP_ON_JOB_FAILURE", "0") == "1"

# Failure isolation cuts the other way when the cause is systemic -- a broken
# venv fails every job identically, two minutes at a time. A run of consecutive
# failures with nothing succeeding in between is that signature, so the batch
# stops there. 0 disables the guard.
MAX_CONSECUTIVE_FAILURES = int(
    os.environ.get("OPTICARVIS_MAX_CONSECUTIVE_FAILURES", "5")
)

# Decide the Gemma gate for a whole round in one process (gemma_gate_batch.py)
# instead of paying the full model load inside every per-job pipeline. 0 falls
# back to the per-job gate.
GATE_BATCH = os.environ.get("OPTICARVIS_GATE_BATCH", "1") == "1"

# Replace each rendered mp4v master with an H.264 encode of itself, same
# filename, so the state's output paths stay valid. The batch render path never
# transcoded; only the manual render_timeline_clip.py did.
BATCH_H264 = os.environ.get("OPTICARVIS_BATCH_H264", "1") == "1"

VIDEO_EXTENSIONS = [".mp4", ".mkv", ".mov", ".avi"]
DOWNLOAD_MISSING_SOURCE_VIDEOS = True


def reset_ftp_video_tmp_dir():
    if os.path.isdir(FTP_VIDEO_TMP_DIR):
        shutil.rmtree(FTP_VIDEO_TMP_DIR)

    os.makedirs(FTP_VIDEO_TMP_DIR, exist_ok=True)


def atomic_video_download_path(filename_with_ext):
    final_path = os.path.join(SOURCE_VIDEO_DIR, filename_with_ext)
    tmp_path = os.path.join(FTP_VIDEO_TMP_DIR, filename_with_ext + ".part")

    return final_path, tmp_path


def config_bool(key, default=False):
    """Read a boolean value from common.get_configs()."""
    value = common.get_configs(key)

    if value is None:
        return default

    if isinstance(value, bool):
        return value

    text_value = str(value).strip().lower()

    if text_value in ["1", "true", "yes", "y", "on"]:
        return True

    if text_value in ["0", "false", "no", "n", "off"]:
        return False

    return default


DELETE_FTP_VIDEOS_AFTER_USE = config_bool(
    "DELETE_FTP_VIDEOS_AFTER_USE",
    False,
)

FTP_ALIASES = ["tue1", "tue2", "tue3", "tue4", "tue5"]
FTP_CRAWL_PAGE_LIMIT = 500
FTP_TIMEOUT_SECONDS = 20
WHEN_START_LOCAL_S = 12.67
WHEN_END_LOCAL_S = 15.60

SOURCE_VIDEO_CACHE = {}
MISSING_SOURCE_VIDEO_IDS = set()

# Sources pulled from the file server, eligible for deletion under
# DELETE_FTP_VIDEOS_AFTER_USE once no unresolved city still needs them. A video
# already on disk that we did not download is never deleted. The registry is a
# file beside the videos, not just process memory: the prefetch script and an
# aborted batch both download in other processes, and memory-only tracking made
# their files permanently ineligible for cleanup.
DOWNLOADED_SOURCE_VIDEOS = set()


def downloaded_registry_path():
    return normalise_path(os.path.join(SOURCE_VIDEO_DIR, ".downloaded_by_opticarvis"))


def register_downloaded_source(local_path):
    DOWNLOADED_SOURCE_VIDEOS.add(local_path)

    try:
        with open(downloaded_registry_path(), "a", encoding="utf-8") as handle:
            handle.write(local_path + "\n")
    except OSError:
        pass


def downloaded_sources_on_record():
    """In-process downloads plus the registry left by other processes."""
    recorded = set(DOWNLOADED_SOURCE_VIDEOS)

    try:
        with open(downloaded_registry_path(), "r", encoding="utf-8") as handle:
            for line in handle:
                path = line.strip()

                if path:
                    recorded.add(path)
    except OSError:
        pass

    return recorded


def forget_downloaded_source(local_path):
    DOWNLOADED_SOURCE_VIDEOS.discard(local_path)
    remaining = downloaded_sources_on_record()
    remaining.discard(local_path)

    try:
        with open(downloaded_registry_path(), "w", encoding="utf-8") as handle:
            for path in sorted(remaining):
                handle.write(path + "\n")
    except OSError:
        pass


# Aliases so far have carried every video on one host (tue5); trying the last
# successful alias first saves four 404 round-trips per video.
PREFERRED_FTP_ALIAS = [None]


def ftp_aliases_in_preference_order():
    preferred = PREFERRED_FTP_ALIAS[0]

    if preferred in FTP_ALIASES:
        return [preferred] + [alias for alias in FTP_ALIASES if alias != preferred]

    return list(FTP_ALIASES)


def as_project_path(path_value):
    """Resolve relative config paths from the main opticarvis folder."""
    if not path_value:
        return path_value

    path_text = str(path_value)

    if os.path.isabs(path_text):
        return normalise_path(path_text)

    return normalise_path(os.path.join(PROJECT_ROOT, path_text))


configured_video_dir = common.get_configs("videos")

SOURCE_VIDEO_DIR = normalise_path(
    os.environ.get(
        "OPTICARVIS_SOURCE_VIDEO_DIR",
        as_project_path(configured_video_dir) if configured_video_dir else VIDEOS_DIR,
    )
)
FTP_VIDEO_TMP_DIR = os.path.join(SOURCE_VIDEO_DIR, ".tmp")

VIDEO_BASE_URL = common.get_configs("VIDEO_BASE_URL")
VIDEO_USERNAME = common.get_secrets("ftp_username")
VIDEO_PASSWORD = common.get_secrets("ftp_password")


def clip_tag(job):
    return job["video_id"] + "_" + str(int(float(job["segment_start_time_s"])))


def clip_length_tag(job):
    value = int(round(float(job.get("clip_length_s", 30.0))))
    return str(value) + "s"


def source_video_path_for(job):
    return normalise_path(os.path.join(SOURCE_VIDEO_DIR, job["video_id"] + ".mp4"))


def clip_video_path_for(job):
    return normalise_path(
        os.path.join(
            ALPAMAYO_OUTPUTS,
            "crowd_clips",
            clip_tag(job) + "_" + clip_length_tag(job) + ".mp4",
        )
    )


def expected_alpamayo_json_path_for(job):
    return normalise_path(os.path.join(ALPAMAYO_JSON_DIR, clip_tag(job) + "_alpamayo.json"))


def state_json_for(job):
    return normalise_path(os.path.join(WORKFLOW_OUTPUTS, clip_tag(job) + "_workflow_state.json"))


def job_produced_render(job):
    """True when the job's state records a final preview video that exists.

    The renderer only runs when the explanation gate says yes, so the state's
    outputs are what distinguishes a rendered clip from a declined one -- the
    per-clip pipeline exits 0 either way.
    """
    state_json = state_json_for(job)

    if not os.path.isfile(state_json):
        return False

    try:
        with open(state_json, "r", encoding="utf-8") as handle:
            outputs = json.load(handle).get("outputs", {})
    except (ValueError, OSError):
        return False

    for key, path in outputs.items():
        if "video" in key and path and os.path.isfile(str(path)):
            return True

    return False


def alpamayo_raw_json_path(job):
    return normalise_path(
        os.path.join(
            ALPAMAYO_JSON_DIR,
            "alpamayo_inference_output_" + clip_tag(job) + ".json",
        )
    )


def normalise_job_paths(job):
    """Force each job to use the consolidated opticarvis folder layout."""
    job = dict(job)
    job["clip_length_s"] = float(job.get("clip_length_s", 30.0))
    job["segment_start_time_s"] = float(job["segment_start_time_s"])
    job["source_video"] = source_video_path_for(job)
    job["clip_video"] = clip_video_path_for(job)
    job["alpamayo_json"] = expected_alpamayo_json_path_for(job)
    return job


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
                jobs.append(normalise_job_paths(json.loads(line)))

    return jobs


def get_video_resolution_label(local_path):
    try:
        pass
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
        pass
    except ImportError:
        return 0.0

    cap = cv2.VideoCapture(local_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()

    if fps > 0:
        return fps

    return 0.0


def save_video_response(response, filename_with_ext, source_url):
    final_path, tmp_path = atomic_video_download_path(filename_with_ext)

    os.makedirs(SOURCE_VIDEO_DIR, exist_ok=True)
    os.makedirs(FTP_VIDEO_TMP_DIR, exist_ok=True)

    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    total_bytes = int(response.headers.get("content-length", "0") or "0")

    progress = tqdm(
        total=total_bytes if total_bytes > 0 else None,
        unit="B",
        unit_scale=True,
        desc="Downloading source video " + filename_with_ext,
    )

    with open(tmp_path, "wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if not chunk:
                continue

            handle.write(chunk)
            progress.update(len(chunk))

    progress.close()

    if not os.path.exists(tmp_path) or os.path.getsize(tmp_path) == 0:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

        logger.warning("Downloaded source video was empty: %s", filename_with_ext)
        return None

    os.replace(tmp_path, final_path)

    file_size = os.path.getsize(final_path)

    capture = cv2.VideoCapture(final_path)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture.release()

    logger.info(
        "Downloaded source video from %s to %s | bytes=%s | resolution=%sx%s | fps=%.3f",
        source_url,
        final_path,
        file_size,
        width,
        height,
        fps,
    )

    return final_path


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

    if video_id in MISSING_SOURCE_VIDEO_IDS:
        return None

    if video_id in SOURCE_VIDEO_CACHE:
        return SOURCE_VIDEO_CACHE[video_id]

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

    for alias in ftp_aliases_in_preference_order():
        direct_url = urljoin(base_url, "v/" + alias + "/files/" + filename_with_ext)
        response = fetch_url(session, direct_url, stream=True)

        if response is not None:
            local_path = save_video_response(response, filename_with_ext, direct_url)

            # save_video_response returns None for an empty body (an error page
            # served with 200, a truncated transfer). Registering that None
            # poisons the cache and crashes the cleanup sweep; try elsewhere.
            if local_path:
                PREFERRED_FTP_ALIAS[0] = alias
                SOURCE_VIDEO_CACHE[video_id] = local_path
                register_downloaded_source(local_path)
                return local_path

    for alias in ftp_aliases_in_preference_order():
        browse_url = urljoin(base_url, "v/" + alias + "/browse")
        found_url = crawl_for_video_url(session, filename_with_ext, browse_url)

        if found_url is None:
            continue

        response = fetch_url(session, found_url, stream=True)

        if response is not None:
            local_path = save_video_response(response, filename_with_ext, found_url)

            if local_path:
                PREFERRED_FTP_ALIAS[0] = alias
                SOURCE_VIDEO_CACHE[video_id] = local_path
                register_downloaded_source(local_path)
                return local_path

    logger.warning("Source video %s was not found on the FTP server", filename_with_ext)
    MISSING_SOURCE_VIDEO_IDS.add(video_id)

    return None


def find_source_video(video_id, preferred_path):
    if preferred_path and os.path.isfile(preferred_path):
        return normalise_path(preferred_path)

    for ext in VIDEO_EXTENSIONS:
        path = os.path.join(SOURCE_VIDEO_DIR, video_id + ext)

        if os.path.isfile(path):
            return normalise_path(path)

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


def config_text_value(key, default=""):
    value = common.get_configs(key)

    if value is None:
        return default

    text_value = str(value).strip()

    if not text_value:
        return default

    return text_value


def get_alpamayo_backend():
    backend = config_text_value("ALPAMAYO_BACKEND", "alpamayo_r1").lower()

    aliases = {
        "old": "alpamayo_r1",
        "r1": "alpamayo_r1",
        "oom_free": "alpamayo_r1",
        "oom-free": "alpamayo_r1",
        "alpamayo": "alpamayo_r1",
        "alpamayo2": "alpamayo2_super",
        "alpamayo2-super": "alpamayo2_super",
        "alpamayo2_super": "alpamayo2_super",
    }

    return aliases.get(backend, backend)


def alpamayo2_super_output_dir_from_jobs(ready_jobs):
    for job in ready_jobs:
        alpamayo_json = str(job.get("alpamayo_json", "")).strip()

        if alpamayo_json:
            return os.path.dirname(alpamayo_json)

    return os.path.join(PROJECT_ROOT, "alpamayo_outputs", "alpamayo_json")


def write_alpamayo2_super_ready_jobs(ready_jobs):
    output_path = os.path.join(
        WORKFLOW_OUTPUTS,
        "alpamayo2_super_ready_jobs.jsonl",
    )

    with open(output_path, "w", encoding="utf-8") as handle:
        for job in ready_jobs:
            row = dict(job)

            # The clip jobs carry no t0, so inject the same window the R1
            # manifest writes. Without it the adapter would fall back to its own
            # default and the two backends would condition on different frames.
            row["when_start_local_s"] = WHEN_START_LOCAL_S
            row["when_end_local_s"] = WHEN_END_LOCAL_S

            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    return output_path


def run_alpamayo2_super_batch(ready_jobs, start_index=0):
    adapter_script = config_text_value("ALPAMAYO2_SUPER_ADAPTER_SCRIPT", "")

    if not adapter_script:
        raise RuntimeError(
            "ALPAMAYO_BACKEND is set to alpamayo2_super, but "
            "ALPAMAYO2_SUPER_ADAPTER_SCRIPT is not configured. "
            "Keep ALPAMAYO_BACKEND as alpamayo_r1 until the Alpamayo2 Super "
            "adapter script is implemented."
        )

    adapter_script = as_project_path(adapter_script)

    if not os.path.isfile(adapter_script):
        raise RuntimeError(
            "ALPAMAYO2_SUPER_ADAPTER_SCRIPT does not exist: " + adapter_script
        )

    model_id = config_text_value(
        "ALPAMAYO2_SUPER_MODEL_ID",
        "nvidia/Alpamayo2-Super",
    )

    alpamayo2_python = config_text_value("ALPAMAYO2_SUPER_PYTHON", sys.executable)
    output_dir = alpamayo2_super_output_dir_from_jobs(ready_jobs)
    ready_jobs_jsonl = write_alpamayo2_super_ready_jobs(ready_jobs)

    os.makedirs(output_dir, exist_ok=True)

    print()
    print("Running Alpamayo2 Super batch")
    print("=============================")
    print("jobs:", len(ready_jobs))
    print("model_id:", model_id)
    print("adapter_script:", adapter_script)
    print("ready_jobs_jsonl:", ready_jobs_jsonl)
    print("output_dir:", output_dir)
    print("interpreter:", alpamayo2_python)

    command = [
        alpamayo2_python,
        adapter_script,
        "--jobs-jsonl",
        ready_jobs_jsonl,
        "--output-dir",
        output_dir,
        "--model-id",
        model_id,
    ]

    command += alpamayo_extra_args()

    # check is deliberately off: the adapter exits non zero only when every clip
    # failed, and a partial batch should still render what it produced.
    completed = subprocess.run(command, cwd=PROJECT_ROOT)

    if completed.returncode != 0:
        print("")
        print("Alpamayo2 Super adapter returned non zero code:", completed.returncode)


def report_alpamayo_outputs(jobs, copy_from_raw):
    """Announce which jobs ended up with planner JSON at the expected path.

    The R1 backend writes alpamayo_inference_output_<tag>.json and needs the
    copy; the Alpamayo2 Super adapter writes job["alpamayo_json"] directly.
    """
    for job in jobs:
        ready = copy_alpamayo_json_to_expected_path(job) if copy_from_raw \
            else os.path.isfile(job["alpamayo_json"])

        if ready:
            print("Alpamayo JSON ready:", job["alpamayo_json"])
        elif copy_from_raw:
            print("Missing Alpamayo output after batch:", alpamayo_raw_json_path(job))
        else:
            print("Missing Alpamayo output after batch:", job["alpamayo_json"])


def run_alpamayo_r1_batch(jobs, start_index):
    ensure_dir(WORKFLOW_OUTPUTS)
    manifest_path = normalise_path(
        os.path.join(
            WORKFLOW_OUTPUTS,
            "alpamayo_manifest_" + str(start_index) + "_" + str(len(jobs)) + ".csv",
        )
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

    backend = get_alpamayo_backend()
    ensure_dir(ALPAMAYO_JSON_DIR)

    print("")
    print("Selected Alpamayo backend:", backend)

    if backend == "alpamayo2_super":
        run_alpamayo2_super_batch(missing_jobs, start_index)
        report_alpamayo_outputs(missing_jobs, copy_from_raw=False)
        return

    if backend != "alpamayo_r1":
        raise RuntimeError(
            "Unknown ALPAMAYO_BACKEND: "
            + backend
            + ". Use alpamayo_r1 or alpamayo2_super."
        )

    # Only the R1 backend shells out to a script on disk; checking this before
    # the dispatch used to skip the planner entirely whenever the unrelated
    # oom-free checkout was absent.
    if not os.path.isfile(ALPAMAYO_SCRIPT):
        print("")
        print("Missing Alpamayo script:")
        print(ALPAMAYO_SCRIPT)
        return

    manifest_path = run_alpamayo_r1_batch(missing_jobs, start_index)

    print("")
    print("Running Alpamayo batch")
    print("======================")
    print("clips:", len(missing_jobs))
    print("manifest:", manifest_path)
    print("output_dir:", ALPAMAYO_JSON_DIR)

    command = [
        alpamayo_python(),
        ALPAMAYO_SCRIPT,
        "--clips",
        manifest_path,
        "--output-dir",
        ALPAMAYO_JSON_DIR,
        "--config",
        ALPAMAYO_CONFIG,
    ]

    # Only passed when set, since a backend that does not accept --model would
    # reject the flag outright.
    if ALPAMAYO_MODEL:
        command += ["--model", ALPAMAYO_MODEL]

    command += alpamayo_extra_args()

    print("interpreter:", command[0])

    if ALPAMAYO_MODEL:
        print("model:", ALPAMAYO_MODEL)

    completed = subprocess.run(command, cwd=OOM_ALPAMAYO_REPO)

    if completed.returncode != 0:
        print("")
        print("Alpamayo batch returned non zero code:", completed.returncode)
        return

    report_alpamayo_outputs(missing_jobs, copy_from_raw=True)


def job_environment(job):
    env = os.environ.copy()

    env["OPTICARVIS_PROJECT_ROOT"] = PROJECT_ROOT
    env["OPTICARVIS_WORKFLOW_OUTPUTS"] = WORKFLOW_OUTPUTS
    env["OPTICARVIS_ALPAMAYO_OUTPUTS"] = ALPAMAYO_OUTPUTS
    env["OPTICARVIS_ALPAMAYO_JSON_DIR"] = ALPAMAYO_JSON_DIR
    env["OPTICARVIS_VIDEOS_DIR"] = SOURCE_VIDEO_DIR
    env["OPTICARVIS_OOM_FREE_ALPAMAYO_REPO"] = OOM_ALPAMAYO_REPO
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
        # An intentional skip, not a failure: counting it as one made a fully
        # skipped re-run exit non zero and trip the systemic-failure guard.
        return "skipped"

    ok, message = extract_clip(job)

    if not ok:
        elapsed = time.time() - started
        append_master(job, "skipped_" + message, message, elapsed)
        print("Skipped:", message)
        return "failed"

    print("Clip ready:", message)
    return "ready"


def run_pipeline_one_job(job, index, total, gate_decision=None):
    """Render one prepared job. gate_decision: True/False when the batched gate
    already decided (state file holds it); None to let the per-job pipeline
    decide as before."""
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
        return "failed"

    # With a batched gate, the decision already sits in the state file, so a
    # declined clip costs nothing further and an approved one skips stages 1-2.
    if gate_decision is False:
        elapsed = time.time() - started
        append_master(job, "complete", "gate_declined", elapsed)
        print("No render: the explanation gate declined this clip.")
        return "gate_declined"

    env = job_environment(job)

    if gate_decision is True:
        env["OPTICARVIS_GATE_PRECOMPUTED"] = "1"

    completed = subprocess.run(
        [sys.executable, PIPELINE_SCRIPT],
        cwd=SRC_DIR,
        env=env,
    )

    elapsed = time.time() - started

    if completed.returncode != 0:
        append_master(job, "failed_pipeline", "return_code_" + str(completed.returncode), elapsed)
        print("FAILED:", job["job_id"], "return code", completed.returncode)

        # A single bad clip must not discard the rest of the batch. Failing the
        # whole run here meant one unrenderable city threw away every city after
        # it, after hours of GPU time -- and clips do fail for ordinary reasons:
        # visual odometry cannot recover ego motion in dense traffic, so the
        # planner refuses the clip rather than feed the model a stopped-car
        # history. The failure is recorded in the master index; main() reports
        # the tally and exits non zero so a caller still notices.
        if STOP_ON_JOB_FAILURE:
            print("Stopping: OPTICARVIS_STOP_ON_JOB_FAILURE is set")
            raise SystemExit(completed.returncode)

        return "failed"

    # Exit code 0 covers two different outcomes: a rendered clip, and a clip the
    # explanation gate declined. Both are correct, but counting a declined clip
    # as rendered overstates what a batch produced, so they are separated here.
    if job_produced_render(job):
        if BATCH_H264:
            transcode_job_renders(job)

        append_master(job, "complete", "pipeline_complete", elapsed)
        return "rendered"

    append_master(job, "complete", "gate_declined", elapsed)
    print("No render: the explanation gate declined this clip.")

    return "gate_declined"


def transcode_job_renders(job):
    """Replace each mp4v master with an H.264 encode under the same filename.

    Same filename on purpose: the state's output paths stay valid, and a re-run
    that finds the file simply skips the job. A failed transcode keeps the
    master -- a worse codec is not a failed render.
    """
    state_json = state_json_for(job)

    try:
        with open(state_json, "r", encoding="utf-8") as handle:
            outputs = json.load(handle).get("outputs", {})
    except (ValueError, OSError):
        return

    for key, path in outputs.items():
        if "video" not in key or not path or not os.path.isfile(str(path)):
            continue

        path = str(path)
        temp_path = path + ".h264.tmp.mp4"

        # A run interrupted mid-transcode leaves the partial behind; clear it
        # rather than letting ffmpeg append confusion to it.
        if os.path.isfile(temp_path):
            os.remove(temp_path)

        try:
            transcode_h264(path, temp_path, remove_source=False)
            os.replace(temp_path, path)
            print("Transcoded to H.264:", path)
        except Exception as error:
            print("H.264 transcode failed for %s (%s); keeping the mp4v master."
                  % (path, type(error).__name__))

            if os.path.isfile(temp_path):
                os.remove(temp_path)


def gate_decisions_for_jobs(jobs, start_index):
    """Decide the gate for every job in one gemma_gate_batch.py process.

    Returns {job_id: True/False/None}; None means no decision was produced and
    the job should count as failed. Stale state files are removed first so a
    decision can only ever come from this run.
    """
    decisions = {}

    if not jobs:
        return decisions

    for job in jobs:
        state_json = state_json_for(job)

        if os.path.isfile(state_json):
            os.remove(state_json)

    jobs_path = normalise_path(
        os.path.join(
            WORKFLOW_OUTPUTS,
            "gate_batch_jobs_" + str(start_index) + ".jsonl",
        )
    )

    with open(jobs_path, "w", encoding="utf-8") as handle:
        for job in jobs:
            handle.write(json.dumps(job, ensure_ascii=False) + "\n")

    command = [
        sys.executable,
        os.path.join(SRC_DIR, "gemma_gate_batch.py"),
        "--jobs-jsonl",
        jobs_path,
    ]

    # No check: decisions are read from the state files, and a driver that died
    # partway leaves the undecided jobs as None.
    subprocess.run(command, cwd=SRC_DIR)

    for job in jobs:
        state_json = state_json_for(job)
        decision = None

        if os.path.isfile(state_json):
            try:
                with open(state_json, "r", encoding="utf-8") as handle:
                    explanation = json.load(handle).get("explanation")
            except (ValueError, OSError):
                explanation = None

            # Stage 1 alone writes explanation.needed = None ("pending"), so a
            # gate that crashed between the stages leaves a state file whose
            # needed is None -- undecided, not declined. bool(None) would have
            # silently recorded every such crash as a legitimate decline.
            if (
                isinstance(explanation, dict)
                and explanation.get("needed") in (True, False)
                and explanation.get("status") in ("explain_now", "do_not_explain")
            ):
                decision = bool(explanation["needed"])

        decisions[job["job_id"]] = decision

    return decisions


def group_jobs_by_city(jobs):
    """Ordered [(city_key, [jobs...])], each city's jobs in window order.

    Jobs from a builder without window_index sort stably at 0, so a one-job
    city behaves exactly as before: a single round.
    """
    groups = {}
    order = []

    for job in jobs:
        key = job.get("city_index", job.get("job_id"))

        if key not in groups:
            groups[key] = []
            order.append(key)

        groups[key].append(job)

    for key in order:
        groups[key].sort(key=lambda item: item.get("window_index", 0))

    return [(key, groups[key]) for key in order]


def source_video_references(jobs):
    """video path -> set of city keys whose unresolved windows still need it."""
    references = {}

    for job in jobs:
        path = normalise_path(str(job.get("source_video", "")))

        if path:
            references.setdefault(path, set()).add(
                job.get("city_index", job.get("job_id"))
            )

    return references


def cleanup_resolved_sources(city_results, source_refs):
    """Delete downloaded sources no unresolved city still references.

    Only videos on the download record are eligible -- this process's own plus
    the registry written by the prefetch script or an earlier aborted run.
    Anything else on disk is the user's, not ours to reclaim.
    """
    if not DELETE_FTP_VIDEOS_AFTER_USE:
        return

    recorded = downloaded_sources_on_record()

    for path, cities in list(source_refs.items()):
        if path not in recorded:
            continue

        if all(city in city_results for city in cities) and os.path.isfile(path):
            size_gib = os.path.getsize(path) / (1024 ** 3)
            os.remove(path)
            forget_downloaded_source(path)
            del source_refs[path]
            print("Deleted downloaded source (%.2f GiB, no unresolved city needs it): %s"
                  % (size_gib, path))


def cleanup_downloaded_sources_final(referenced_sources):
    """End-of-run sweep for recorded downloads this batch no longer needs.

    referenced_sources: sources jobs OUTSIDE this invocation's slice may still
    need (main.py chunking, start_index) -- those stay, and stay on record so a
    later invocation reclaims them.
    """
    if not DELETE_FTP_VIDEOS_AFTER_USE:
        return

    for path in sorted(downloaded_sources_on_record()):
        if path in referenced_sources:
            continue

        if os.path.isfile(path):
            size_gib = os.path.getsize(path) / (1024 ** 3)
            os.remove(path)
            print("Deleted downloaded source (%.2f GiB): %s" % (size_gib, path))

        forget_downloaded_source(path)


def main():
    reset_ftp_video_tmp_dir()
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
    print("project_root:", PROJECT_ROOT)
    print("source_video_dir:", SOURCE_VIDEO_DIR)
    print("workflow_outputs:", WORKFLOW_OUTPUTS)
    print("alpamayo_outputs:", ALPAMAYO_OUTPUTS)
    print("video_base_url:", "configured" if VIDEO_BASE_URL else "missing")
    print("ftp_credentials:", "configured" if VIDEO_USERNAME and VIDEO_PASSWORD else "missing")
    print("ftp_aliases:", ", ".join(FTP_ALIASES))
    print("alpamayo_repo:", OOM_ALPAMAYO_REPO)
    print("alpamayo_script:", ALPAMAYO_SCRIPT)
    print("alpamayo_config:", ALPAMAYO_CONFIG)
    print("alpamayo_json_dir:", ALPAMAYO_JSON_DIR)

    city_groups = group_jobs_by_city(selected)
    source_refs = source_video_references(selected)
    max_rounds = max((len(group) for _, group in city_groups), default=0)

    # Sources that jobs OUTSIDE this slice reference must survive the cleanup
    # sweep: main.py chunking and start_index can split the job list, and a
    # chunk boundary can even split one city's windows -- warn about that,
    # since the later invocation cannot know this one already rendered the city
    # except through the render-credit check below.
    selected_ids = {id(job) for job in selected}
    outside_sources = {
        normalise_path(str(job.get("source_video", "")))
        for job in jobs
        if id(job) not in selected_ids and job.get("source_video")
    }

    for city_key, group in city_groups:
        total_windows = sum(
            1 for job in jobs
            if job.get("city_index", job.get("job_id")) == city_key
        )

        if total_windows > len(group):
            print("")
            print("WARNING: city %s has %d window(s) outside this run's slice; "
                  "run the whole city in one invocation to avoid duplicate or "
                  "missed renders." % (city_key, total_windows - len(group)))

    # How many rendered clips resolve a city. 0 keeps the old uncapped meaning:
    # never resolve early, attempt every emitted window.
    clips_wanted = max(int(os.environ.get("OPTICARVIS_CLIPS_PER_CITY", "1")), 0)

    outcomes = {"rendered": [], "gate_declined": [], "failed": [], "skipped": []}
    city_results = {}
    city_render_counts = {}
    consecutive_failures = [0]
    attempted = 0

    def note_failure(job_id):
        outcomes["failed"].append(job_id)
        consecutive_failures[0] += 1

        if MAX_CONSECUTIVE_FAILURES and consecutive_failures[0] >= MAX_CONSECUTIVE_FAILURES:
            print("")
            print("%d consecutive failures with no success in between -- this "
                  "looks systemic, not per-clip. Stopping. Raise or disable "
                  "OPTICARVIS_MAX_CONSECUTIVE_FAILURES to override."
                  % consecutive_failures[0])
            raise SystemExit(2)

    def credit_render(city_key):
        city_render_counts[city_key] = city_render_counts.get(city_key, 0) + 1

        if clips_wanted and city_render_counts[city_key] >= clips_wanted:
            city_results[city_key] = "rendered"

    # A window rendered by an earlier invocation counts towards the city's
    # target now, so a resumed or re-sliced run does not render the city again.
    for city_key, group in city_groups:
        for job in group:
            if job_produced_render(job):
                print("Existing render found for %s; crediting %s"
                      % (job["job_id"], city_key))
                credit_render(city_key)

    for round_index in range(max_rounds):
        round_jobs = [
            (city_key, group[round_index])
            for city_key, group in city_groups
            if city_key not in city_results and round_index < len(group)
        ]

        if not round_jobs:
            break

        if max_rounds > 1:
            print("")
            print("#" * 80)
            print("Window round %d: %d unresolved city(ies)"
                  % (round_index + 1, len(round_jobs)))
            print("#" * 80)

        prepared = []

        for city_key, job in round_jobs:
            attempted += 1
            status = prepare_one_job(job, attempted - 1, len(selected))

            if status == "ready":
                prepared.append((city_key, job))
            elif status == "skipped":
                outcomes["skipped"].append(job["job_id"])
            else:
                note_failure(job["job_id"])

        run_alpamayo_for_ready_jobs([job for _key, job in prepared], start_index)

        decisions = (
            gate_decisions_for_jobs(
                [job for _key, job in prepared
                 if os.path.isfile(job["alpamayo_json"])],
                start_index,
            )
            if GATE_BATCH
            else {}
        )

        round_base = attempted - len(prepared)

        for prepared_index, (city_key, job) in enumerate(prepared):
            if GATE_BATCH and decisions.get(job["job_id"]) is None:
                # The driver produced no decision for this job (planner JSON
                # missing, or the gate crashed on it). Falling back to the full
                # per-job pipeline here would quietly reintroduce the per-clip
                # model load the batch exists to avoid.
                print("")
                print("No gate decision for %s; counting it as failed."
                      % job["job_id"])
                append_master(job, "failed_gate", "no_gate_decision", 0.0)
                note_failure(job["job_id"])
                continue

            result = run_pipeline_one_job(
                job,
                round_base + prepared_index,
                len(selected),
                gate_decision=decisions.get(job["job_id"]) if GATE_BATCH else None,
            )

            if result == "rendered":
                outcomes["rendered"].append(job["job_id"])
                credit_render(city_key)
                consecutive_failures[0] = 0
            elif result == "gate_declined":
                outcomes["gate_declined"].append(job["job_id"])
                consecutive_failures[0] = 0
            else:
                note_failure(job["job_id"])

        # A city whose windows are exhausted is finished too. With at least one
        # render it still counts as rendered -- it just fell short of a >1
        # target; only a city with none is reported as unfired.
        for city_key, group in city_groups:
            if city_key not in city_results and len(group) <= round_index + 1:
                city_results[city_key] = (
                    "rendered" if city_render_counts.get(city_key) else "no_window_fired"
                )

        cleanup_resolved_sources(city_results, source_refs)

    cleanup_downloaded_sources_final(outside_sources)

    cities_rendered = sum(1 for value in city_results.values() if value == "rendered")
    cities_unfired = sum(1 for value in city_results.values() if value == "no_window_fired")

    print("")
    print("Batch complete.")
    print("cities: %d rendered, %d with no approved window, of %d"
          % (cities_rendered, cities_unfired, len(city_groups)))
    print("windows: %d attempted | rendered: %d | gate declined: %d | failed: %d "
          "| skipped: %d"
          % (attempted, len(outcomes["rendered"]),
             len(outcomes["gate_declined"]), len(outcomes["failed"]),
             len(outcomes["skipped"])))

    # Declined is a result, not a fault: the gate deciding no explanation is
    # needed is the pipeline working. Failed is listed separately so it is not
    # buried in the same count.
    for label, job_ids in (("Gate declined (no render, working as intended)",
                            outcomes["gate_declined"]),
                           ("Failed", outcomes["failed"])):
        if job_ids:
            print("")
            print(label + ":")

            for job_id in job_ids:
                print("  " + job_id)

    print("Master index:", MASTER_INDEX_JSONL)

    # Non zero only when every genuinely attempted window outright failed --
    # intentional skips do not count. main.py drives the chunks with check=True,
    # so exiting non zero on a partial batch would abort every later chunk --
    # reintroducing at the chunk level exactly the failure this change removes
    # at the job level. A batch the gate declined in full is not an error.
    genuinely_attempted = attempted - len(outcomes["skipped"])

    if genuinely_attempted and len(outcomes["failed"]) == genuinely_attempted:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
