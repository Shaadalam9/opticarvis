r"""Shared helpers and per-clip configuration for the OptiCarVis workflow.

This file is the single source of truth for the current clip/job.
The batch runner overrides these values with environment variables, so you do
not need to edit every module for every city or every 30 second window.

Common environment variables:
    OPTICARVIS_PROJECT_ROOT
    OPTICARVIS_VIDEO_ID
    OPTICARVIS_SEGMENT_START_S
    OPTICARVIS_CLIP_LENGTH_S
    OPTICARVIS_CLIP_VIDEO
    OPTICARVIS_SOURCE_VIDEO
    OPTICARVIS_ALPAMAYO_JSON
    OPTICARVIS_JOB_ID
    OPTICARVIS_LOCALITY
    OPTICARVIS_COUNTRY
    OPTICARVIS_CONTINENT
"""

import json
import os
import shutil
import subprocess


PROJECT_ROOT = os.environ.get(
    "OPTICARVIS_PROJECT_ROOT",
    "C:/Users/localadmin/Desktop/Shadab",
)

VIDEO_ID = os.environ.get("OPTICARVIS_VIDEO_ID", "TuCsyBF3nHU")
SEGMENT_START_TIME_S = float(os.environ.get("OPTICARVIS_SEGMENT_START_S", "4630.0"))
CLIP_LENGTH_S = float(os.environ.get("OPTICARVIS_CLIP_LENGTH_S", "30.0"))

JOB_ID = os.environ.get(
    "OPTICARVIS_JOB_ID",
    VIDEO_ID + "_" + str(int(SEGMENT_START_TIME_S)),
)

LOCALITY = os.environ.get("OPTICARVIS_LOCALITY", "")
COUNTRY = os.environ.get("OPTICARVIS_COUNTRY", "")
CONTINENT = os.environ.get("OPTICARVIS_CONTINENT", "")

WORKFLOW_OUTPUTS = PROJECT_ROOT + "/workflow_outputs"
OPTICARVIS_DIR = PROJECT_ROOT + "/opticarvis"

MAPPING_CSV = os.environ.get(
    "OPTICARVIS_MAPPING_CSV",
    OPTICARVIS_DIR + "/mapping.csv",
)

SOURCE_VIDEO = os.environ.get(
    "OPTICARVIS_SOURCE_VIDEO",
    OPTICARVIS_DIR + "/videos/" + VIDEO_ID + ".mp4",
)


def clip_length_tag():
    value = int(round(CLIP_LENGTH_S))
    return str(value) + "s"


def segment_tag():
    """Filename stem shared by every per-clip artefact."""
    return VIDEO_ID + "_" + str(int(SEGMENT_START_TIME_S))


def workflow_path(*parts):
    """Join parts under PROJECT_ROOT/workflow_outputs with forward slashes."""
    return "/".join([WORKFLOW_OUTPUTS, *parts])


def clip_stem(clip_path):
    """Basename of a clip without extension."""
    return os.path.splitext(os.path.basename(clip_path))[0]


def ensure_dir(path):
    """Create a directory and parents if it does not already exist."""
    if path and not os.path.isdir(path):
        os.makedirs(path)


CLIP_VIDEO = os.environ.get(
    "OPTICARVIS_CLIP_VIDEO",
    workflow_path(
        "..",
        "alpamayo_outputs",
        "crowd_clips",
        segment_tag() + "_" + clip_length_tag() + ".mp4",
    ).replace("/workflow_outputs/../", "/"),
)

ALPAMAYO_JSON = os.environ.get(
    "OPTICARVIS_ALPAMAYO_JSON",
    workflow_path(
        "..",
        "alpamayo_outputs",
        "alpamayo_json",
        segment_tag() + "_alpamayo.json",
    ).replace("/workflow_outputs/../", "/"),
)

STATE_JSON = workflow_path(segment_tag() + "_workflow_state.json")


# Delivery encode. cv2.VideoWriter can only emit mp4v/FMP4 here, which many
# players and browsers will not play, so renders can be transcoded to H.264.
H264_CRF = "20"


def read_json(path, label="input JSON"):
    """Load JSON, exiting with a clear message if the file is missing."""
    if not path or not os.path.isfile(path):
        print("Missing " + label + ":")
        print(path)
        raise SystemExit(1)
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, payload):
    """Write payload as pretty printed JSON."""
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def read_jsonl(path):
    rows = []
    if not os.path.isfile(path):
        return rows
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path, payload):
    ensure_dir(os.path.dirname(path))
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def clamp(value, low, high):
    """Clamp value into the inclusive range [low, high]."""
    return max(low, min(high, value))


def ffmpeg_path():
    """Absolute path to ffmpeg, or None when it is not on PATH."""
    return shutil.which("ffmpeg")


def transcode_h264(source, destination, crf=H264_CRF, remove_source=True):
    """Transcode source to H.264/yuv420p at destination.

    Returns the destination path on success, or None when ffmpeg is unavailable
    or fails, so the caller can still report the original master file.
    """
    binary = ffmpeg_path()
    if binary is None:
        print("WARNING: ffmpeg not found on PATH; keeping the mp4v master:")
        print("  " + source)
        return None

    command = [
        binary,
        "-y",
        "-loglevel",
        "error",
        "-i",
        source,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        str(crf),
        destination,
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0 or not os.path.isfile(destination):
        print("WARNING: ffmpeg failed, keeping the mp4v master:")
        if completed.stderr:
            print(completed.stderr.strip()[:600])
        print("  " + source)
        return None

    if remove_source and os.path.abspath(source) != os.path.abspath(destination):
        os.remove(source)

    return destination


def current_job_summary():
    """Small metadata block that every module can store in its JSON output."""
    return {
        "job_id": JOB_ID,
        "video_id": VIDEO_ID,
        "segment_start_time_s": SEGMENT_START_TIME_S,
        "clip_length_s": CLIP_LENGTH_S,
        "locality": LOCALITY,
        "country": COUNTRY,
        "continent": CONTINENT,
        "source_video": SOURCE_VIDEO,
        "clip_video": CLIP_VIDEO,
        "alpamayo_json": ALPAMAYO_JSON,
        "state_json": STATE_JSON,
    }
