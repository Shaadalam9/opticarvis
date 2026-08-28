"""Shared OptiCarVis configuration, paths and small file helpers.

This module is intentionally OS independent:
- path construction uses os.path.join
- paths are normalised with os.path.normpath
- no local absolute paths are hardcoded
- environment variables can override all runtime paths
"""

import json
import os
import shutil
import subprocess


def env_text(name, default):
    value = os.environ.get(name)
    if value is None or str(value).strip() == "":
        return default
    return str(value)


def env_float(name, default):
    value = os.environ.get(name)
    if value is None or str(value).strip() == "":
        return float(default)
    return float(value)


def env_float_alias(names, default):
    """First non-empty of several env names, for variables that were renamed.

    A plain env_float on the wrong name fails silently -- the child keeps the
    default and every artefact it writes is misnamed -- so the alias list is
    resolved here rather than at each call site.
    """
    for name in names:
        value = os.environ.get(name)
        if value is not None and str(value).strip() != "":
            return float(value)

    return float(default)


def env_bool(name, default):
    value = os.environ.get(name)
    if value is None or str(value).strip() == "":
        return bool(default)

    text = str(value).strip().lower()
    return text in ("1", "true", "yes", "y", "on")


def normalise_path(path):
    """Return a native path for the current operating system."""
    if path is None:
        return None

    text = os.path.expandvars(os.path.expanduser(str(path)))
    return os.path.normpath(text)


def join_path(*parts):
    clean_parts = [str(part) for part in parts if part is not None and str(part) != ""]
    return normalise_path(os.path.join(*clean_parts))


def env_path(name, *default_parts):
    value = os.environ.get(name)
    if value is not None and str(value).strip() != "":
        return normalise_path(value)

    return join_path(*default_parts)


def number_tag(value):
    value = float(value)
    if value.is_integer():
        return str(int(value))
    text = ("%0.3f" % value).rstrip("0").rstrip(".")
    return text.replace(".", "p")


def clip_length_tag():
    return number_tag(CLIP_LENGTH_S) + "s"


def segment_tag():
    return VIDEO_ID + "_" + number_tag(SEGMENT_START_TIME_S)


def ensure_dir(path):
    path = normalise_path(path)
    os.makedirs(path, exist_ok=True)
    return path


def workflow_path(*parts):
    if not parts:
        return WORKFLOW_OUTPUTS
    return join_path(WORKFLOW_OUTPUTS, *parts)


def alpamayo_output_path(*parts):
    if not parts:
        return ALPAMAYO_OUTPUTS
    return join_path(ALPAMAYO_OUTPUTS, *parts)


def clip_stem(clip_path):
    return os.path.splitext(os.path.basename(str(clip_path)))[0]


def read_json(path, label="input JSON"):
    path = normalise_path(path)
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, payload):
    path = normalise_path(path)
    parent = os.path.dirname(path)
    if parent:
        ensure_dir(parent)

    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    return path


def read_jsonl(path):
    path = normalise_path(path)
    rows = []

    if not os.path.exists(path):
        return rows

    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))

    return rows


def append_jsonl(path, payload):
    path = normalise_path(path)
    parent = os.path.dirname(path)
    if parent:
        ensure_dir(parent)

    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    return path


def clamp(value, low, high):
    return max(low, min(high, value))


def ffmpeg_path():
    configured = os.environ.get("OPTICARVIS_FFMPEG")
    if configured is not None and str(configured).strip() != "":
        return normalise_path(configured)

    discovered = shutil.which("ffmpeg")
    if discovered:
        return normalise_path(discovered)

    return "ffmpeg"


def transcode_h264(source, destination, crf=None, remove_source=True):
    source = normalise_path(source)
    destination = normalise_path(destination)
    crf_value = H264_CRF if crf is None else crf

    parent = os.path.dirname(destination)
    if parent:
        ensure_dir(parent)

    command = [
        ffmpeg_path(),
        "-y",
        "-i",
        source,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        str(crf_value),
        "-preset",
        "medium",
        "-movflags",
        "+faststart",
        destination,
    ]

    subprocess.run(command, check=True)

    if remove_source and os.path.exists(source) and normalise_path(source) != normalise_path(destination):
        os.remove(source)

    return destination


# ---------------------------------------------------------------------------
# Core job settings
# ---------------------------------------------------------------------------

VIDEO_ID = env_text("OPTICARVIS_VIDEO_ID", "TuCsyBF3nHU")
# OPTICARVIS_SEGMENT_START_S is the canonical name: it is what the batch runner
# exports (batch_corrected_pipeline.job_environment) and what the README
# documents. OPTICARVIS_SEGMENT_START_TIME_S is the older name, still honoured so
# existing shells keep working. Reading only the old name meant the batch's value
# never arrived and every clip of a video reused the 4630 artefact names.
SEGMENT_START_TIME_S = env_float_alias(
    ("OPTICARVIS_SEGMENT_START_S", "OPTICARVIS_SEGMENT_START_TIME_S"),
    4630.0,
)
CLIP_LENGTH_S = env_float("OPTICARVIS_CLIP_LENGTH_S", 30.0)

JOB_ID = env_text("OPTICARVIS_JOB_ID", "manual_" + segment_tag())
LOCALITY = env_text("OPTICARVIS_LOCALITY", "unknown")
COUNTRY = env_text("OPTICARVIS_COUNTRY", "unknown")
CONTINENT = env_text("OPTICARVIS_CONTINENT", "unknown")

H264_CRF = int(env_float("OPTICARVIS_H264_CRF", 20))

GEMMA4_MODEL = env_text("OPTICARVIS_GEMMA4_MODEL", "google/gemma-4-E2B-it")
CANDIDATE_SEMANTIC_MODEL = env_text(
    "OPTICARVIS_CANDIDATE_SEMANTIC_MODEL",
    "google/siglip2-base-patch16-224",
)
HF_LOCAL_FILES_ONLY = env_bool("OPTICARVIS_HF_LOCAL_FILES_ONLY", True)


# ---------------------------------------------------------------------------
# Project root and runtime paths
# ---------------------------------------------------------------------------

SRC_DIR = normalise_path(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PROJECT_ROOT = normalise_path(os.path.dirname(SRC_DIR))

PROJECT_ROOT = env_path("OPTICARVIS_PROJECT_ROOT", DEFAULT_PROJECT_ROOT)
OPTICARVIS_DIR = PROJECT_ROOT

VIDEOS_DIR = env_path("OPTICARVIS_VIDEOS_DIR", PROJECT_ROOT, "videos")
WORKFLOW_OUTPUTS = env_path("OPTICARVIS_WORKFLOW_OUTPUTS", PROJECT_ROOT, "workflow_outputs")
ALPAMAYO_OUTPUTS = env_path("OPTICARVIS_ALPAMAYO_OUTPUTS", PROJECT_ROOT, "alpamayo_outputs")

CROWD_CLIPS_DIR = env_path("OPTICARVIS_CROWD_CLIPS_DIR", ALPAMAYO_OUTPUTS, "crowd_clips")
ALPAMAYO_JSON_DIR = env_path("OPTICARVIS_ALPAMAYO_JSON_DIR", ALPAMAYO_OUTPUTS, "alpamayo_json")

EXTERNAL_DIR = env_path("OPTICARVIS_EXTERNAL_DIR", PROJECT_ROOT, "external")
ALPAMAYO_REPO = env_path("OPTICARVIS_ALPAMAYO_REPO", EXTERNAL_DIR, "alpamayo")
OOM_FREE_ALPAMAYO_REPO = env_path(
    "OPTICARVIS_OOM_FREE_ALPAMAYO_REPO",
    EXTERNAL_DIR,
    "oom-free-alpamayo",
)
UFLDV2_DIR = env_path("OPTICARVIS_UFLDV2_DIR", EXTERNAL_DIR, "UFLDv2")

MAPPING_CSV = env_path("OPTICARVIS_MAPPING_CSV", PROJECT_ROOT, "mapping.csv")
SOURCE_VIDEO = env_path("OPTICARVIS_SOURCE_VIDEO", VIDEOS_DIR, VIDEO_ID + ".mp4")

CLIP_VIDEO = env_path(
    "OPTICARVIS_CLIP_VIDEO",
    CROWD_CLIPS_DIR,
    segment_tag() + "_" + clip_length_tag() + ".mp4",
)

ALPAMAYO_JSON = env_path(
    "OPTICARVIS_ALPAMAYO_JSON",
    ALPAMAYO_JSON_DIR,
    segment_tag() + "_alpamayo.json",
)

STATE_JSON = env_path(
    "OPTICARVIS_STATE_JSON",
    WORKFLOW_OUTPUTS,
    segment_tag() + "_workflow_state.json",
)


def current_job_summary():
    return {
        "job_id": JOB_ID,
        "video_id": VIDEO_ID,
        "segment_start_time_s": SEGMENT_START_TIME_S,
        "clip_length_s": CLIP_LENGTH_S,
        "locality": LOCALITY,
        "country": COUNTRY,
        "continent": CONTINENT,
        "project_root": PROJECT_ROOT,
        "opticarvis_dir": OPTICARVIS_DIR,
        "videos_dir": VIDEOS_DIR,
        "workflow_outputs": WORKFLOW_OUTPUTS,
        "alpamayo_outputs": ALPAMAYO_OUTPUTS,
        "external_dir": EXTERNAL_DIR,
        "alpamayo_repo": ALPAMAYO_REPO,
        "oom_free_alpamayo_repo": OOM_FREE_ALPAMAYO_REPO,
        "ufldv2_dir": UFLDV2_DIR,
        "mapping_csv": MAPPING_CSV,
        "source_video": SOURCE_VIDEO,
        "clip_video": CLIP_VIDEO,
        "alpamayo_json": ALPAMAYO_JSON,
        "state_json": STATE_JSON,
        "gemma4_model": GEMMA4_MODEL,
        "candidate_semantic_model": CANDIDATE_SEMANTIC_MODEL,
        "hf_local_files_only": HF_LOCAL_FILES_ONLY,
    }


# Backward compatible Alpamayo runtime constants.
# These are used by batch_corrected_pipeline.py.
ALPAMAYO_MODEL = os.environ.get("OPTICARVIS_ALPAMAYO_MODEL", "").strip()

ALPAMAYO_CONFIG = os.environ.get(
    "OPTICARVIS_ALPAMAYO_CONFIG",
    "config_5080_16gb.json",
)

ALPAMAYO_CONFIG_PATH = normalise_path(
    os.environ.get(
        "OPTICARVIS_ALPAMAYO_CONFIG_PATH",
        os.path.join(OOM_FREE_ALPAMAYO_REPO, ALPAMAYO_CONFIG),
    )
)


def alpamayo_extra_args():
    """Return optional extra command line arguments for Alpamayo."""
    import shlex

    raw_args = os.environ.get("OPTICARVIS_ALPAMAYO_EXTRA_ARGS", "").strip()

    if not raw_args:
        return []

    return shlex.split(raw_args)


def alpamayo_python():
    """Return the Python interpreter used to run Alpamayo."""
    import sys

    return normalise_path(
        os.environ.get("OPTICARVIS_ALPAMAYO_PYTHON", sys.executable)
    )


# Semantic segmentation runtime constants.
# Kept as compatibility names for semantic_segmentation_module.py.
YOLO_SEG_MODEL = os.environ.get(
    "OPTICARVIS_YOLO_SEG_MODEL",
    "yolo26x-seg.pt",
)

YOLO_SEG_IMAGE_SIZE = int(
    float(os.environ.get("OPTICARVIS_YOLO_SEG_IMAGE_SIZE", "1280"))
)

YOLO_SEG_CONFIDENCE = float(
    os.environ.get("OPTICARVIS_YOLO_SEG_CONFIDENCE", "0.25")
)

# Scene model compatibility constants.
# These are used by scene_models.py.
DEPTH_MODEL = os.environ.get(
    "OPTICARVIS_DEPTH_MODEL",
    "depth-anything/Depth-Anything-V2-Small-hf",
)

ROAD_SEG_MODEL = os.environ.get(
    "OPTICARVIS_ROAD_SEG_MODEL",
    "nvidia/segformer-b0-finetuned-cityscapes-1024-1024",
)
