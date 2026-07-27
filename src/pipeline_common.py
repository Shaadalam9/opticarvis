r"""Shared helpers and per-clip configuration for the OptiCarVis workflow.

Intentionally separate from common.py (which belongs to the mobility study and
pulls in pycountry / email / a custom logger). This module has no third-party
dependencies so every workflow stage can import it cheaply.

Clip selection can be overridden with environment variables so rendering a
second clip does not require editing every module:
    OPTICARVIS_PROJECT_ROOT, OPTICARVIS_VIDEO_ID, OPTICARVIS_SEGMENT_START_S
"""

import json
import os
import shutil
import subprocess


PROJECT_ROOT = os.environ.get(
    "OPTICARVIS_PROJECT_ROOT", "C:/Users/localadmin/Desktop/Shadab"
)
VIDEO_ID = os.environ.get("OPTICARVIS_VIDEO_ID", "TuCsyBF3nHU")
SEGMENT_START_TIME_S = float(os.environ.get("OPTICARVIS_SEGMENT_START_S", "4630.0"))

WORKFLOW_OUTPUTS = PROJECT_ROOT + "/workflow_outputs"

# Delivery encode. cv2.VideoWriter can only emit mp4v/FMP4 here, which many players
# and browsers will not play, so every render is transcoded to H.264 in-process
# (previously an undocumented manual ffmpeg step sat between the code and the
# shipped artefact, which meant no invocation actually reproduced a deliverable).
H264_CRF = "20"


def segment_tag():
    """Filename stem shared by every per-clip artefact, e.g. TuCsyBF3nHU_4630."""
    return VIDEO_ID + "_" + str(int(SEGMENT_START_TIME_S))


def workflow_path(*parts):
    """Join parts under PROJECT_ROOT/workflow_outputs with forward slashes."""
    return "/".join([WORKFLOW_OUTPUTS, *parts])


def ensure_dir(path):
    """Create a directory (and parents) if it does not already exist."""
    if not os.path.isdir(path):
        os.makedirs(path)


def read_json(path, label="input JSON"):
    """Load JSON, exiting with a clear message if the file is missing."""
    if not os.path.isfile(path):
        print("Missing " + label + ":")
        print(path)
        raise SystemExit(1)
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, payload):
    """Write ``payload`` as pretty-printed JSON."""
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def clamp(value, low, high):
    """Clamp ``value`` into the inclusive range [low, high]."""
    return max(low, min(high, value))


def clip_stem(clip_path):
    """Basename of a clip without extension, e.g. .../TuCsyBF3nHU_turn.mp4 -> TuCsyBF3nHU_turn.

    Output names are derived from the clip actually rendered, so a second clip
    cannot inherit (and overwrite) the first clip's filename.
    """
    return os.path.splitext(os.path.basename(clip_path))[0]


def ffmpeg_path():
    """Absolute path to ffmpeg, or None when it is not on PATH."""
    return shutil.which("ffmpeg")


def transcode_h264(source, destination, crf=H264_CRF, remove_source=True):
    """Transcode ``source`` to H.264/yuv420p at ``destination``.

    Returns the destination path on success, or None (leaving the source in place)
    when ffmpeg is unavailable or fails, so the caller can still report where the
    playable-but-unencoded master is.
    """
    binary = ffmpeg_path()
    if binary is None:
        print("WARNING: ffmpeg not found on PATH; keeping the mp4v master:")
        print("  " + source)
        return None

    command = [
        binary, "-y", "-loglevel", "error",
        "-i", source,
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(crf),
        destination,
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0 or not os.path.isfile(destination):
        print("WARNING: ffmpeg failed (exit %d); keeping the mp4v master:" % completed.returncode)
        if completed.stderr:
            print(completed.stderr.strip()[:600])
        print("  " + source)
        return None

    if remove_source and os.path.abspath(source) != os.path.abspath(destination):
        os.remove(source)
    return destination
