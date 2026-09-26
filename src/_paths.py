"""Put src/ and each of its module groups on sys.path.

The pipeline is organised into folders by stage (core, candidates, selection,
batch, gate, perception, trajectory, render), but its modules keep importing
each other by bare name (``from pipeline_common import ...``) and are still
launched as plain scripts (``python src/render/final_preview_renderer.py``).
Every such script starts with::

    sys.path.insert(0, <its parent folder, i.e. src/>)
    import _paths

and this module makes every group importable from there. ``script_path`` finds
a pipeline script by file name, so subprocess launches do not hardcode which
folder a stage lives in.
"""

import os
import sys


SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)

GROUPS = (
    "core",
    "candidates",
    "selection",
    "batch",
    "gate",
    "perception",
    "trajectory",
    "render",
)

for _path in [os.path.join(SRC_DIR, group) for group in reversed(GROUPS)] + [SRC_DIR]:
    if _path not in sys.path:
        sys.path.insert(0, _path)


def script_path(file_name):
    """Absolute path of a pipeline script (e.g. "run_corrected_pipeline.py")."""
    for group in GROUPS:
        candidate = os.path.join(SRC_DIR, group, file_name)
        if os.path.isfile(candidate):
            return candidate

    raise FileNotFoundError("No pipeline script named %s under %s" % (file_name, SRC_DIR))
