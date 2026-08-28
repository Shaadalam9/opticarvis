r"""Fetch or verify every model and checkout the pipeline needs.

A fresh clone renders, but degraded: the lane model is an optional download,
and without it the renderer silently falls back to a straight in-lane ribbon --
a whole batch can complete looking plausibly wrong. This script makes that
state loud, and fixes what it can:

    .venv/bin/python scripts/setup_assets.py              # download what is missing
    .venv/bin/python scripts/setup_assets.py --check-only # report, change nothing
    .venv/bin/python scripts/setup_assets.py --with-planner  # also the 67 GB planner

Exit code is 0 only when every REQUIRED item is present. scripts/run_100_cities.sh
runs the check before a batch and refuses to start on failures, so a machine
missing the lane model can no longer quietly produce 100 straight-ribbon videos.

The planner checkpoint (nvidia/Alpamayo2-Super, ~67 GB) and the planner venv
are reported but only downloaded with --with-planner: they are a deliberate,
disk-heavy decision, and building the venv needs platform-specific torch (see
AGENTS.md for the DGX Spark's cu129 pin).
"""

import argparse
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

UFLD_REPO_URL = "https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2.git"
UFLD_WEIGHTS_GDOWN_ID = "1AjnvAD3qmqt_dGPveZJsLZ1bOyWv62Yj"
UFLD_WEIGHTS_MIN_BYTES = 700 * 1024 * 1024  # a partial gdown leaves a small file

ALPAMAYO2_REPO_URL = "https://github.com/NVlabs/alpamayo2.git"

SRC_DIR = os.path.join(ROOT, "src")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from pipeline_common import CANDIDATE_SEMANTIC_MODEL  # noqa: E402


HF_MODELS = [
    ("nvidia/segformer-b0-finetuned-cityscapes-1024-1024", "road segmentation"),
    ("depth-anything/Depth-Anything-V2-Small-hf", "metric depth labels"),
    ("google/gemma-4-E2B-it", "explanation gate (9.6 GB)"),
    (CANDIDATE_SEMANTIC_MODEL, "offline candidate discovery"),
]

PLANNER_MODEL = "nvidia/Alpamayo2-Super"


class Report(object):
    def __init__(self):
        self.failed = 0
        self.warned = 0

    def ok(self, label, detail=""):
        print("  PASS  %-26s %s" % (label, detail))

    def warn(self, label, detail=""):
        self.warned += 1
        print("  WARN  %-26s %s" % (label, detail))

    def fail(self, label, detail=""):
        self.failed += 1
        print("  FAIL  %-26s %s" % (label, detail))


def hf_model_cached(model_id):
    """True when the model resolves from the local cache alone."""
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(model_id, local_files_only=True)
        return True
    except Exception:
        return False


def download_hf_model(model_id):
    from huggingface_hub import snapshot_download

    snapshot_download(model_id)


def check_ufld(report, download):
    repo = os.path.join(ROOT, "external", "UFLDv2")
    weights = os.path.join(repo, "culane_res34.pth")

    if not os.path.isdir(repo):
        if download:
            print("  ...   cloning UFLDv2")
            subprocess.run(["git", "clone", "--quiet", "--depth", "1",
                            UFLD_REPO_URL, repo], check=False)

        if not os.path.isdir(repo):
            report.fail("UFLDv2 checkout",
                        "missing -- the ribbon will stay STRAIGHT in every render. "
                        "git clone --depth 1 %s external/UFLDv2" % UFLD_REPO_URL)
            return

    report.ok("UFLDv2 checkout", repo)

    weights_present = (
        os.path.isfile(weights) and os.path.getsize(weights) >= UFLD_WEIGHTS_MIN_BYTES
    )

    if not weights_present and download:
        print("  ...   downloading culane_res34.pth (~830 MB)")

        try:
            import gdown

            gdown.download(id=UFLD_WEIGHTS_GDOWN_ID, output=weights, quiet=False)
        except Exception as error:
            print("  ...   gdown failed: %s %s" % (type(error).__name__, str(error)[:150]))

        weights_present = (
            os.path.isfile(weights) and os.path.getsize(weights) >= UFLD_WEIGHTS_MIN_BYTES
        )

    if weights_present:
        report.ok("UFLDv2 weights", "%.0f MB" % (os.path.getsize(weights) / 1024 / 1024))
    else:
        report.fail("UFLDv2 weights",
                    "missing -- lane centering and curve fit are OFF and every "
                    "ribbon renders STRAIGHT. python -c \"import gdown; "
                    "gdown.download(id='%s', output='external/UFLDv2/culane_res34.pth')\""
                    % UFLD_WEIGHTS_GDOWN_ID)


def check_hf_models(report, download):
    for model_id, purpose in HF_MODELS:
        if hf_model_cached(model_id):
            report.ok(model_id.split("/")[-1], purpose)
            continue

        if download:
            print("  ...   downloading %s" % model_id)

            try:
                download_hf_model(model_id)
            except Exception as error:
                print("  ...   download failed: %s %s"
                      % (type(error).__name__, str(error)[:150]))

        if hf_model_cached(model_id):
            report.ok(model_id.split("/")[-1], purpose)
        else:
            report.fail(model_id.split("/")[-1],
                        "not in the HF cache (%s). Set HF_HOME to the cache that "
                        "holds it, or let this script download it." % purpose)


def check_yolo(report, download):
    """ultralytics fetches its own checkpoint on first use; prefetch it so the
    first batch render does not stall on a download."""
    name = os.environ.get("OPTICARVIS_YOLO_SEG_MODEL", "yolo26x-seg.pt")

    if os.path.isabs(name) and os.path.isfile(name):
        report.ok("YOLO checkpoint", name)
        return

    try:
        from ultralytics.utils.downloads import attempt_download_asset

        path = attempt_download_asset(name) if download else name

        if os.path.isfile(str(path)):
            report.ok("YOLO checkpoint", str(path))
        else:
            report.warn("YOLO checkpoint",
                        "%s not present yet; ultralytics will fetch it on first use" % name)
    except Exception as error:
        report.warn("YOLO checkpoint", "could not verify (%s)" % type(error).__name__)


def check_planner(report, download_planner):
    repo = os.path.join(ROOT, "external", "alpamayo2")
    venv_python = os.path.join(repo, ".venv", "bin", "python")

    if os.path.isdir(repo):
        report.ok("alpamayo2 checkout", repo)
    elif download_planner:
        print("  ...   cloning NVlabs/alpamayo2")
        subprocess.run(["git", "clone", "--quiet", "--depth", "1",
                        ALPAMAYO2_REPO_URL, repo], check=False)

        if os.path.isdir(repo):
            report.ok("alpamayo2 checkout", repo)
        else:
            report.fail("alpamayo2 checkout", "clone failed")
    else:
        report.fail("alpamayo2 checkout",
                    "missing -- the planner cannot run. Re-run with --with-planner, "
                    "or: git clone --depth 1 %s external/alpamayo2" % ALPAMAYO2_REPO_URL)

    if os.path.isfile(venv_python):
        report.ok("planner venv", venv_python)
    else:
        report.fail("planner venv",
                    "missing -- build it per AGENTS.md (torch==2.8.0 needs the "
                    "cu129 index on aarch64; PyPI's wheel is CPU-only there; "
                    "skip flash-attn, it is not used)")

    if hf_model_cached(PLANNER_MODEL):
        report.ok("Alpamayo2-Super weights", "cached")
    elif download_planner:
        print("  ...   downloading %s (~67 GB, this takes a while)" % PLANNER_MODEL)

        try:
            download_hf_model(PLANNER_MODEL)
            report.ok("Alpamayo2-Super weights", "downloaded")
        except Exception as error:
            report.fail("Alpamayo2-Super weights",
                        "download failed: %s" % type(error).__name__)
    else:
        report.fail("Alpamayo2-Super weights",
                    "~67 GB, not cached. Re-run with --with-planner to download, "
                    "and set HF_HOME first if the default cache volume is small.")


def check_host(report):
    for name, why in (("ffmpeg", "clip extraction + H.264 delivery"),
                      ("git", "external checkouts")):
        path = shutil.which(name)

        if path:
            report.ok(name, path)
        else:
            report.fail(name, "not on PATH -- needed for " + why)

    python_h = "/usr/include/python%d.%d/Python.h" % sys.version_info[:2]

    if os.path.isfile(python_h):
        report.ok("python dev headers", python_h)
    else:
        report.warn("python dev headers",
                    "missing -- triton JIT fails and the Gemma gate falls back "
                    "to a heuristic (or errors under OPTICARVIS_REQUIRE_GEMMA_GATE). "
                    "apt install python%d.%d-dev" % sys.version_info[:2])

    for name in ("config", "secret"):
        if os.path.isfile(os.path.join(ROOT, name)):
            report.ok(name, "present")
        else:
            report.fail(name,
                        "missing -- cp default.%s %s and fill it in "
                        "(the batch cannot start without it)" % (name, name))


def main():
    parser = argparse.ArgumentParser(
        description="Fetch or verify the models and checkouts the pipeline needs.")
    parser.add_argument("--check-only", action="store_true",
                        help="report only; download nothing")
    parser.add_argument("--with-planner", action="store_true",
                        help="also clone alpamayo2 and download the ~67 GB checkpoint")
    args = parser.parse_args()

    download = not args.check_only

    print("")
    print("OptiCarVis asset setup" + (" (check only)" if args.check_only else ""))
    print("======================")

    report = Report()

    print("\nhost")
    check_host(report)

    print("\nrender models")
    check_ufld(report, download)
    check_hf_models(report, download)
    check_yolo(report, download)

    print("\nplanner")
    check_planner(report, download and args.with_planner)

    print("")

    if report.failed:
        print("%d required item(s) missing, %d warning(s)." % (report.failed, report.warned))
        print("Run scripts/setup_assets.py (add --with-planner for the checkpoint) "
              "to fetch what can be fetched automatically.")
        return 1

    print("All required assets present (%d warning(s))." % report.warned)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
