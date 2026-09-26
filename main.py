r"""Run the complete OptiCarVis study preparation from one command.

This is the entry point of the repository (formerly ``analysis.py``) and the
public entry point for the staged city search.  It is safe to run
on a clean checkout with no ``videos`` or ``workflow_outputs`` directory and
safe to rerun after interruption.  For every unresolved city it downloads one
mapped source video, builds or extends the semantic candidate index, evaluates
the ranked 30 second candidates, and repeats with the next mapped video until
one valid rendered explanation is accepted or the city's mapping is exhausted.

Run from the repository root::

    uv run python .\main.py

Resuming versus starting over is controlled by ``always_analyse`` in
``config`` (env override ``OPTICARVIS_ALWAYS_ANALYSE``):

* ``false`` (default): resume. Accepted cities, rejected candidates, the
  semantic index and a half-finished download round are all kept, and the run
  continues from the next unfinished step.
* ``true``: start from scratch. Every generated artefact under
  ``workflow_outputs`` and ``alpamayo_outputs`` is deleted first (index, jobs,
  gate decisions, planner output, renders, selection progress). Downloaded
  source videos in ``videos`` are kept - they are inputs, not analysis.

The specialised modules in ``src`` and ``scripts`` remain implementation
modules.  Keeping them separate makes the expensive stages independently
testable and resumable; users do not need to invoke them manually.
``scripts/run_batch_jobs.py`` (the former ``main.py``) remains for running every
generated clip job in bulk, without the per-city staged selection.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
WORKFLOW_OUTPUTS = PROJECT_ROOT / "workflow_outputs"
SELECTION_DIR = WORKFLOW_OUTPUTS / "final_study_selection"
MAIN_INDEX_FILE = WORKFLOW_OUTPUTS / "candidate_index.parquet"
MAIN_INDEX_SUMMARY = WORKFLOW_OUTPUTS / "candidate_index_summary.json"
NEXT_STAGE_FILE = SELECTION_DIR / "next_video_stage.jsonl"
PROGRESS_FILE = SELECTION_DIR / "selection_progress.json"
FINAL_MANIFEST = WORKFLOW_OUTPUTS / "final_study_segments.json"
ANALYSIS_SUMMARY = SELECTION_DIR / "analysis_summary.json"

BUILD_INDEX_SCRIPT = SRC_DIR / "candidates" / "build_candidate_index.py"
BUILD_JOBS_SCRIPT = SRC_DIR / "candidates" / "clip_job_builder.py"
SELECT_SEGMENTS_SCRIPT = SRC_DIR / "selection" / "run_final_study_segment_selection.py"
INDEX_STAGE_SCRIPT = SRC_DIR / "candidates" / "index_next_video_stage.py"
PREFETCH_SCRIPT = SCRIPTS_DIR / "prefetch_source_videos.py"

VIDEO_EXTENSIONS = (".mp4", ".mkv", ".mov", ".avi")
ANALYSIS_VERSION = 1


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return default


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            if isinstance(value, dict):
                rows.append(value)
    return rows


def load_config() -> dict[str, Any]:
    configured = PROJECT_ROOT / "config"
    fallback = PROJECT_ROOT / "default.config"
    path = configured if configured.is_file() else fallback
    value = read_json(path, {})
    return value if isinstance(value, dict) else {}


def project_path(value: Any, fallback: str) -> Path:
    text = str(value or fallback).strip()
    path = Path(text)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


CONFIG = load_config()
VIDEO_ROOT = project_path(
    CONFIG.get("videos"),
    "videos",
)
ALPAMAYO_OUTPUTS = project_path(
    CONFIG.get("alpamayo_outputs"),
    "alpamayo_outputs",
)
# One candidate file for every stage: clip_job_builder.py writes it, the
# selector reads it, and the batch indexes into it. All three are pointed at it
# explicitly (child_environment) so they cannot drift apart again - they once
# defaulted to different files and the first selection step always failed.
JOBS_FILE = project_path(
    CONFIG.get("clip_jobs_jsonl"),
    "workflow_outputs/clip_jobs.jsonl",
)
JOBS_SUMMARY_FILE = project_path(
    CONFIG.get("clip_jobs_summary_json"),
    "workflow_outputs/clip_jobs_summary.json",
)


def environment_int(name: str, default: int, minimum: int = 0) -> int:
    try:
        value = int(str(os.environ.get(name, default)).strip())
    except (TypeError, ValueError):
        value = int(default)
    return max(minimum, value)


def config_flag(key: str, default: bool) -> bool:
    """A boolean from OPTICARVIS_<KEY> or the config file."""
    value = os.environ.get("OPTICARVIS_" + key.upper())
    if value is None or not str(value).strip():
        value = CONFIG.get(key, default)
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "y", "on"):
        return True
    if text in ("0", "false", "no", "n", "off"):
        return False
    return default


def child_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.setdefault("OPTICARVIS_VIDEOS_DIR", str(VIDEO_ROOT))
    # City names are not ASCII ("Hạ Long"); on Windows a piped or logged
    # stdout is cp1252 and the first such print() kills the stage.
    environment.setdefault("PYTHONIOENCODING", "utf-8")
    environment["OPTICARVIS_CLIP_JOBS_JSONL"] = str(JOBS_FILE)
    environment["OPTICARVIS_CLIP_JOBS_SUMMARY_JSON"] = str(JOBS_SUMMARY_FILE)
    environment["OPTICARVIS_NEXT_VIDEO_STAGE_JSONL"] = str(NEXT_STAGE_FILE)
    return environment


def run_step(
    label: str,
    command: list[str],
    accepted_codes: tuple[int, ...] = (0,),
) -> int:
    print("")
    print("=" * 80)
    print(label)
    print("=" * 80)
    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=child_environment(),
    )
    if result.returncode not in accepted_codes:
        raise SystemExit(
            "%s failed with exit code %d. Rerunning main.py is safe."
            % (label, result.returncode)
        )
    return result.returncode


def is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def reset_generated_outputs() -> None:
    """Delete every generated artefact so the analysis starts from scratch.

    Only directories inside the repository are touched, and never one that is
    or contains the source-video directory: re-downloading the corpus is not
    part of re-running the analysis.
    """
    print("")
    print("always_analyse is true: removing previous analysis outputs.")
    for directory in (WORKFLOW_OUTPUTS, ALPAMAYO_OUTPUTS):
        if not directory.exists():
            continue
        if directory.resolve() == PROJECT_ROOT.resolve() or not is_within(directory, PROJECT_ROOT):
            raise SystemExit(
                "Refusing to delete %s: it is not a subdirectory of the repository."
                % directory
            )
        if is_within(VIDEO_ROOT, directory):
            raise SystemExit(
                "Refusing to delete %s: it contains the source videos (%s)."
                % (directory, VIDEO_ROOT)
            )
        print("  removing", directory)
        shutil.rmtree(directory)
    print("  kept source videos in", VIDEO_ROOT)


def ensure_layout() -> None:
    VIDEO_ROOT.mkdir(parents=True, exist_ok=True)
    WORKFLOW_OUTPUTS.mkdir(parents=True, exist_ok=True)
    SELECTION_DIR.mkdir(parents=True, exist_ok=True)
    ALPAMAYO_OUTPUTS.mkdir(parents=True, exist_ok=True)


def validate_scripts() -> None:
    required = [
        BUILD_INDEX_SCRIPT,
        BUILD_JOBS_SCRIPT,
        SELECT_SEGMENTS_SCRIPT,
        INDEX_STAGE_SCRIPT,
        PREFETCH_SCRIPT,
        PROJECT_ROOT / "mapping.csv",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("Required files are missing:\n  " + "\n  ".join(missing))


def stage_signature(rows: list[dict[str, Any]]) -> tuple[tuple[int, str], ...]:
    signature = []
    for row in rows:
        try:
            city_index = int(row.get("city_index"))
        except (TypeError, ValueError):
            continue
        video_id = str(row.get("video_id", "")).strip()
        if video_id:
            signature.append((city_index, video_id))
    return tuple(sorted(signature))


def video_is_present(video_id: str) -> bool:
    return any((VIDEO_ROOT / (video_id + extension)).is_file() for extension in VIDEO_EXTENSIONS)


def partition_stage_rows(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    present = []
    missing = []
    for row in rows:
        video_id = str(row.get("video_id", "")).strip()
        if video_id and video_is_present(video_id):
            present.append(row)
        else:
            missing.append(row)
    return present, missing


def record_unavailable_stage_rows(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    progress = read_json(PROGRESS_FILE, {"attempts": [], "city_status": {}})
    if not isinstance(progress, dict):
        progress = {"attempts": [], "city_status": {}}
    unavailable = progress.setdefault("unavailable_video_ids_by_city", {})
    if not isinstance(unavailable, dict):
        unavailable = {}
        progress["unavailable_video_ids_by_city"] = unavailable
    failures = progress.setdefault("video_download_failures", [])
    if not isinstance(failures, list):
        failures = []
        progress["video_download_failures"] = failures

    for row in rows:
        try:
            city_index = str(int(row.get("city_index")))
        except (TypeError, ValueError):
            continue
        video_id = str(row.get("video_id", "")).strip()
        if not video_id:
            continue
        blocked = unavailable.setdefault(city_index, [])
        if video_id not in blocked:
            blocked.append(video_id)
        failures.append(
            {
                "city_index": int(city_index),
                "city": row.get("city", "Unknown"),
                "country": row.get("country", "Unknown"),
                "video_id": video_id,
                "failed_at": utc_now(),
            }
        )

    progress["updated_at"] = utc_now()
    write_json_atomic(PROGRESS_FILE, progress)


def initialise_candidate_index() -> None:
    if MAIN_INDEX_FILE.is_file():
        return
    return_code = run_step(
        "Initial semantic candidate index",
        [sys.executable, str(BUILD_INDEX_SCRIPT)],
        accepted_codes=(0, 1),
    )
    if MAIN_INDEX_FILE.is_file():
        return

    summary = read_json(MAIN_INDEX_SUMMARY, {})
    local_count = int(summary.get("videos_selected", 0) or 0)
    failed = summary.get("failed_videos", [])
    if return_code == 1 and local_count == 0:
        print("No local mapped videos were found. The first FTP stage will be prepared.")
        return
    if failed:
        raise SystemExit(
            "The initial index failed for local videos. Fix the reported video "
            "errors, then rerun main.py."
        )
    raise SystemExit("The initial candidate index was not created.")


def run_selection_cycle() -> None:
    run_step(
        "Build ranked 30 second city candidates",
        [sys.executable, str(BUILD_JOBS_SCRIPT)],
    )
    run_step(
        "Evaluate candidates and render accepted explanations",
        [sys.executable, str(SELECT_SEGMENTS_SCRIPT)],
    )


def download_stage(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    workers = environment_int("OPTICARVIS_ANALYSIS_DOWNLOAD_WORKERS", 3, 1)
    attempts = environment_int("OPTICARVIS_ANALYSIS_DOWNLOAD_ATTEMPTS", 2, 1)
    present, missing = partition_stage_rows(rows)

    for attempt in range(1, attempts + 1):
        if not missing:
            break
        print(
            "Download attempt %d/%d for %d missing stage video(s)."
            % (attempt, attempts, len(missing))
        )
        run_step(
            "Download missing mapped videos",
            [
                sys.executable,
                str(PREFETCH_SCRIPT),
                "--jobs-jsonl",
                str(NEXT_STAGE_FILE),
                "--workers",
                str(workers),
            ],
            accepted_codes=(0, 1),
        )
        present, missing = partition_stage_rows(rows)

    return present, missing


def index_downloaded_stage() -> None:
    run_step(
        "Index newly downloaded videos and extend the semantic index",
        [sys.executable, str(INDEX_STAGE_SCRIPT)],
    )


def final_summary(rounds: int) -> dict[str, Any]:
    manifest = read_json(FINAL_MANIFEST, {})
    progress = read_json(PROGRESS_FILE, {})
    jobs_summary = read_json(JOBS_SUMMARY_FILE, {})
    segments = manifest.get("segments", []) if isinstance(manifest, dict) else []
    statuses = progress.get("city_status", {}) if isinstance(progress, dict) else {}
    expected = int(jobs_summary.get("cities", 0) or 0)
    exhausted = [
        int(city_index)
        for city_index, status in statuses.items()
        if status == "all_mapped_videos_exhausted"
    ]
    accepted = len(segments) if isinstance(segments, list) else 0
    unresolved = max(0, expected - accepted - len(exhausted))
    summary = {
        "analysis_version": ANALYSIS_VERSION,
        "completed_at": utc_now(),
        "video_rounds": rounds,
        "expected_cities": expected,
        "accepted_cities": accepted,
        "exhausted_cities": len(exhausted),
        "exhausted_city_indices": sorted(exhausted),
        "unresolved_cities": unresolved,
        "manifest": str(FINAL_MANIFEST),
        "segments": segments if isinstance(segments, list) else [],
    }
    write_json_atomic(ANALYSIS_SUMMARY, summary)
    return summary


def main() -> int:
    validate_scripts()
    ensure_layout()

    print("OptiCarVis complete analysis")
    print("============================")
    print("analysis_version:", ANALYSIS_VERSION)
    print("project_root:", PROJECT_ROOT)
    print("video_root:", VIDEO_ROOT)
    print("workflow_outputs:", WORKFLOW_OUTPUTS)

    always_analyse = config_flag("always_analyse", False)
    print("always_analyse:", always_analyse)
    if always_analyse:
        reset_generated_outputs()
        ensure_layout()

    initialise_candidate_index()

    # A pending next-video stage means an earlier run stopped mid-round
    # (while downloading, indexing or evaluating). Finish that round first:
    # rebuilding jobs now would give its downloaded-but-unindexed videos
    # unranked stride windows and mark them evaluated.
    if read_jsonl(NEXT_STAGE_FILE):
        print("")
        print("Resuming the unfinished video round from", NEXT_STAGE_FILE)
    else:
        run_selection_cycle()

    max_rounds = environment_int("OPTICARVIS_ANALYSIS_MAX_VIDEO_ROUNDS", 0, 0)
    rounds = 0

    while True:
        stage_rows = read_jsonl(NEXT_STAGE_FILE)
        if not stage_rows:
            break
        if max_rounds and rounds >= max_rounds:
            print(
                "Stopped at the configured video round limit (%d). "
                "Rerun main.py to continue." % max_rounds
            )
            break

        rounds += 1
        before = stage_signature(stage_rows)
        print("")
        print("#" * 80)
        print("VIDEO ROUND %d: %d unresolved city(ies)" % (rounds, len(stage_rows)))
        print("#" * 80)

        present, missing = download_stage(stage_rows)
        if missing:
            print(
                "%d video(s) remained unavailable after all download attempts; "
                "their cities will advance to the next mapped video."
                % len(missing)
            )
            record_unavailable_stage_rows(missing)

        if present:
            index_downloaded_stage()

        run_selection_cycle()
        after_rows = read_jsonl(NEXT_STAGE_FILE)
        after = stage_signature(after_rows)
        if after and after == before:
            raise SystemExit(
                "The staged search made no progress. The current files were "
                "preserved; inspect selection_progress.json and rerun."
            )

    summary = final_summary(rounds)
    print("")
    print("OptiCarVis analysis complete")
    print("============================")
    print("accepted cities:", summary["accepted_cities"])
    print("mapped videos exhausted:", summary["exhausted_cities"])
    print("still unresolved:", summary["unresolved_cities"])
    print("final manifest:", FINAL_MANIFEST)
    print("analysis summary:", ANALYSIS_SUMMARY)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nStopped by user. Rerun the same command to continue safely.")
        raise SystemExit(130)
