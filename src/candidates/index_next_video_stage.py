"""Index downloaded next stage videos and merge them into the main index.

Run after ``scripts/prefetch_source_videos.py`` has downloaded the videos listed
in ``workflow_outputs/final_study_selection/next_video_stage.jsonl``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


SRC_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SRC_DIR.parent
WORKFLOW_OUTPUTS = PROJECT_ROOT / "workflow_outputs"
SELECTION_DIR = WORKFLOW_OUTPUTS / "final_study_selection"

NEXT_VIDEO_STAGE_FILE = Path(
    os.environ.get(
        "OPTICARVIS_NEXT_VIDEO_STAGE_JSONL",
        SELECTION_DIR / "next_video_stage.jsonl",
    )
).resolve()
VIDEO_ROOT = Path(
    os.environ.get("OPTICARVIS_VIDEOS_DIR", PROJECT_ROOT / "videos")
).resolve()
MAIN_INDEX_FILE = Path(
    os.environ.get(
        "OPTICARVIS_CANDIDATE_INDEX_PARQUET",
        WORKFLOW_OUTPUTS / "candidate_index.parquet",
    )
).resolve()
STAGE_INDEX_FILE = WORKFLOW_OUTPUTS / "candidate_index_next_stage.parquet"
MAIN_SUMMARY_FILE = MAIN_INDEX_FILE.with_name(MAIN_INDEX_FILE.stem + "_summary.json")
STAGE_SUMMARY_FILE = STAGE_INDEX_FILE.with_name(
    STAGE_INDEX_FILE.stem + "_summary.json"
)
BUILD_INDEX_SCRIPT = SRC_DIR / "candidates" / "build_candidate_index.py"
STAGE_INDEXER_VERSION = 2
VIDEO_EXTENSIONS = (".mp4", ".mkv", ".mov", ".avi")


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


def stage_video_ids() -> list[str]:
    if not NEXT_VIDEO_STAGE_FILE.is_file():
        raise SystemExit(f"Next video stage not found: {NEXT_VIDEO_STAGE_FILE}")

    video_ids = []
    seen = set()
    with NEXT_VIDEO_STAGE_FILE.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            video_id = str(row.get("video_id", "")).strip()
            if video_id and video_id not in seen:
                seen.add(video_id)
                video_ids.append(video_id)
    return video_ids


def downloaded_video_ids(video_ids: list[str]) -> tuple[list[str], list[str]]:
    present = []
    missing = []
    for video_id in video_ids:
        if any(
            (VIDEO_ROOT / f"{video_id}{extension}").is_file()
            for extension in VIDEO_EXTENSIONS
        ):
            present.append(video_id)
        else:
            missing.append(video_id)
    return present, missing


def build_stage_index(video_ids: list[str]) -> bool:
    environment = os.environ.copy()
    environment["OPTICARVIS_CANDIDATE_INDEX_PARQUET"] = str(STAGE_INDEX_FILE)
    environment["OPTICARVIS_CANDIDATE_INDEX_ALLOW_PARTIAL"] = "1"
    if STAGE_INDEX_FILE.is_file():
        # A stale stage file from an earlier round must never be merged again.
        STAGE_INDEX_FILE.unlink()
    command = [sys.executable, str(BUILD_INDEX_SCRIPT), *video_ids]
    result = subprocess.run(command, cwd=PROJECT_ROOT, env=environment)
    return result.returncode == 0 and STAGE_INDEX_FILE.is_file()


def combine_index_frames(
    main: Any | None,
    stage: Any,
    pandas_module: Any,
) -> tuple[Any, set[str], int]:
    """Promote the first stage or replace its videos in an existing index."""
    if stage.empty:
        raise SystemExit("The next stage index contains no candidate windows.")
    if "video_id" not in stage.columns:
        raise SystemExit("Candidate index is missing the video_id column.")

    stage_ids = {str(value) for value in stage["video_id"].dropna().unique()}

    if main is None:
        combined = stage.copy().reset_index(drop=True)
    else:
        if "video_id" not in main.columns:
            raise SystemExit("Candidate index is missing the video_id column.")
        if list(main.columns) != list(stage.columns):
            raise SystemExit(
                "Main and next stage candidate index schemas do not match."
            )
        retained = main[~main["video_id"].astype(str).isin(stage_ids)]
        combined = pandas_module.concat([retained, stage], ignore_index=True)

    event_count = 0
    if "is_event_representative" in combined.columns:
        event_count = int(combined["is_event_representative"].fillna(False).sum())

    return combined, stage_ids, event_count


def merge_stage_index() -> tuple[int, int, int]:
    import pandas as pd

    stage = pd.read_parquet(STAGE_INDEX_FILE, engine="pyarrow")
    main = (
        pd.read_parquet(MAIN_INDEX_FILE, engine="pyarrow")
        if MAIN_INDEX_FILE.is_file()
        else None
    )
    combined, stage_ids, event_count = combine_index_frames(main, stage, pd)
    temporary = MAIN_INDEX_FILE.with_suffix(MAIN_INDEX_FILE.suffix + ".tmp")
    MAIN_INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)

    try:
        combined.to_parquet(temporary, index=False, engine="pyarrow")
        os.replace(temporary, MAIN_INDEX_FILE)
    finally:
        if temporary.exists():
            temporary.unlink()

    summary = read_json(MAIN_SUMMARY_FILE, {})
    if not summary:
        summary = read_json(STAGE_SUMMARY_FILE, {})
    indexed_ids = sorted(
        {str(value) for value in combined["video_id"].dropna().unique()}
    )
    summary.update(
        {
            "index_path": str(MAIN_INDEX_FILE),
            "index_written": True,
            "index_complete": True,
            "videos_selected": len(indexed_ids),
            "videos_indexed": len(indexed_ids),
            "indexed_video_ids": indexed_ids,
            "candidate_windows": int(len(combined)),
            "candidate_events": event_count,
            "failed_videos": [],
            "last_stage_video_ids": sorted(stage_ids),
        }
    )
    write_json_atomic(MAIN_SUMMARY_FILE, summary)
    return len(stage_ids), len(combined), event_count


def main() -> int:
    requested = stage_video_ids()
    present, missing = downloaded_video_ids(requested)

    print("")
    print("Next video candidate index stage")
    print("================================")
    print("stage indexer version:", STAGE_INDEXER_VERSION)
    print("stage videos:", len(requested))
    print("downloaded videos:", len(present))
    print("missing videos:", len(missing))

    for video_id in missing:
        print("  not downloaded:", video_id)

    if not present:
        print("No downloaded stage videos are available to index.")
        return 1

    if not build_stage_index(present):
        # Every downloaded video failed to index. That costs ranking only:
        # clip_job_builder.py gives unindexed local videos stride-fallback
        # candidates, so the staged search can still evaluate them.
        print("")
        print("No stage video produced candidate windows; the main index was")
        print("preserved and these videos will use stride-fallback candidates.")
        return 0

    merged_videos, windows, events = merge_stage_index()

    print("")
    print("Stage index merged")
    print("videos merged:", merged_videos)
    print("candidate windows:", windows)
    print("candidate events:", events)
    print("main index:", MAIN_INDEX_FILE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
