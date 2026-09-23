"""Focused, model free tests for the single command analysis workflow."""

from __future__ import annotations

import importlib
import json
import os
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"

for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


analysis = importlib.import_module("analysis")
selector = importlib.import_module("run_final_study_segment_selection")
stage_indexer = importlib.import_module("index_next_video_stage")


def test_main_runs_download_index_and_selection_round():
    names = [
        "validate_scripts",
        "ensure_layout",
        "initialise_candidate_index",
        "run_selection_cycle",
        "read_jsonl",
        "download_stage",
        "index_downloaded_stage",
        "final_summary",
    ]
    original = {name: getattr(analysis, name) for name in names}
    original_limit = os.environ.pop("OPTICARVIS_ANALYSIS_MAX_VIDEO_ROUNDS", None)
    calls = []
    stage = [{"city_index": 1, "video_id": "video_one"}]
    stage_reads = [stage, [], []]

    try:
        analysis.validate_scripts = lambda: calls.append("validate")
        analysis.ensure_layout = lambda: calls.append("layout")
        analysis.initialise_candidate_index = lambda: calls.append("initialise")
        analysis.run_selection_cycle = lambda: calls.append("select")
        analysis.read_jsonl = lambda _path: stage_reads.pop(0)
        analysis.download_stage = lambda rows: (rows, [])
        analysis.index_downloaded_stage = lambda: calls.append("index")
        analysis.final_summary = lambda rounds: {
            "accepted_cities": 1,
            "exhausted_cities": 0,
            "unresolved_cities": 0,
        }

        assert analysis.main() == 0
        assert calls == [
            "validate",
            "layout",
            "initialise",
            "select",
            "index",
            "select",
        ]
    finally:
        for name, value in original.items():
            setattr(analysis, name, value)
        if original_limit is not None:
            os.environ["OPTICARVIS_ANALYSIS_MAX_VIDEO_ROUNDS"] = original_limit


def test_stage_partition_detects_downloaded_video():
    original = analysis.VIDEO_ROOT
    with tempfile.TemporaryDirectory() as directory:
        analysis.VIDEO_ROOT = Path(directory)
        (analysis.VIDEO_ROOT / "available.mp4").write_bytes(b"video")
        rows = [
            {"city_index": 1, "video_id": "available"},
            {"city_index": 2, "video_id": "missing"},
        ]
        present, missing = analysis.partition_stage_rows(rows)
        assert [row["video_id"] for row in present] == ["available"]
        assert [row["video_id"] for row in missing] == ["missing"]
    analysis.VIDEO_ROOT = original


def test_failed_download_is_recorded_per_city():
    original = analysis.PROGRESS_FILE
    with tempfile.TemporaryDirectory() as directory:
        analysis.PROGRESS_FILE = Path(directory) / "selection_progress.json"
        analysis.record_unavailable_stage_rows(
            [
                {
                    "city_index": 39,
                    "city": "Guangzhou",
                    "country": "China",
                    "video_id": "missing_video",
                }
            ]
        )
        progress = json.loads(analysis.PROGRESS_FILE.read_text(encoding="utf-8"))
        assert progress["unavailable_video_ids_by_city"]["39"] == [
            "missing_video"
        ]
        assert progress["video_download_failures"][0]["city"] == "Guangzhou"
    analysis.PROGRESS_FILE = original


def test_selector_skips_unavailable_video_and_advances():
    original = selector.NEXT_VIDEO_STAGE_FILE
    with tempfile.TemporaryDirectory() as directory:
        selector.NEXT_VIDEO_STAGE_FILE = Path(directory) / "next.jsonl"
        progress = {
            "city_status": {"39": "no_candidate_accepted"},
            "unavailable_video_ids_by_city": {"39": ["unavailable"]},
        }
        rows = selector.prepare_next_video_stage(
            [
                {
                    "city_index": 39,
                    "city": "Guangzhou",
                    "country": "China",
                    "deferred_video_ids": ["unavailable", "next_video"],
                }
            ],
            set(),
            progress,
        )
        assert rows[0]["video_id"] == "next_video"
    selector.NEXT_VIDEO_STAGE_FILE = original


def test_selector_does_not_redownload_a_completed_video():
    original = selector.NEXT_VIDEO_STAGE_FILE
    with tempfile.TemporaryDirectory() as directory:
        selector.NEXT_VIDEO_STAGE_FILE = Path(directory) / "next.jsonl"
        progress = {
            "city_status": {"39": "no_candidate_accepted"},
            "completed_video_ids_by_city": {"39": ["first_video"]},
        }
        rows = selector.prepare_next_video_stage(
            [
                {
                    "city_index": 39,
                    "city": "Guangzhou",
                    "country": "China",
                    "deferred_video_ids": ["first_video", "second_video"],
                }
            ],
            set(),
            progress,
        )
        assert rows[0]["video_id"] == "second_video"
    selector.NEXT_VIDEO_STAGE_FILE = original


def test_selector_marks_city_exhausted_when_no_video_remains():
    original = selector.NEXT_VIDEO_STAGE_FILE
    with tempfile.TemporaryDirectory() as directory:
        selector.NEXT_VIDEO_STAGE_FILE = Path(directory) / "next.jsonl"
        progress = {
            "city_status": {"39": "no_local_candidate"},
            "unavailable_video_ids_by_city": {"39": ["only_video"]},
        }
        rows = selector.prepare_next_video_stage(
            [
                {
                    "city_index": 39,
                    "city": "Guangzhou",
                    "country": "China",
                    "deferred_video_ids": ["only_video"],
                }
            ],
            set(),
            progress,
        )
        assert rows == []
        assert progress["city_status"]["39"] == "all_mapped_videos_exhausted"
    selector.NEXT_VIDEO_STAGE_FILE = original


def test_first_stage_can_be_promoted_without_existing_index():
    try:
        import pandas as pd
    except ImportError:
        return

    stage = pd.DataFrame(
        [
            {"video_id": "first", "is_event_representative": True},
            {"video_id": "first", "is_event_representative": False},
        ]
    )
    combined, stage_ids, event_count = stage_indexer.combine_index_frames(
        None,
        stage,
        pd,
    )
    assert list(combined["video_id"]) == ["first", "first"]
    assert stage_ids == {"first"}
    assert event_count == 1


if __name__ == "__main__":
    failures = 0
    for name, test in sorted(globals().items()):
        if not name.startswith("test_") or not callable(test):
            continue
        try:
            test()
            print("PASS  %s" % name)
        except Exception as error:
            failures += 1
            print("FAIL  %s\n      %s" % (name, error))
    raise SystemExit(1 if failures else 0)
