r"""Select one reproducible 30 second OptiCarVis segment per mapped city.

The candidate file is produced by ``clip_job_builder.py``.  This runner tries
each city's candidates in their recorded order, stops at the first fully valid
visual explanation, and writes an atomic manifest.  A candidate is valid only
when Gemma approved the moment, MIRAGE planned at least one visual layer, and
both the reference render and replayable geometry exist.

Usage from the repository root::

    uv run python .\src\run_final_study_segment_selection.py
    uv run python .\src\run_final_study_segment_selection.py 1 149

With no arguments, all mapped cities are processed.  Optional arguments are
the number of cities and the zero based city offset in mapping order.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
WORKFLOW_OUTPUTS = PROJECT_ROOT / "workflow_outputs"
SELECTION_DIR = WORKFLOW_OUTPUTS / "final_study_selection"

JOBS_FILE = Path(
    os.environ.get(
        "OPTICARVIS_CLIP_JOBS_JSONL",
        SELECTION_DIR / "candidate_windows.jsonl",
    )
).resolve()
MASTER_INDEX_FILE = Path(
    os.environ.get(
        "OPTICARVIS_MASTER_CLIP_INDEX_JSONL",
        SELECTION_DIR / "master_clip_index.jsonl",
    )
).resolve()
FINAL_MANIFEST = WORKFLOW_OUTPUTS / "final_study_segments.json"
PROGRESS_FILE = SELECTION_DIR / "selection_progress.json"
CANDIDATE_SUMMARY_FILE = Path(
    os.environ.get(
        "OPTICARVIS_CLIP_JOBS_SUMMARY_JSON",
        SELECTION_DIR / "candidate_windows_summary.json",
    )
).resolve()
NEXT_VIDEO_STAGE_FILE = Path(
    os.environ.get(
        "OPTICARVIS_NEXT_VIDEO_STAGE_JSONL",
        SELECTION_DIR / "next_video_stage.jsonl",
    )
).resolve()
BATCH_PIPELINE = SRC_DIR / "batch_corrected_pipeline.py"
FINAL_RENDER_DIR = WORKFLOW_OUTPUTS / "final_renders"
GEOMETRY_DIR = WORKFLOW_OUTPUTS / "overlay_geometry"
GATE_DIR = WORKFLOW_OUTPUTS / "gemma_reasoning"
MIRAGE_DIR = WORKFLOW_OUTPUTS / "mirage"
EXPECTED_DURATION_SECONDS = 30.0
STAGED_SELECTOR_VERSION = 3


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


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, path)


def relative_path(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_jobs() -> list[dict[str, Any]]:
    if not JOBS_FILE.is_file():
        raise SystemExit(f"Candidate file not found: {JOBS_FILE}")
    jobs: list[dict[str, Any]] = []
    with JOBS_FILE.open("r", encoding="utf-8-sig") as handle:
        for line_index, line in enumerate(handle):
            if not line.strip():
                continue
            job = json.loads(line)
            if float(job.get("clip_length_s", 0.0)) != EXPECTED_DURATION_SECONDS:
                raise ValueError(
                    f"{job.get('job_id')} is not a 30 second candidate"
                )
            job["_job_index"] = line_index
            jobs.append(job)
    return jobs


def group_jobs(jobs: list[dict[str, Any]]) -> OrderedDict[Any, list[dict[str, Any]]]:
    grouped: OrderedDict[Any, list[dict[str, Any]]] = OrderedDict()
    for job in jobs:
        grouped.setdefault(job.get("city_index"), []).append(job)
    return grouped


def load_city_summaries() -> list[dict[str, Any]]:
    summary = read_json(CANDIDATE_SUMMARY_FILE, {})
    rows = summary.get("city_summaries", []) if isinstance(summary, dict) else []
    return rows if isinstance(rows, list) else []


def city_metadata_by_index(
    city_summaries: list[dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    result = {}
    for row in city_summaries:
        try:
            result[int(row["city_index"])] = row
        except (KeyError, TypeError, ValueError):
            continue
    return result


def clip_tag(job: dict[str, Any]) -> str:
    start = int(round(float(job["segment_start_time_s"])))
    return f"{job['video_id']}_{start}"


def newest_matching_file(directory: Path, prefix: str, suffix: str) -> Path | None:
    if not directory.is_dir():
        return None
    matches = [
        path
        for path in directory.rglob(f"{prefix}*{suffix}")
        if path.is_file() and path.stat().st_size > 0
    ]
    return max(matches, key=lambda path: path.stat().st_mtime) if matches else None


def accepted_outputs(job: dict[str, Any]) -> tuple[Path | None, Path | None]:
    tag = clip_tag(job)
    render = newest_matching_file(FINAL_RENDER_DIR, tag + "_", "_vehicles.mp4")
    if render is None:
        render = newest_matching_file(FINAL_RENDER_DIR, tag + "_", ".mp4")
    geometry = newest_matching_file(GEOMETRY_DIR, tag, "_geometry.jsonl.gz")
    return render, geometry


def acceptance_evidence(
    job: dict[str, Any],
    render: Path | None = None,
    geometry: Path | None = None,
) -> tuple[bool, str, Path | None, Path | None]:
    """Validate semantic, visual, and file evidence for one candidate."""
    if render is None or geometry is None:
        discovered_render, discovered_geometry = accepted_outputs(job)
        render = render or discovered_render
        geometry = geometry or discovered_geometry

    tag = clip_tag(job)
    gate_path = GATE_DIR / f"{tag}_gemma_gate.json"
    plan_path = MIRAGE_DIR / f"{tag}_effect_plan.json"
    gate = read_json(gate_path, {})
    plan = read_json(plan_path, {})

    if gate.get("proper_time_to_explain") is not True:
        return False, "gemma_gate_not_approved", render, geometry
    if gate.get("decision") != "explain_now":
        return False, "gemma_decision_not_explain_now", render, geometry

    policy = str(plan.get("explanation_policy", ""))
    if not policy.startswith("render_"):
        return False, "mirage_policy_does_not_render", render, geometry
    if plan.get("display_target") in (None, "", "none"):
        return False, "mirage_display_target_missing", render, geometry
    if plan.get("mirage_effect_family") in (None, "", "none"):
        return False, "mirage_effect_family_missing", render, geometry
    if not isinstance(plan.get("visual_layers"), list) or not plan["visual_layers"]:
        return False, "mirage_visual_layers_missing", render, geometry
    if render is None or not render.is_file() or render.stat().st_size == 0:
        return False, "reference_render_missing", render, geometry
    if geometry is None or not geometry.is_file() or geometry.stat().st_size == 0:
        return False, "overlay_geometry_missing", render, geometry

    return True, "valid_visual_explanation", render, geometry


def run_candidate(job: dict[str, Any]) -> tuple[bool, int, str]:
    passed, reason, render, geometry = acceptance_evidence(job)
    if passed:
        print("Using existing validated outputs:", render)
        return True, 0, reason

    environment = os.environ.copy()
    environment["OPTICARVIS_CLIP_JOBS_JSONL"] = str(JOBS_FILE)
    environment["OPTICARVIS_MASTER_CLIP_INDEX_JSONL"] = str(MASTER_INDEX_FILE)
    command = [
        sys.executable,
        str(BATCH_PIPELINE),
        "1",
        str(job["_job_index"]),
    ]
    result = subprocess.run(command, cwd=PROJECT_ROOT, env=environment)
    passed, reason, _render, _geometry = acceptance_evidence(job)
    return passed, result.returncode, reason


def load_manifest() -> dict[str, Any]:
    return read_json(
        FINAL_MANIFEST,
        {
            "manifest_version": "opticarvis_final_city_segments_v2",
            "purpose": "one approved 30 second candidate per sampled city",
            "mapping_rows": 150,
            "expected_cities": 150,
            "duration_seconds": EXPECTED_DURATION_SECONDS,
            "segments": [],
            "invalidated_segments": [],
        },
    )


def segment_as_job(segment: dict[str, Any]) -> dict[str, Any]:
    return {
        "video_id": segment["source_video_id"],
        "segment_start_time_s": segment["start_seconds"],
    }


def revalidate_manifest(manifest: dict[str, Any]) -> int:
    """Move legacy file only acceptances out of the active segment list."""
    retained = []
    invalidated = list(manifest.get("invalidated_segments", []))

    for segment in manifest.get("segments", []):
        render_value = segment.get("reference_render")
        geometry_value = segment.get("geometry")
        render = project_path(render_value) if render_value else None
        geometry = project_path(geometry_value) if geometry_value else None
        passed, reason, _render, _geometry = acceptance_evidence(
            segment_as_job(segment),
            render,
            geometry,
        )
        if passed:
            retained.append(segment)
            continue

        invalid = dict(segment)
        invalid["invalidated_at"] = utc_now()
        invalid["invalidation_reason"] = reason
        invalidated.append(invalid)
        print(
            "INVALIDATED previous segment:",
            segment.get("city", segment.get("city_id", "unknown")),
            "reason:",
            reason,
        )

    changed = len(retained) != len(manifest.get("segments", []))
    manifest["manifest_version"] = "opticarvis_final_city_segments_v2"
    manifest["segments"] = retained
    manifest["invalidated_segments"] = invalidated
    if changed:
        manifest["updated_at"] = utc_now()
        write_json_atomic(FINAL_MANIFEST, manifest)
    return len(retained)


def candidate_rank(job: dict[str, Any]) -> int:
    value = job.get("selection_rank")
    if value is None:
        value = job.get("window_index", 0)
    return int(value)


def segment_record(
    job: dict[str, Any], render: Path, geometry: Path
) -> dict[str, Any]:
    tag = clip_tag(job)
    plan_path = MIRAGE_DIR / f"{tag}_effect_plan.json"
    plan = read_json(plan_path, {})
    return {
        "city_id": f"city{int(job['city_index']):03d}",
        "city_index": int(job["city_index"]),
        "city": job["city"],
        "country": job["country"],
        "continent": job.get("continent", ""),
        "source_video_id": job["video_id"],
        "start_seconds": float(job["segment_start_time_s"]),
        "duration_seconds": EXPECTED_DURATION_SECONDS,
        "candidate_rank": candidate_rank(job),
        "selection_method": job.get("selection_method"),
        "selection_score": job.get("selection_score"),
        "geometry": relative_path(geometry),
        "reference_render": relative_path(render),
        "gemma_gate": relative_path(GATE_DIR / f"{tag}_gemma_gate.json"),
        "mirage_effect_plan": relative_path(plan_path),
        "explanation_policy": plan.get("explanation_policy"),
        "visual_layer_count": len(plan.get("visual_layers", [])),
        "automatic_gate_passed": True,
        "automatic_visual_plan_passed": True,
        "manual_review_status": "pending",
        "selected_at": utc_now(),
    }


def save_progress(progress: dict[str, Any]) -> None:
    progress["updated_at"] = utc_now()
    write_json_atomic(PROGRESS_FILE, progress)


def attempted_rejection_ids(progress: dict[str, Any]) -> set[str]:
    rejected = set()
    for attempt in progress.get("attempts", []):
        if attempt.get("accepted") is False and attempt.get("job_id"):
            rejected.add(str(attempt["job_id"]))
    return rejected


def prepare_next_video_stage(
    city_summaries: list[dict[str, Any]],
    accepted_indices: set[int],
    progress: dict[str, Any],
) -> list[dict[str, Any]]:
    """Write one next mapped video for each city that exhausted this round."""
    stage_rows = []
    statuses = progress.get("city_status", {})
    unavailable_by_city = progress.get("unavailable_video_ids_by_city", {})
    completed_by_city = progress.get("completed_video_ids_by_city", {})

    if not isinstance(unavailable_by_city, dict):
        unavailable_by_city = {}
    if not isinstance(completed_by_city, dict):
        completed_by_city = {}

    for city in city_summaries:
        try:
            city_index = int(city["city_index"])
        except (KeyError, TypeError, ValueError):
            continue

        if city_index in accepted_indices:
            continue

        status = statuses.get(str(city_index))
        if status not in ("no_candidate_accepted", "no_local_candidate"):
            continue

        deferred = city.get("deferred_video_ids", [])
        blocked = unavailable_by_city.get(str(city_index), [])
        blocked_ids = {
            str(video_id).strip()
            for video_id in blocked
            if str(video_id).strip()
        } if isinstance(blocked, list) else set()
        completed = completed_by_city.get(str(city_index), [])
        if isinstance(completed, list):
            blocked_ids.update(
                str(video_id).strip()
                for video_id in completed
                if str(video_id).strip()
            )
        available_deferred = [
            str(video_id).strip()
            for video_id in deferred
            if str(video_id).strip() and str(video_id).strip() not in blocked_ids
        ] if isinstance(deferred, list) else []

        if not available_deferred:
            statuses[str(city_index)] = "all_mapped_videos_exhausted"
            continue

        video_id = available_deferred[0]

        stage_rows.append(
            {
                "job_id": f"next_city{city_index:03d}_{video_id}",
                "city_index": city_index,
                "city": city.get("city", "Unknown"),
                "country": city.get("country", "Unknown"),
                "continent": city.get("continent", ""),
                "video_id": video_id,
                "stage_reason": status,
            }
        )

    progress["city_status"] = statuses
    write_jsonl_atomic(NEXT_VIDEO_STAGE_FILE, stage_rows)
    return stage_rows


def main() -> None:
    requested_max_cities = int(sys.argv[1]) if len(sys.argv) > 1 else None
    start_city = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    if requested_max_cities is not None and requested_max_cities < 1:
        raise SystemExit("max_cities must be positive")
    if start_city < 0:
        raise SystemExit("max_cities must be positive and start_city cannot be negative")

    jobs = load_jobs()
    groups = group_jobs(jobs)
    groups_by_index = {int(index): rows for index, rows in groups.items()}
    city_summaries = load_city_summaries()
    metadata = city_metadata_by_index(city_summaries)
    mapped_city_count = max(
        [int(index) for index in groups_by_index] + list(metadata) + [0]
    )
    max_cities = requested_max_cities or max(0, mapped_city_count - start_city)
    first_city_index = start_city + 1
    last_city_index = min(mapped_city_count, start_city + max_cities)
    selected_city_indices = range(first_city_index, last_city_index + 1)
    manifest = load_manifest()
    revalidate_manifest(manifest)
    accepted_indices = {int(item["city_index"]) for item in manifest["segments"]}
    progress = read_json(PROGRESS_FILE, {"attempts": [], "city_status": {}})
    progress.setdefault("attempts", [])
    progress.setdefault("city_status", {})
    progress.setdefault("completed_video_ids_by_city", {})
    rejected_job_ids = attempted_rejection_ids(progress)

    for city_index, status in list(progress["city_status"].items()):
        if status == "validated_visual_explanation" and int(city_index) not in accepted_indices:
            progress["city_status"][city_index] = "needs_reselection"

    print("Final study city segment selection")
    print("==================================")
    print("staged selector version:", STAGED_SELECTOR_VERSION)
    print("candidate file:", JOBS_FILE)
    print("candidate windows:", len(jobs))
    print("mapped cities:", mapped_city_count)
    print("processing city offsets:", start_city, "to", last_city_index - 1)

    for city_index in selected_city_indices:
        city_jobs = groups_by_index.get(city_index, [])
        city_info = metadata.get(city_index, {})
        if city_jobs:
            first = city_jobs[0]
            label = f"{first['city']}, {first['country']}"
        else:
            label = f"{city_info.get('city', 'Unknown')}, {city_info.get('country', 'Unknown')}"

        if int(city_index) in accepted_indices:
            print("SKIP already validated:", label)
            continue

        if not city_jobs:
            progress["city_status"][str(city_index)] = "no_local_candidate"
            save_progress(progress)
            print("NO LOCAL CANDIDATE:", label)
            continue

        print("\nCITY", city_index, label, "candidates:", len(city_jobs))
        accepted = False
        for candidate_number, job in enumerate(city_jobs, start=1):
            if str(job["job_id"]) in rejected_job_ids:
                passed, _reason, _render, _geometry = acceptance_evidence(job)
                if not passed:
                    print(
                        "SKIP previously rejected candidate:",
                        job["video_id"],
                        "start",
                        job["segment_start_time_s"],
                    )
                    continue

            print(
                "Candidate",
                f"{candidate_number}/{len(city_jobs)}",
                job["video_id"],
                "start",
                job["segment_start_time_s"],
            )
            passed, return_code, reason = run_candidate(job)
            progress["attempts"].append(
                {
                    "job_id": job["job_id"],
                    "city_index": int(city_index),
                    "video_id": job["video_id"],
                    "segment_start_time_s": float(job["segment_start_time_s"]),
                    "attempted_at": utc_now(),
                    "return_code": return_code,
                    "accepted": passed,
                    "acceptance_reason": reason,
                }
            )
            save_progress(progress)
            if not passed:
                rejected_job_ids.add(str(job["job_id"]))
                print("Candidate not accepted:", reason)
                continue

            render, geometry = accepted_outputs(job)
            assert render is not None and geometry is not None
            manifest["segments"].append(segment_record(job, render, geometry))
            manifest["segments"].sort(key=lambda item: int(item["city_index"]))
            manifest["updated_at"] = utc_now()
            write_json_atomic(FINAL_MANIFEST, manifest)
            accepted_indices.add(int(city_index))
            progress["city_status"][str(city_index)] = "validated_visual_explanation"
            save_progress(progress)
            print("ACCEPTED:", relative_path(render))
            accepted = True
            break

        if not accepted:
            completed = progress["completed_video_ids_by_city"].setdefault(
                str(city_index),
                [],
            )
            for video_id in dict.fromkeys(
                str(job.get("video_id", "")).strip() for job in city_jobs
            ):
                if video_id and video_id not in completed:
                    completed.append(video_id)
            progress["city_status"][str(city_index)] = "no_candidate_accepted"
            save_progress(progress)
            print("NO ACCEPTED SEGMENT:", label)

    stage_rows = prepare_next_video_stage(
        city_summaries,
        accepted_indices,
        progress,
    )
    save_progress(progress)

    print("\nBatch complete")
    print("accepted segments in manifest:", len(manifest["segments"]))
    print("next videos prepared:", len(stage_rows))
    print("manifest:", FINAL_MANIFEST)
    print("progress:", PROGRESS_FILE)
    print("next video stage:", NEXT_VIDEO_STAGE_FILE)


if __name__ == "__main__":
    main()
