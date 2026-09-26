"""CPU only guards for final study segment acceptance evidence."""

import json
import os
import sys
import tempfile
from pathlib import Path


SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")

if SRC not in sys.path:
    sys.path.insert(0, SRC)

import _paths  # noqa: E402,F401  (every src/<group>/ onto sys.path)

import run_final_study_segment_selection as selector  # noqa: E402


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def acceptance_fixture():
    root = Path(tempfile.mkdtemp())
    selector.PROJECT_ROOT = root
    selector.WORKFLOW_OUTPUTS = root / "workflow_outputs"
    selector.FINAL_RENDER_DIR = selector.WORKFLOW_OUTPUTS / "final_renders"
    selector.GEOMETRY_DIR = selector.WORKFLOW_OUTPUTS / "overlay_geometry"
    selector.GATE_DIR = selector.WORKFLOW_OUTPUTS / "gemma_reasoning"
    selector.MIRAGE_DIR = selector.WORKFLOW_OUTPUTS / "mirage"
    selector.FINAL_MANIFEST = selector.WORKFLOW_OUTPUTS / "final_study_segments.json"

    job = {"video_id": "videoX", "segment_start_time_s": 120.0}
    tag = selector.clip_tag(job)
    render = selector.FINAL_RENDER_DIR / f"{tag}_preview_vehicles.mp4"
    geometry = selector.GEOMETRY_DIR / f"{tag}_geometry.jsonl.gz"
    render.parent.mkdir(parents=True, exist_ok=True)
    geometry.parent.mkdir(parents=True, exist_ok=True)
    render.write_bytes(b"video")
    geometry.write_bytes(b"geometry")
    write_json(
        selector.GATE_DIR / f"{tag}_gemma_gate.json",
        {"proper_time_to_explain": True, "decision": "explain_now"},
    )
    write_json(
        selector.MIRAGE_DIR / f"{tag}_effect_plan.json",
        {
            "explanation_policy": "render_subtle_contextual_explanation",
            "display_target": "ego_future_path",
            "mirage_effect_family": "trajectory",
            "visual_layers": [{"type": "trajectory"}],
        },
    )
    return job, render, geometry


def test_acceptance_requires_gate_plan_layers_render_and_geometry():
    job, render, geometry = acceptance_fixture()

    passed, reason, found_render, found_geometry = selector.acceptance_evidence(job)

    assert passed is True
    assert reason == "valid_visual_explanation"
    assert found_render == render
    assert found_geometry == geometry


def test_gate_yes_with_do_not_render_plan_is_rejected():
    job, _render, _geometry = acceptance_fixture()
    tag = selector.clip_tag(job)
    write_json(
        selector.MIRAGE_DIR / f"{tag}_effect_plan.json",
        {
            "explanation_policy": "do_not_render",
            "display_target": "ego_future_path",
            "mirage_effect_family": "none",
            "visual_layers": [],
        },
    )

    passed, reason, _found_render, _found_geometry = selector.acceptance_evidence(job)

    assert passed is False
    assert reason == "mirage_policy_does_not_render"


def test_revalidation_removes_legacy_file_only_acceptance():
    job, render, geometry = acceptance_fixture()
    tag = selector.clip_tag(job)
    write_json(
        selector.MIRAGE_DIR / f"{tag}_effect_plan.json",
        {
            "explanation_policy": "do_not_render",
            "display_target": "ego_future_path",
            "mirage_effect_family": "none",
            "visual_layers": [],
        },
    )
    manifest = {
        "segments": [
            {
                "city_id": "city001",
                "city_index": 1,
                "city": "Example City",
                "source_video_id": job["video_id"],
                "start_seconds": job["segment_start_time_s"],
                "reference_render": selector.relative_path(render),
                "geometry": selector.relative_path(geometry),
            }
        ]
    }

    retained = selector.revalidate_manifest(manifest)

    assert retained == 0
    assert manifest["segments"] == []
    assert len(manifest["invalidated_segments"]) == 1
    assert (
        manifest["invalidated_segments"][0]["invalidation_reason"]
        == "mirage_policy_does_not_render"
    )


if __name__ == "__main__":
    failures = 0

    for name, test in sorted(globals().items()):
        if not name.startswith("test_") or not callable(test):
            continue

        try:
            test()
            print("PASS  %s" % name)
        except AssertionError as error:
            failures += 1
            print("FAIL  %s\n      %s" % (name, error))

    raise SystemExit(1 if failures else 0)
