"""Tests for the isolated explicit-window validation manifest."""

import os
import sys
import tempfile


SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")

if SRC not in sys.path:
    sys.path.insert(0, SRC)

import run_validation_window


def test_validation_job_is_isolated_and_exact():
    directory = tempfile.mkdtemp()
    mapping_path = os.path.join(directory, "mapping.csv")
    videos_dir = os.path.join(directory, "videos")
    workflow_dir = os.path.join(directory, "workflow")
    alpamayo_dir = os.path.join(directory, "alpamayo")

    with open(mapping_path, "w", encoding="utf-8", newline="") as handle:
        handle.write("locality,country,continent,videos\n")
        handle.write('Toronto,Canada,North America,"[3ai7SUaPoHM]"\n')

    original = {
        name: os.environ.get(name)
        for name in (
            "OPTICARVIS_MAPPING_CSV",
            "OPTICARVIS_VIDEOS_DIR",
            "OPTICARVIS_WORKFLOW_OUTPUTS",
            "OPTICARVIS_ALPAMAYO_OUTPUTS",
        )
    }

    try:
        os.environ["OPTICARVIS_MAPPING_CSV"] = mapping_path
        os.environ["OPTICARVIS_VIDEOS_DIR"] = videos_dir
        os.environ["OPTICARVIS_WORKFLOW_OUTPUTS"] = workflow_dir
        os.environ["OPTICARVIS_ALPAMAYO_OUTPUTS"] = alpamayo_dir
        job, jobs_path, master_path = run_validation_window.validation_job(
            "3ai7SUaPoHM",
            15.0,
            30.0,
        )
    finally:
        for name, value in original.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    assert job["video_id"] == "3ai7SUaPoHM"
    assert job["segment_start_time_s"] == 15.0
    assert job["segment_end_time_s"] == 45.0
    assert job["selection_method"] == "manual_validation"
    assert job["city"] == "Toronto"
    assert jobs_path.startswith(workflow_dir)
    assert master_path.startswith(workflow_dir)


if __name__ == "__main__":
    test_validation_job_is_isolated_and_exact()
    print("PASS  test_validation_job_is_isolated_and_exact")
