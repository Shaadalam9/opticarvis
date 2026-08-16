r"""Guard the two job-fanout contracts that fail silently.

Neither of these announces itself when it breaks, which is why they are pinned
here rather than left to a batch run to discover:

  * The batch runner hands each clip its start time through an environment
    variable, and the child re-derives every artefact name from it. When the two
    sides disagree on the variable's name the child does not error -- it keeps
    the 4630.0 default, so every clip of a video writes to the same STATE_JSON
    and the master index records state_available False for all of them. A rename
    on either side reintroduces that, so the contract is asserted directly.

  * One rendered video per city is the deliverable. The footage budget accrues
    STRIDE_S per clip rather than CLIP_LENGTH_S, so leaning on it gives 60 clips
    per city (~6000 for mapping.csv) -- a plausible-looking number that is two
    orders of magnitude too much render time. CLIPS_PER_CITY is the real cap.

Runs without a GPU, a clip, a model, or a `config` file. Standalone:

    python tests/test_job_naming.py

or under pytest:

    pytest tests/
"""

import csv
import importlib
import os
import re
import sys
import tempfile

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")

if SRC not in sys.path:
    sys.path.insert(0, SRC)


def reload_with_env(module_name, env):
    """Import a module fresh under a temporary environment."""
    saved = {key: os.environ.get(key) for key in env}

    try:
        for key, value in env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = str(value)

        module = importlib.import_module(module_name)

        return importlib.reload(module)
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def segment_start_env_name_written_by_batch():
    """The variable name batch_corrected_pipeline.job_environment actually sets.

    Read from source rather than by import: batch_corrected_pipeline pulls in
    common.get_configs at import time, which sys.exit(1)s without a `config`
    file, and this test must run on a bare checkout.
    """
    path = os.path.join(SRC, "batch_corrected_pipeline.py")

    with open(path, "r", encoding="utf-8") as handle:
        source = handle.read()

    names = re.findall(r'env\[\s*"(OPTICARVIS_SEGMENT_START[A-Z_]*)"\s*\]', source)

    assert names, "job_environment no longer exports any OPTICARVIS_SEGMENT_START* variable"

    return names[0]


def test_batch_segment_start_reaches_the_child():
    """The name the batch exports must be one pipeline_common reads."""
    name = segment_start_env_name_written_by_batch()

    common = reload_with_env(
        "pipeline_common",
        {
            "OPTICARVIS_VIDEO_ID": "vid",
            "OPTICARVIS_SEGMENT_START_S": None,
            "OPTICARVIS_SEGMENT_START_TIME_S": None,
            name: "1234",
        },
    )

    assert common.SEGMENT_START_TIME_S == 1234.0, (
        "batch_corrected_pipeline exports %s but pipeline_common ignored it and "
        "fell back to %r -- every clip of a video would share one artefact name"
        % (name, common.SEGMENT_START_TIME_S)
    )
    assert common.segment_tag() == "vid_1234"


def test_legacy_segment_start_name_still_honoured():
    """The pre-rename variable keeps working, and the canonical one wins."""
    legacy = reload_with_env(
        "pipeline_common",
        {
            "OPTICARVIS_VIDEO_ID": "vid",
            "OPTICARVIS_SEGMENT_START_S": None,
            "OPTICARVIS_SEGMENT_START_TIME_S": "1056",
        },
    )
    assert legacy.SEGMENT_START_TIME_S == 1056.0

    both = reload_with_env(
        "pipeline_common",
        {
            "OPTICARVIS_VIDEO_ID": "vid",
            "OPTICARVIS_SEGMENT_START_S": "36",
            "OPTICARVIS_SEGMENT_START_TIME_S": "1056",
        },
    )
    assert both.SEGMENT_START_TIME_S == 36.0, "canonical name must win over the alias"


def test_adapter_accepts_the_flags_the_batch_passes():
    """run_alpamayo2_super_batch and the adapter must agree on the CLI.

    They are separated by a subprocess boundary and by two different
    interpreters, so a flag rename on either side only shows up as a failed
    batch after the model has already loaded.
    """
    batch_path = os.path.join(SRC, "batch_corrected_pipeline.py")

    with open(batch_path, "r", encoding="utf-8") as handle:
        source = handle.read()

    start = source.index("def run_alpamayo2_super_batch")
    body = source[start:source.index("\ndef ", start + 1)]
    passed = set(re.findall(r'"(--[a-z0-9-]+)"', body))

    assert passed, "run_alpamayo2_super_batch no longer passes any flags"

    adapter = importlib.import_module("alpamayo2_super_adapter")
    parsed = adapter.parse_args(
        ["--jobs-jsonl", "j", "--output-dir", "o", "--model-id", "m"]
    )

    assert parsed.model == "m", "--model-id must reach the wrapper as .model"
    assert parsed.jobs_jsonl == "j"
    assert parsed.output_dir == "o"

    for flag in passed:
        # argparse turns --jobs-jsonl into jobs_jsonl
        attribute = flag.lstrip("-").replace("-", "_")
        assert hasattr(parsed, attribute), (
            "batch passes %s but the adapter has no matching argument" % flag
        )


def test_adapter_tags_clips_by_video_and_start():
    """Two clips of one video must not collapse onto the same artefact tag."""
    adapter = importlib.import_module("alpamayo2_super_adapter")

    first = adapter.clip_from_job(
        {"video_id": "vid", "segment_start_time_s": 36.0, "clip_video": "a.mp4"}
    )
    second = adapter.clip_from_job(
        {"video_id": "vid", "segment_start_time_s": 96.0, "clip_video": "b.mp4"}
    )

    assert first["video_id"] == "vid_36"
    assert second["video_id"] == "vid_96"
    assert first["video_id"] != second["video_id"]


def test_one_failed_job_does_not_abort_the_batch():
    """A failing render must be recorded, not thrown, unless asked otherwise.

    Read from source: batch_corrected_pipeline pulls in common.get_configs at
    import time, which sys.exit(1)s without a `config` file, so this test cannot
    import it on a bare checkout.

    The regression is cheap to reintroduce and expensive to discover -- it costs
    a whole batch, hours in, and only on the clips that fail.
    """
    path = os.path.join(SRC, "batch_corrected_pipeline.py")

    with open(path, "r", encoding="utf-8") as handle:
        source = handle.read()

    start = source.index("def run_pipeline_one_job")
    body = source[start:source.index("\ndef ", start + 1)]

    assert "STOP_ON_JOB_FAILURE" in body, (
        "run_pipeline_one_job must gate its abort on STOP_ON_JOB_FAILURE"
    )

    for line in body.splitlines():
        stripped = line.strip()

        if stripped.startswith("raise SystemExit"):
            assert "STOP_ON_JOB_FAILURE" in body[: body.index(stripped)][-400:], (
                "run_pipeline_one_job raises SystemExit outside the "
                "STOP_ON_JOB_FAILURE guard -- one bad clip would abort the batch"
            )

    assert os.environ.get("OPTICARVIS_STOP_ON_JOB_FAILURE") is None, (
        "this test assumes the flag is unset by default"
    )
    assert '"OPTICARVIS_STOP_ON_JOB_FAILURE", "0"' in source, (
        "STOP_ON_JOB_FAILURE must default to off"
    )


def test_gate_provenance_is_derived_not_asserted():
    """decided_by must follow model_called, never be hardcoded.

    call_gemma4_gate() falls back to the heuristic whenever the model will not
    load -- a missing CUDA compile toolchain is enough, and it only prints. A
    hardcoded gemma4_gate made every state file claim a decision the model never
    made, which is the field an analysis of explanation timing would trust.
    """
    path = os.path.join(SRC, "gemma_reasoning_module.py")

    with open(path, "r", encoding="utf-8") as handle:
        source = handle.read()

    start = source.index('state["explanation"] = {')
    block = source[start:source.index("}", start)]

    assert '"decided_by": "gemma4_gate",' not in block, (
        "decided_by is hardcoded again; a heuristic decision would be recorded "
        "as a Gemma one"
    )
    assert "model_called" in block, "decided_by must be derived from model_called"
    assert "heuristic_gate" in block, "the heuristic path needs its own label"


def test_gate_fallback_can_be_made_fatal():
    """A batch must be able to refuse the silent downgrade to the heuristic."""
    path = os.path.join(SRC, "gemma_reasoning_module.py")

    with open(path, "r", encoding="utf-8") as handle:
        source = handle.read()

    assert '"OPTICARVIS_REQUIRE_GEMMA_GATE", "0"' in source, (
        "REQUIRE_GEMMA_GATE must exist and default to off"
    )
    assert "if REQUIRE_GEMMA_GATE:" in source, (
        "the fallback path must honour REQUIRE_GEMMA_GATE"
    )


def test_declined_clip_is_not_counted_as_rendered():
    """The three outcomes must stay distinct in the batch tally.

    The per-clip pipeline exits 0 both when it renders and when the gate
    declines, so counting on the return code alone overstates what a batch
    produced.
    """
    path = os.path.join(SRC, "batch_corrected_pipeline.py")

    with open(path, "r", encoding="utf-8") as handle:
        source = handle.read()

    start = source.index("def run_pipeline_one_job")
    body = source[start:source.index("\ndef ", start + 1)]

    for outcome in ('"rendered"', '"gate_declined"', '"failed"'):
        assert "return " + outcome in body, (
            "run_pipeline_one_job must return %s" % outcome
        )

    assert "job_produced_render(job)" in body, (
        "the rendered/declined split must check for an actual video, not the "
        "subprocess return code"
    )


def test_batch_exit_code_survives_a_partial_run():
    """main() may only exit non zero when nothing rendered at all.

    main.py runs the chunks with check=True, so exiting non zero on a partial
    batch would abort every later chunk and reintroduce the same failure one
    level up.
    """
    path = os.path.join(SRC, "batch_corrected_pipeline.py")

    with open(path, "r", encoding="utf-8") as handle:
        source = handle.read()

    body = source[source.index("\ndef main("):]

    assert 'len(outcomes["failed"]) == len(ready_jobs)' in body, (
        "main() must only fail the run when every job failed; a partial batch, "
        "or one the gate declined in full, would otherwise abort later chunks "
        "under main.py"
    )


def write_mapping(path, cities):
    """A mapping.csv with only the columns clip_job_builder reads."""
    header = [
        "id", "locality", "locality_aka", "state", "country", "iso3", "continent",
        "lat", "lon", "traffic_mortality", "traffic_index",
        "videos", "time_of_day", "start_time", "end_time",
    ]

    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)

        for index in range(cities):
            writer.writerow(
                [
                    index + 1, "city%d" % index, "[]", "", "country%d" % index, "AAA",
                    "Europe", "0", "0", "0", "0",
                    "[vid%d]" % index, "[[0]]", "[[0]]", "[[3600]]",
                ]
            )


def build_jobs(clips_per_city, cities=3):
    directory = tempfile.mkdtemp()
    mapping = os.path.join(directory, "mapping.csv")
    write_mapping(mapping, cities)

    builder = reload_with_env(
        "clip_job_builder",
        {
            "OPTICARVIS_CLIPS_PER_CITY": clips_per_city,
            "OPTICARVIS_CLIP_JOBS": os.path.join(directory, "jobs.jsonl"),
            "OPTICARVIS_CLIP_JOBS_SUMMARY": os.path.join(directory, "summary.json"),
        },
    )

    jobs = []

    with open(mapping, "r", encoding="utf-8-sig") as handle:
        for index, row in enumerate(csv.DictReader(handle), start=1):
            city_jobs, _intervals = builder.build_jobs_for_city(row, index)
            jobs.append(city_jobs)

    return jobs


def test_one_clip_per_city_by_default():
    for city_jobs in build_jobs(clips_per_city=1):
        assert len(city_jobs) == 1, "expected one clip per city, got %d" % len(city_jobs)


def test_clips_per_city_cap_is_respected():
    for city_jobs in build_jobs(clips_per_city=3):
        assert len(city_jobs) == 3, "expected three clips per city, got %d" % len(city_jobs)


def test_zero_means_uncapped():
    """0 restores the old footage-budget-only behaviour."""
    for city_jobs in build_jobs(clips_per_city=0):
        assert len(city_jobs) > 1, (
            "CLIPS_PER_CITY=0 must fall back to the footage budget, got %d clip(s)"
            % len(city_jobs)
        )


def test_clips_within_a_city_get_distinct_artefact_names():
    """Two clips of one video must not collide on clip_video / alpamayo_json."""
    for city_jobs in build_jobs(clips_per_city=3):
        starts = [job["segment_start_time_s"] for job in city_jobs]
        assert len(set(starts)) == len(starts), "duplicate segment starts: %r" % starts

        for key in ("clip_video", "alpamayo_json", "job_id"):
            values = [job[key] for job in city_jobs]
            assert len(set(values)) == len(values), "duplicate %s: %r" % (key, values)


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
