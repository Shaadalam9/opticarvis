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


def batch_source():
    path = os.path.join(SRC, "batch_corrected_pipeline.py")

    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def test_round_loop_stops_a_city_after_a_render():
    """The window scan must resolve a city on its first rendered window."""
    source = batch_source()
    body = source[source.index("\ndef main("):]

    assert "group_jobs_by_city" in body, "main() must group windows by city"
    assert 'city_results[city_key] = "rendered"' in body, (
        "a rendered window must resolve its city so later windows are skipped"
    )
    assert '"no_window_fired"' in body, (
        "a city whose windows are exhausted must be resolved, not retried forever"
    )


def test_downloaded_sources_are_reclaimed():
    """DELETE_FTP_VIDEOS_AFTER_USE was configured but never implemented."""
    source = batch_source()

    assert "register_downloaded_source(local_path)" in source, (
        "downloads must be registered for cleanup -- durably, since the "
        "prefetch script and an aborted run download in other processes"
    )
    assert "def cleanup_resolved_sources" in source
    assert "cleanup_downloaded_sources_final(" in source[source.index("\ndef main("):], (
        "main() must sweep leftovers at the end of the run"
    )
    assert "if not DELETE_FTP_VIDEOS_AFTER_USE:" in source, (
        "cleanup must respect the config flag"
    )
    assert "if local_path:" in source, (
        "an empty download returns None from save_video_response and must not "
        "be registered or cached"
    )


def test_systemic_failure_guard_exists():
    """N consecutive failures with no success between must stop the batch."""
    source = batch_source()

    assert '"OPTICARVIS_MAX_CONSECUTIVE_FAILURES", "5"' in source
    assert "consecutive_failures[0] >= MAX_CONSECUTIVE_FAILURES" in source

    # The streak must reset in BOTH success branches of the pipeline loop.
    # Checking for one bare reset passed even with the resets deleted, since
    # the initialisation matched the same substring.
    body = source[source.index("\ndef main("):]
    rendered_branch = body[body.index('result == "rendered"'):]
    assert "consecutive_failures[0] = 0" in rendered_branch[:400], (
        "a rendered clip must reset the failure streak"
    )
    declined_branch = body[body.index('result == "gate_declined"'):]
    assert "consecutive_failures[0] = 0" in declined_branch[:400], (
        "a gate decline is a success for the streak and must reset it"
    )

    # Prepare-stage failures must feed the same guard: a down FTP server or a
    # broken ffmpeg fails every window in prepare, and a guard that only
    # watches the pipeline loop never fires.
    prepare_loop = body[body.index("status = prepare_one_job"):]
    assert "note_failure(" in prepare_loop[:400], (
        "prepare failures must count towards the systemic-failure streak"
    )


def test_undecided_gate_is_not_a_decline():
    """explanation.needed=None (stage 1's 'pending') must read as undecided.

    A gate that crashes between the stages leaves needed=None in the state;
    bool(None) recorded every such crash as a legitimate decline.
    """
    source = batch_source()
    start = source.index("def gate_decisions_for_jobs")
    body = source[start:source.index("\ndef ", start + 1)]

    assert 'explanation.get("needed") in (True, False)' in body, (
        "the readback must require a real boolean decision"
    )
    assert '"explain_now"' in body and '"do_not_explain"' in body, (
        "the readback must also require a decided status"
    )

    main_body = source[source.index("\ndef main("):]
    assert "no_gate_decision" in main_body, (
        "an undecided job must be recorded as failed, not fall back to the "
        "full per-job pipeline"
    )


def test_city_render_target_and_credit():
    """CLIPS_PER_CITY is a per-city render target, and prior renders count.

    Resolving a city on its first render broke CLIPS_PER_CITY>1 and =0, and a
    resumed run re-rendered cities whose video already existed.
    """
    source = batch_source()
    body = source[source.index("\ndef main("):]

    assert "clips_wanted" in body and "city_render_counts" in body, (
        "cities must resolve against a render target, not a boolean"
    )
    assert "credit_render(" in body
    assert "job_produced_render(job)" in body, (
        "renders from earlier invocations must be credited before the rounds"
    )
    # exhausted-with-renders is still a rendered city, not an unfired one
    assert '"rendered" if city_render_counts.get(city_key) else "no_window_fired"' in body


def test_launcher_refuses_to_start_with_assets_missing():
    """The batch launcher must gate on the asset check.

    The failure this prevents is silent: without the UFLDv2 lane model every
    ribbon renders straight, and a 100 city batch completes looking plausibly
    wrong. A crash would have been caught; this would not have been.
    """
    path = os.path.join(SRC, "..", "scripts", "run_100_cities.sh")

    with open(path, "r", encoding="utf-8") as handle:
        source = handle.read()

    assert "setup_assets.py --check-only" in source, (
        "run_100_cities.sh must run the asset check before launching"
    )
    assert "exit 1" in source.split("setup_assets.py --check-only", 1)[1][:400], (
        "a failed asset check must stop the launch, not just warn"
    )

    setup_path = os.path.join(SRC, "..", "scripts", "setup_assets.py")

    with open(setup_path, "r", encoding="utf-8") as handle:
        setup_source = handle.read()

    assert "STRAIGHT" in setup_source, (
        "the UFLDv2 failure message must say what actually goes wrong -- a "
        "straight ribbon -- not just that a file is missing"
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

    assert 'len(outcomes["failed"]) == genuinely_attempted' in body, (
        "main() must only fail the run when every genuinely attempted window "
        "failed; a partial batch, a fully skipped resume, or one the gate "
        "declined in full, would otherwise abort later chunks under main.py"
    )
    assert 'attempted - len(outcomes["skipped"])' in body, (
        "intentional skips must not count towards the failure exit rule"
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
            "OPTICARVIS_WINDOWS_PER_CITY": None,
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


def test_windows_per_city_emits_ordered_candidates():
    """WINDOWS_PER_CITY candidates, stride apart, indexed in order."""
    directory = tempfile.mkdtemp()
    mapping = os.path.join(directory, "mapping.csv")
    write_mapping(mapping, 2)

    builder = reload_with_env(
        "clip_job_builder",
        {
            "OPTICARVIS_CLIPS_PER_CITY": "1",
            "OPTICARVIS_WINDOWS_PER_CITY": "4",
            "OPTICARVIS_CLIP_JOBS": os.path.join(directory, "jobs.jsonl"),
            "OPTICARVIS_CLIP_JOBS_SUMMARY": os.path.join(directory, "summary.json"),
        },
    )

    with open(mapping, "r", encoding="utf-8-sig") as handle:
        for index, row in enumerate(csv.DictReader(handle), start=1):
            city_jobs, _intervals = builder.build_jobs_for_city(row, index)

            assert len(city_jobs) == 4, "expected 4 windows, got %d" % len(city_jobs)
            assert [job["window_index"] for job in city_jobs] == [0, 1, 2, 3]

            starts = [job["segment_start_time_s"] for job in city_jobs]
            strides = [b - a for a, b in zip(starts, starts[1:])]
            assert all(s == builder.STRIDE_S for s in strides), (
                "windows must advance by the stride, got %r" % strides
            )


def test_windows_default_matches_clips_cap():
    """Without OPTICARVIS_WINDOWS_PER_CITY nothing changes: one job per city."""
    for city_jobs in build_jobs(clips_per_city=1):
        assert len(city_jobs) == 1


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
