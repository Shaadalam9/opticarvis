"""Adapter between OptiCarVis clip jobs and NVIDIA Alpamayo2 Super.

batch_corrected_pipeline.py drives this backend with its own CLI:

    <python> src/alpamayo2_super_adapter.py --jobs-jsonl <jobs.jsonl>
             --output-dir <dir> --model-id <hf-id> [extra args...]

The inference itself is NOT reimplemented here. scripts/alpamayo2_super_wrapper.py
already holds it -- planar-VO ego history, the single-camera bypass around
select_task_input(), the expert attention-mask dtype fix, and the exact output
serialisation workflow_runner.py parses. That file is loaded as a module and
called, so the two entry points cannot drift apart: this one is a translation
layer from clip jobs to the wrapper's clip dicts, nothing more.

It runs under the *planner's* interpreter (ALPAMAYO2_SUPER_PYTHON), not the
OptiCarVis venv, so like the wrapper it imports nothing from src/ and depends on
the standard library alone.

Output, one file per job, written straight to the job's "alpamayo_json" -- the
final path the renderer expects, so no copy step follows:

    <alpamayo_json>
    -> video_id, segment_start_time_s, clip_video, run_meta
    -> result   : serialised 3-tuple
    -> pred_xyz : [1, 1, 1, 64, 3]
    -> pred_rot : [1, 1, 1, 64, 3, 3]
    -> extra    : dict whose values.cot holds the Chain-of-Causation text

Validate the plumbing without a GPU or the checkpoint:

    python src/alpamayo2_super_adapter.py --jobs-jsonl j.jsonl --output-dir out \
        --model-id nvidia/Alpamayo2-Super --self-test
"""

import argparse
import importlib.util
import json
import os
import sys
import time


# Matches WHEN_START_LOCAL_S in batch_corrected_pipeline.py, used when a job
# carries no explicit t0. The wrapper clamps it up if the history window does
# not fit, so this only has to be a sane default.
DEFAULT_T0_LOCAL_S = 12.67


def load_wrapper():
    """Import scripts/alpamayo2_super_wrapper.py as a module.

    Located relative to this file rather than the working directory: the batch
    runner sets cwd to the project root, but nothing guarantees that.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.normpath(os.path.join(here, "..", "scripts", "alpamayo2_super_wrapper.py"))

    if not os.path.isfile(path):
        print("Missing planner wrapper:", path)
        raise SystemExit(1)

    spec = importlib.util.spec_from_file_location("alpamayo2_super_wrapper", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["alpamayo2_super_wrapper"] = module
    spec.loader.exec_module(module)

    return module


def read_jsonl(path):
    rows = []

    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()

            if text:
                rows.append(json.loads(text))

    return rows


def job_output_path(job, output_dir):
    configured_path = str(job.get("alpamayo_json", "")).strip()

    if configured_path:
        return configured_path

    video_id = str(job.get("video_id"))
    start_time = int(round(float(job.get("segment_start_time_s", 0))))

    return os.path.join(output_dir, video_id + "_" + str(start_time) + "_alpamayo.json")


def clip_from_job(job):
    """Translate a clip job into the clip dict read_manifest() would produce."""
    start = float(job.get("segment_start_time_s") or 0.0)

    # The renderer derives artefact names from <video_id>_<segment start>, so the
    # tag has to carry the start time -- a bare video_id would collide across
    # every clip cut from the same source video.
    video_id = str(job.get("video_id", ""))
    tag = video_id + "_" + str(int(round(start)))

    return {
        "video_id": tag,
        "segment_start_time_s": start,
        "clip_video": os.path.abspath(str(job.get("clip_video", ""))),
        "when_start_local_s": float(job.get("when_start_local_s") or DEFAULT_T0_LOCAL_S),
        "when_end_local_s": float(job.get("when_end_local_s") or 0.0),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run Alpamayo2-Super over an OptiCarVis clip job list.",
    )
    parser.add_argument("--jobs-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-id", required=True)

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--diffusion-steps", type=int, default=10)
    parser.add_argument("--num-traj-samples", type=int, default=1)
    parser.add_argument("--top-p", type=float, default=0.98)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--t0-local-s", type=float, default=None)
    parser.add_argument("--allow-stationary-ego-history", action="store_true")
    parser.add_argument("--self-test", action="store_true")

    # OPTICARVIS_ALPAMAYO_EXTRA_ARGS is forwarded verbatim, so tolerate flags
    # meant for a different backend rather than dying on them.
    args, unknown = parser.parse_known_args(argv)

    if unknown:
        print("Ignoring unrecognised arguments:", " ".join(unknown))

    # process_clip()/build_run_meta() read the wrapper's own argument names.
    args.model = args.model_id
    args.config = ""

    return args


def main():
    args = parse_args()
    wrapper = load_wrapper()

    jobs = read_jsonl(args.jobs_jsonl)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print("")
    print("Alpamayo2 Super adapter")
    print("=======================")
    print("jobs:       ", len(jobs))
    print("model_id:   ", "SELF TEST (model not loaded)" if args.self_test else args.model_id)
    print("output_dir: ", output_dir)
    print("wrapper:    ", wrapper.__file__)
    print("cameras:     1 (front wide, id %d) x %d frames"
          % (wrapper.FRONT_WIDE_CAMERA_ID, wrapper.NUM_CONTEXT_FRAMES))
    print("")

    if not jobs:
        print("Job list contained no rows:", args.jobs_jsonl)
        return 1

    model = None if args.self_test else wrapper.load_model(args)

    written = failed = 0

    for index, job in enumerate(jobs, start=1):
        clip = clip_from_job(job)
        path = job_output_path(job, output_dir)

        print("[%d/%d] %s" % (index, len(jobs), clip["video_id"]))
        started = time.time()

        try:
            pred_xyz, pred_rot, extra, t0_s, ego_frames, serialise = wrapper.process_clip(
                clip, args, model
            )
            elapsed = time.time() - started

            wrapper.write_clip_json(
                path,
                clip,
                wrapper.build_run_meta(args, clip, t0_s, ego_frames, elapsed),
                pred_xyz,
                pred_rot,
                extra,
                serialise,
            )
            print("    wrote %s (%.1f s)" % (path, elapsed))
            written += 1

        except Exception as error:
            # One bad clip must not abandon a 100 city batch; the batch runner
            # reports any job whose JSON never appeared.
            print("    FAILED:", type(error).__name__, str(error)[:400])
            failed += 1

    print("")
    print("written:", written, "| failed:", failed)

    return 1 if failed and not written else 0


if __name__ == "__main__":
    raise SystemExit(main())
