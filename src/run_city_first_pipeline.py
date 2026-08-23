import json
import os
import subprocess
import sys
from collections import OrderedDict


SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import common


def normalise_path(path_value):
    return os.path.abspath(path_value).replace("\\", os.sep)


def resolve_project_path(path_value):
    if path_value is None:
        return None

    path_text = str(path_value).strip()

    if os.path.isabs(path_text):
        return normalise_path(path_text)

    return normalise_path(os.path.join(PROJECT_ROOT, path_text))


def config_value(key, default):
    configs = common.get_configs()

    if isinstance(configs, dict) and key in configs:
        value = configs[key]

        if value is not None:
            return value

    return default


def config_int_value(key, default):
    value = config_value(key, default)

    try:
        return int(float(value))
    except Exception:
        return int(default)


WORKFLOW_OUTPUTS = resolve_project_path(
    config_value("workflow_outputs", "workflow_outputs")
)

JOBS_JSONL = resolve_project_path(
    config_value(
        "clip_jobs_jsonl",
        os.path.join(WORKFLOW_OUTPUTS, "clip_jobs.jsonl"),
    )
)

BATCH_SCRIPT = resolve_project_path(
    config_value(
        "batch_pipeline_script",
        "src/batch_corrected_pipeline.py",
    )
)

FINAL_RENDER_DIR = resolve_project_path(
    config_value(
        "final_render_dir",
        os.path.join(WORKFLOW_OUTPUTS, "final_renders"),
    )
)

DEFAULT_MAX_CITIES = config_int_value("CITY_RUNNER_MAX_CITIES", 0)


def load_jobs():
    jobs = []

    if not os.path.isfile(JOBS_JSONL):
        print("Missing clip jobs file:")
        print(JOBS_JSONL)
        raise SystemExit(1)

    with open(JOBS_JSONL, "r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            line = line.strip()

            if not line:
                continue

            job = json.loads(line)
            job["_job_index"] = index
            jobs.append(job)

    return jobs


def group_jobs_by_city(jobs):
    grouped = OrderedDict()

    for job in jobs:
        city_index = job.get("city_index")

        if city_index not in grouped:
            grouped[city_index] = []

        grouped[city_index].append(job)

    return grouped


def clip_tag(job):
    video_id = str(job.get("video_id", "")).strip()
    start_s = float(job.get("segment_start_time_s", 0.0))
    start_int = int(round(start_s))

    return video_id + "_" + str(start_int)


def final_render_for_job(job):
    tag = clip_tag(job)

    if not os.path.isdir(FINAL_RENDER_DIR):
        return None

    matched_paths = []

    for root, dirs, files in os.walk(FINAL_RENDER_DIR):
        for file_name in files:
            lower_name = file_name.lower()

            if not lower_name.endswith(".mp4"):
                continue

            if not file_name.startswith(tag + "_"):
                continue

            path = os.path.join(root, file_name)

            if os.path.isfile(path) and os.path.getsize(path) > 0:
                matched_paths.append(path)

    if not matched_paths:
        return None

    matched_paths.sort(key=lambda path: os.path.getmtime(path), reverse=True)

    return matched_paths[0]


def run_one_job(job):
    job_index = int(job["_job_index"])

    existing_render = final_render_for_job(job)

    if existing_render is not None:
        print("")
        print("Existing final render found:")
        print(existing_render)
        return True

    command = [
        sys.executable,
        BATCH_SCRIPT,
        "1",
        str(job_index),
    ]

    print("")
    print("=" * 80)
    print("Running job index:", job_index)
    print("=" * 80)

    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
    )

    render_path = final_render_for_job(job)

    if render_path is not None:
        print("")
        print("ACCEPTED FINAL RENDER:")
        print(render_path)
        return True

    if result.returncode != 0:
        print("")
        print("Job failed, no final render produced.")
        return False

    print("")
    print("No final render for this candidate. Trying next candidate from same city.")
    return False


def main():
    max_cities = DEFAULT_MAX_CITIES

    if len(sys.argv) >= 2:
        max_cities = int(sys.argv[1])

    jobs = load_jobs()
    grouped = group_jobs_by_city(jobs)

    print("")
    print("City first OptiCarVis search")
    print("===========================")
    print("project_root:", PROJECT_ROOT)
    print("jobs:", len(jobs))
    print("cities:", len(grouped))
    print("max_cities:", max_cities)
    print("jobs_file:", JOBS_JSONL)
    print("batch_script:", BATCH_SCRIPT)
    print("final_render_dir:", FINAL_RENDER_DIR)

    cities_checked = 0
    cities_with_render = 0
    cities_without_render = 0

    for city_index, city_jobs in grouped.items():
        if max_cities > 0 and cities_checked >= max_cities:
            break

        cities_checked += 1

        first_job = city_jobs[0]
        city = first_job.get("city", "unknown")
        country = first_job.get("country", "unknown")

        print("")
        print("#" * 80)
        print("Searching city:", city_index, city, country, "candidates:", len(city_jobs))
        print("#" * 80)

        found = False

        for candidate_number, job in enumerate(city_jobs, start=1):
            print("")
            print("Candidate", str(candidate_number) + "/" + str(len(city_jobs)))
            print("job_id:", job.get("job_id"))
            print("video_id:", job.get("video_id"))
            print("start_s:", job.get("segment_start_time_s"))

            accepted = run_one_job(job)

            if accepted:
                found = True
                cities_with_render += 1
                print("")
                print("FOUND RENDERED EXPLANATION CLIP FOR CITY:", city_index)
                break

        if not found:
            cities_without_render += 1
            print("")
            print("No rendered explanation clip found for city:", city_index)

    print("")
    print("City first search complete")
    print("==========================")
    print("cities_checked:", cities_checked)
    print("cities_with_render:", cities_with_render)
    print("cities_without_render:", cities_without_render)


if __name__ == "__main__":
    main()
