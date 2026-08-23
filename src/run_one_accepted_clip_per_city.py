import copy
import json
import os
import subprocess
import sys
import time
from collections import OrderedDict


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
WORKFLOW_OUTPUTS = os.path.join(PROJECT_ROOT, "workflow_outputs")
JOBS_FILE = os.path.join(WORKFLOW_OUTPUTS, "clip_jobs.jsonl")
JOBS_SUMMARY_FILE = os.path.join(WORKFLOW_OUTPUTS, "clip_jobs_summary.json")
GEMMA_DIR = os.path.join(WORKFLOW_OUTPUTS, "gemma_reasoning")

CLIP_JOB_BUILDER = os.path.join(SRC_DIR, "clip_job_builder.py")
BATCH_PIPELINE = os.path.join(SRC_DIR, "batch_corrected_pipeline.py")


def env_int(name, default):
    value = os.environ.get(name, "")
    if not value:
        return default
    return int(float(value))


def env_float(name, default):
    value = os.environ.get(name, "")
    if not value:
        return default
    return float(value)


CLIP_DURATION_S = env_float("OPTICARVIS_CLIP_DURATION_S", 30.0)

# Recenter offsets are tried around a rejected candidate before moving to the next candidate.
# Example with 30 s clip and 15 s stride:
# original 30 to 60, refined +8 gives 38 to 68.
REFINE_OFFSETS_S = [
    float(item.strip())
    for item in os.environ.get("OPTICARVIS_REFINE_OFFSETS_S", "5,10,-5,-10").split(",")
    if item.strip()
]

MAX_REFINES_PER_CANDIDATE = env_int("OPTICARVIS_MAX_REFINES_PER_CANDIDATE", 2)


def read_jsonl(file_name):
    rows = []

    if not os.path.exists(file_name):
        return rows

    with open(file_name, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    return rows


def append_jsonl(file_name, row):
    with open(file_name, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def city_key(job):
    for key in ["city_index", "city_id", "city", "city_name"]:
        value = job.get(key)
        if value is not None and str(value).strip():
            return str(value)

    job_id = str(job.get("job_id", "unknown_city"))
    parts = job_id.split("_")
    if len(parts) >= 3:
        return "_".join(parts[:3])

    return job_id


def grouped_jobs(jobs):
    groups = OrderedDict()

    for index, job in enumerate(jobs):
        if job.get("temporary_recenter_job"):
            continue

        key = city_key(job)
        if key not in groups:
            groups[key] = []
        groups[key].append((index, job))

    return groups


def rebuild_jobs_with_multiple_candidates():
    print("")
    print("Rebuilding clip jobs with multiple candidates per city")
    print("===================================================")

    for file_name in [JOBS_FILE, JOBS_SUMMARY_FILE]:
        if os.path.exists(file_name):
            os.remove(file_name)

    env = os.environ.copy()
    env["OPTICARVIS_ONE_CLIP_PER_CITY"] = "0"

    command = [sys.executable, CLIP_JOB_BUILDER]
    result = subprocess.run(command, cwd=PROJECT_ROOT, env=env)

    if result.returncode != 0:
        raise SystemExit("clip_job_builder.py failed")

    print("Rebuilt:", JOBS_FILE)


def ensure_multiple_candidates():
    jobs = read_jsonl(JOBS_FILE)

    if not jobs:
        rebuild_jobs_with_multiple_candidates()
        return read_jsonl(JOBS_FILE)

    groups = grouped_jobs(jobs)
    max_candidates = max(len(items) for items in groups.values()) if groups else 0

    if max_candidates <= 1:
        print("Current clip_jobs.jsonl has only one candidate per city.")
        rebuild_jobs_with_multiple_candidates()
        return read_jsonl(JOBS_FILE)

    print("Using existing candidate job file:", JOBS_FILE)
    print("total_jobs:", len(jobs))
    print("cities:", len(groups))
    print("max_candidates_per_city:", max_candidates)

    return jobs


def get_video_id(job):
    for key in ["video_id", "youtube_id", "source_video_id"]:
        value = job.get(key)
        if value:
            return str(value)

    job_id = str(job.get("job_id", ""))
    parts = job_id.split("_")
    if len(parts) >= 2:
        return parts[-2]

    return ""


def get_start_s(job):
    for key in ["segment_start_time_s", "start_time_s", "clip_start_s", "start_s"]:
        value = job.get(key)
        if value is not None:
            return float(value)

    job_id = str(job.get("job_id", ""))
    parts = job_id.split("_")
    if parts:
        try:
            return float(parts[-1])
        except ValueError:
            pass

    return None


def get_end_limit_s(job):
    for key in ["segment_end_time_s", "end_time_s", "clip_end_s", "end_s", "interval_end_s"]:
        value = job.get(key)
        if value is not None:
            return float(value)

    return None


def set_start_s(job, new_start_s):
    for key in ["segment_start_time_s", "start_time_s", "clip_start_s", "start_s"]:
        if key in job:
            job[key] = float(new_start_s)

    video_id = get_video_id(job)
    start_int = int(round(float(new_start_s)))

    old_job_id = str(job.get("job_id", "job"))
    parts = old_job_id.split("_")

    if len(parts) >= 2:
        parts[-1] = str(start_int)
        if video_id:
            parts[-2] = video_id
        job["job_id"] = "_".join(parts)
    elif video_id:
        job["job_id"] = video_id + "_" + str(start_int)
    else:
        job["job_id"] = old_job_id + "_recenter_" + str(start_int)

    # Common names used by downstream files.
    for key in ["clip_id", "clip_name", "video_id_with_start"]:
        if key in job:
            job[key] = video_id + "_" + str(start_int)

    return job


def make_recentered_job(job, offset_s):
    old_start_s = get_start_s(job)

    if old_start_s is None:
        return None

    new_start_s = old_start_s + float(offset_s)

    if new_start_s < 0:
        return None

    end_limit_s = get_end_limit_s(job)
    if end_limit_s is not None and new_start_s + CLIP_DURATION_S > end_limit_s:
        return None

    new_job = copy.deepcopy(job)
    new_job = set_start_s(new_job, new_start_s)
    new_job["temporary_recenter_job"] = True
    new_job["recenter_source_job_id"] = job.get("job_id", "")
    new_job["recenter_offset_s"] = float(offset_s)

    return new_job


def append_recentered_job(job):
    current_jobs = read_jsonl(JOBS_FILE)

    for existing in current_jobs:
        if existing.get("job_id") == job.get("job_id"):
            return len(current_jobs), None

    append_jsonl(JOBS_FILE, job)
    return len(current_jobs), job


def possible_gate_stems(job):
    stems = []

    for key in ["clip_id", "clip_name", "video_clip_id", "video_id_with_start"]:
        value = job.get(key)
        if value:
            stems.append(str(value))

    video_id = get_video_id(job)
    start_s = get_start_s(job)

    if video_id and start_s is not None:
        start_text = str(int(round(float(start_s))))
        stems.append(str(video_id) + "_" + start_text)

    job_id = str(job.get("job_id", ""))
    if job_id:
        parts = job_id.split("_")
        for start_index in range(len(parts)):
            suffix = "_".join(parts[start_index:])
            if suffix:
                stems.append(suffix)

    unique = []
    for stem in stems:
        if stem not in unique:
            unique.append(stem)

    return unique


def read_gate_file_for_job(job, started_at):
    candidates = []

    for stem in possible_gate_stems(job):
        file_name = os.path.join(GEMMA_DIR, stem + "_gemma_gate.json")
        if os.path.exists(file_name):
            candidates.append(file_name)

    if not candidates and os.path.exists(GEMMA_DIR):
        for name in os.listdir(GEMMA_DIR):
            if name.endswith("_gemma_gate.json"):
                file_name = os.path.join(GEMMA_DIR, name)
                if os.path.getmtime(file_name) >= started_at - 1.0:
                    candidates.append(file_name)

    if not candidates:
        return None, None

    candidates.sort(key=lambda path: os.path.getmtime(path), reverse=True)
    gate_file = candidates[0]

    with open(gate_file, "r", encoding="utf-8") as handle:
        return gate_file, json.load(handle)


def find_value(value, key):
    if isinstance(value, dict):
        if key in value:
            return value[key]
        for child in value.values():
            found = find_value(child, key)
            if found is not None:
                return found

    if isinstance(value, list):
        for child in value:
            found = find_value(child, key)
            if found is not None:
                return found

    return None


def run_single_candidate(job_index):
    env = os.environ.copy()
    env["OPTICARVIS_ONE_CLIP_PER_CITY"] = "0"

    command = [sys.executable, BATCH_PIPELINE, "1", str(job_index)]

    started_at = time.time()
    result = subprocess.run(command, cwd=SRC_DIR, env=env)

    return result.returncode, started_at


def candidate_is_interesting_enough_for_refine(gate_json):
    if not isinstance(gate_json, dict):
        return False

    text_parts = []

    for key in ["decision_reason", "passenger_facing_text", "display_target", "decision"]:
        value = find_value(gate_json, key)
        if value:
            text_parts.append(str(value).lower())

    text = " ".join(text_parts)

    negative_only = [
        "clear road",
        "lane keeping",
        "no relevant object",
        "no critical agent",
        "standard driving context",
    ]

    if any(item in text for item in negative_only):
        return False

    interesting_terms = [
        "pedestrian",
        "cyclist",
        "cross",
        "lead vehicle",
        "vehicle",
        "traffic",
        "yield",
        "stop",
        "slow",
        "speed bump",
        "merge",
        "fork",
        "occlusion",
        "parked",
        "sign",
        "light",
        "intersection",
        "distance",
    ]

    return any(item in text for item in interesting_terms)


def run_candidate_and_read_gate(job_index, job):
    return_code, started_at = run_single_candidate(job_index)
    gate_file, gate_json = read_gate_file_for_job(job, started_at)

    if gate_json is None:
        return return_code, gate_file, gate_json, None

    proper_time = find_value(gate_json, "proper_time_to_explain")
    return return_code, gate_file, gate_json, proper_time


def try_original_and_refined(job_index, job):
    print("")
    print("Original candidate")
    print("job_index:", job_index)
    print("job_id:", job.get("job_id", "unknown"))

    return_code, gate_file, gate_json, proper_time = run_candidate_and_read_gate(job_index, job)

    print("gate_file:", gate_file)
    print("proper_time_to_explain:", proper_time)

    if return_code == 0 and proper_time is True:
        return True

    if gate_json is None:
        print("No Gemma gate found; not refining this candidate.")
        return False

    if not candidate_is_interesting_enough_for_refine(gate_json):
        print("Declined candidate looks uninteresting; skipping recenter refinements.")
        return False

    refine_count = 0

    for offset_s in REFINE_OFFSETS_S:
        if refine_count >= MAX_REFINES_PER_CANDIDATE:
            break

        refined_job = make_recentered_job(job, offset_s)
        if refined_job is None:
            continue

        refined_index, appended = append_recentered_job(refined_job)

        if appended is None:
            continue

        refine_count += 1

        print("")
        print("Refined candidate")
        print("offset_s:", offset_s)
        print("job_index:", refined_index)
        print("job_id:", refined_job.get("job_id", "unknown"))

        return_code, gate_file, gate_json, proper_time = run_candidate_and_read_gate(refined_index, refined_job)

        print("gate_file:", gate_file)
        print("proper_time_to_explain:", proper_time)

        if return_code == 0 and proper_time is True:
            return True

    return False


def main():
    max_cities = 100
    start_city = 0

    if len(sys.argv) >= 2:
        max_cities = int(sys.argv[1])

    if len(sys.argv) >= 3:
        start_city = int(sys.argv[2])

    jobs = ensure_multiple_candidates()
    groups = grouped_jobs(jobs)

    selected_groups = list(groups.items())[start_city:start_city + max_cities]

    print("")
    print("One accepted clip per city runner with recenter refinement")
    print("=========================================================")
    print("cities_to_process:", len(selected_groups))
    print("start_city:", start_city)
    print("max_cities:", max_cities)
    print("clip_duration_s:", CLIP_DURATION_S)
    print("refine_offsets_s:", REFINE_OFFSETS_S)
    print("max_refines_per_candidate:", MAX_REFINES_PER_CANDIDATE)
    print("render_config:", os.environ.get("OPTICARVIS_RENDER_CONFIG", ""))

    accepted_cities = []
    rejected_cities = []

    for city_number, (key, items) in enumerate(selected_groups, start=start_city + 1):
        print("")
        print("=" * 80)
        print("City %d: %s | candidates: %d" % (city_number, key, len(items)))
        print("=" * 80)

        city_accepted = False

        for candidate_number, (job_index, job) in enumerate(items, start=1):
            print("")
            print("Candidate %d/%d" % (candidate_number, len(items)))

            if try_original_and_refined(job_index, job):
                print("Accepted one clip for this city. Skipping remaining candidates.")
                accepted_cities.append(key)
                city_accepted = True
                break

            print("No accepted clip from this candidate. Trying next candidate from same city.")

        if not city_accepted:
            rejected_cities.append(key)

    print("")
    print("Summary")
    print("=======")
    print("accepted_cities:", len(accepted_cities))
    print("rejected_cities:", len(rejected_cities))

    if accepted_cities:
        print("")
        print("Accepted:")
        for key in accepted_cities:
            print(" ", key)

    if rejected_cities:
        print("")
        print("No accepted clip found:")
        for key in rejected_cities:
            print(" ", key)


if __name__ == "__main__":
    main()
