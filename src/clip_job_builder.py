import ast
import csv
import json
import os
import re
import sys


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


def load_config_dict():
    try:
        configs = common.get_configs()
    except Exception:
        configs = {}

    if isinstance(configs, dict):
        return configs

    return {}


CONFIGS = load_config_dict()


def config_value(key, default):
    if key in CONFIGS:
        value = CONFIGS[key]
        if value is not None:
            return value

    try:
        value = common.get_configs(key)
    except Exception:
        return default

    if value is None:
        return default

    return value


def config_bool_value(key, default):
    value = config_value(key, default)

    if isinstance(value, bool):
        return value

    text = str(value).strip().lower()

    if text in ["1", "true", "yes", "y", "on"]:
        return True

    if text in ["0", "false", "no", "n", "off"]:
        return False

    return bool(default)


def config_float_value(key, default):
    value = config_value(key, default)

    try:
        return float(value)
    except Exception:
        return float(default)


def config_int_value(key, default):
    value = config_value(key, default)

    try:
        return int(float(value))
    except Exception:
        return int(default)


MAPPING_CSV = resolve_project_path(config_value("mapping", "mapping.csv"))
VIDEO_ROOT = resolve_project_path(config_value("videos", "videos"))

WORKFLOW_OUTPUTS = resolve_project_path(
    config_value("workflow_outputs", "workflow_outputs")
)

ALPAMAYO_OUTPUTS = resolve_project_path(
    config_value("alpamayo_outputs", "alpamayo_outputs")
)

CLIP_ROOT = resolve_project_path(
    config_value("clip_root", os.path.join(ALPAMAYO_OUTPUTS, "crowd_clips"))
)

ALPAMAYO_JSON_ROOT = resolve_project_path(
    config_value(
        "alpamayo_json_dir",
        config_value(
            "ALPAMAYO_JSON_DIR",
            os.path.join(ALPAMAYO_OUTPUTS, "alpamayo_json"),
        ),
    )
)

JOBS_JSONL = resolve_project_path(
    config_value("clip_jobs_jsonl", os.path.join(WORKFLOW_OUTPUTS, "clip_jobs.jsonl"))
)

SUMMARY_JSON = resolve_project_path(
    config_value(
        "clip_jobs_summary_json",
        os.path.join(WORKFLOW_OUTPUTS, "clip_jobs_summary.json"),
    )
)

CLIP_LENGTH_S = config_float_value("CLIP_LENGTH_S", 30.0)
STRIDE_S = config_float_value("STRIDE_S", 15.0)
CITY_LIMIT = config_int_value("CITY_LIMIT", 0)
CITY_FOOTAGE_S = config_float_value("CITY_FOOTAGE_S", 3600.0)
ONE_CLIP_PER_CITY = config_bool_value("ONE_CLIP_PER_CITY", False)

USE_EVENT_CANDIDATE_SEARCH = config_bool_value("USE_EVENT_CANDIDATE_SEARCH", False)
EVENT_MAX_EVENTS_PER_VIDEO = config_int_value("EVENT_MAX_EVENTS_PER_VIDEO", 10)
EVENT_WINDOWS_PER_EVENT = config_int_value("EVENT_WINDOWS_PER_EVENT", 6)
EVENT_MAX_WINDOWS_PER_VIDEO = config_int_value(
    "EVENT_MAX_WINDOWS_PER_VIDEO",
    EVENT_MAX_EVENTS_PER_VIDEO * EVENT_WINDOWS_PER_EVENT,
)
EVENT_FALLBACK_WINDOWS_PER_VIDEO = config_int_value(
    "EVENT_FALLBACK_WINDOWS_PER_VIDEO",
    10,
)
PRINT_PROGRESS_EVERY_N_CITIES = config_int_value("PRINT_PROGRESS_EVERY_N_CITIES", 100)


def ensure_dir(path_value):
    os.makedirs(path_value, exist_ok=True)


def slugify(value):
    text = str(value or "unknown").strip().lower()
    text = text.replace("&", "and")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")

    if not text:
        return "unknown"

    return text


def get_first(row, names, default=""):
    lower_lookup = {}
    for key in row:
        lower_lookup[str(key).strip().lower()] = key

    for name in names:
        lookup_key = str(name).strip().lower()
        if lookup_key in lower_lookup:
            value = row.get(lower_lookup[lookup_key])
            if value is not None and str(value).strip() != "":
                return value

    return default


def parse_literal(value):
    if value is None:
        return None

    if isinstance(value, (list, tuple, dict)):
        return value

    text = str(value).strip()

    if text == "":
        return None

    try:
        return json.loads(text)
    except Exception:
        pass

    try:
        return ast.literal_eval(text)
    except Exception:
        pass

    return text


def clean_video_id(value):
    text = str(value or "").strip()
    text = text.replace("\\", "/")
    text = text.split("/")[-1]

    for extension in [".mp4", ".mkv", ".mov", ".avi"]:
        if text.lower().endswith(extension):
            text = text[: -len(extension)]

    matches = re.findall(r"[A-Za-z0-9_-]{6,}", text)
    if matches:
        return matches[0]

    text = re.sub(r"[^A-Za-z0-9_-]", "", text)
    return text


def parse_video_ids(row):
    videos_value = get_first(
        row,
        [
            "videos",
            "video_ids",
            "video_id_list",
            "youtube_ids",
            "youtube_id_list",
            "files",
        ],
        "",
    )

    parsed = parse_literal(videos_value)
    video_ids = []

    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, dict):
                video_id = get_first(
                    item,
                    ["video_id", "youtube_id", "id", "file", "filename", "name"],
                    "",
                )
                video_id = clean_video_id(video_id)
                if video_id:
                    video_ids.append(video_id)
            else:
                video_id = clean_video_id(item)
                if video_id:
                    video_ids.append(video_id)

    elif isinstance(parsed, dict):
        for key in parsed:
            video_id = clean_video_id(key)
            if video_id:
                video_ids.append(video_id)

    elif parsed:
        parts = re.split(r"[,;|\s]+", str(parsed))
        for part in parts:
            video_id = clean_video_id(part)
            if video_id:
                video_ids.append(video_id)

    if not video_ids:
        video_id = get_first(
            row,
            ["video_id", "youtube_id", "id", "file", "filename"],
            "",
        )
        video_id = clean_video_id(video_id)
        if video_id:
            video_ids.append(video_id)

    seen = set()
    unique_video_ids = []

    for video_id in video_ids:
        if video_id not in seen:
            seen.add(video_id)
            unique_video_ids.append(video_id)

    return unique_video_ids


def parse_number(value, default):
    try:
        return float(value)
    except Exception:
        return float(default)


def interval_from_pair(pair_value):
    if not isinstance(pair_value, (list, tuple)):
        return None

    if len(pair_value) < 2:
        return None

    start_s = parse_number(pair_value[0], 0.0)
    end_s = parse_number(pair_value[1], start_s + CITY_FOOTAGE_S)

    if end_s <= start_s:
        return None

    return {
        "start_s": start_s,
        "end_s": end_s,
    }


def intervals_from_value(value):
    parsed = parse_literal(value)
    intervals = []

    if parsed is None:
        return intervals

    if isinstance(parsed, dict):
        start_value = None
        end_value = None

        for key in ["start", "start_s", "start_time", "begin"]:
            if key in parsed:
                start_value = parsed[key]

        for key in ["end", "end_s", "end_time", "stop"]:
            if key in parsed:
                end_value = parsed[key]

        if start_value is not None and end_value is not None:
            interval = interval_from_pair([start_value, end_value])
            if interval is not None:
                intervals.append(interval)

        return intervals

    if isinstance(parsed, list):
        if len(parsed) >= 2 and not isinstance(parsed[0], (list, tuple, dict)):
            interval = interval_from_pair(parsed)
            if interval is not None:
                intervals.append(interval)
            return intervals

        for item in parsed:
            if isinstance(item, dict):
                interval = intervals_from_value(item)
                intervals.extend(interval)
            else:
                interval = interval_from_pair(item)
                if interval is not None:
                    intervals.append(interval)

    return intervals


def parse_intervals_for_row(row, video_ids):
    intervals_value = get_first(
        row,
        [
            "intervals",
            "video_intervals",
            "valid_intervals",
            "time_intervals",
            "ranges",
            "segments",
        ],
        "",
    )

    parsed = parse_literal(intervals_value)
    intervals_by_video = {}

    for video_id in video_ids:
        intervals_by_video[video_id] = []

    if isinstance(parsed, dict):
        for key, value in parsed.items():
            video_id = clean_video_id(key)

            if video_id in intervals_by_video:
                intervals_by_video[video_id].extend(intervals_from_value(value))

    elif isinstance(parsed, list):
        if len(video_ids) == len(parsed):
            looks_like_per_video = False

            for item in parsed:
                if isinstance(item, list) and item and isinstance(item[0], (list, tuple, dict)):
                    looks_like_per_video = True
                if isinstance(item, dict):
                    looks_like_per_video = True

            if looks_like_per_video:
                for index, video_id in enumerate(video_ids):
                    intervals_by_video[video_id].extend(intervals_from_value(parsed[index]))
            else:
                shared_intervals = intervals_from_value(parsed)
                for video_id in video_ids:
                    intervals_by_video[video_id].extend(shared_intervals)
        else:
            shared_intervals = intervals_from_value(parsed)
            for video_id in video_ids:
                intervals_by_video[video_id].extend(shared_intervals)

    for video_id in video_ids:
        if not intervals_by_video[video_id]:
            start_value = get_first(
                row,
                ["start_s", "start", "segment_start_time_s", "begin"],
                "",
            )
            end_value = get_first(
                row,
                ["end_s", "end", "segment_end_time_s", "stop"],
                "",
            )

            if str(start_value).strip() != "" and str(end_value).strip() != "":
                interval = interval_from_pair([start_value, end_value])
                if interval is not None:
                    intervals_by_video[video_id].append(interval)

        if not intervals_by_video[video_id]:
            fallback_end = CITY_FOOTAGE_S
            if fallback_end <= CLIP_LENGTH_S:
                fallback_end = CLIP_LENGTH_S
            intervals_by_video[video_id].append(
                {
                    "start_s": 0.0,
                    "end_s": fallback_end,
                }
            )

    return intervals_by_video


def source_video_path_for(video_id):
    return normalise_path(os.path.join(VIDEO_ROOT, video_id + ".mp4"))


def clip_video_path_for(video_id, start_s):
    start_int = int(round(float(start_s)))
    length_int = int(round(float(CLIP_LENGTH_S)))

    return normalise_path(
        os.path.join(
            CLIP_ROOT,
            video_id + "_" + str(start_int) + "_" + str(length_int) + "s.mp4",
        )
    )


def alpamayo_json_path_for(video_id, start_s):
    start_int = int(round(float(start_s)))

    return normalise_path(
        os.path.join(
            ALPAMAYO_JSON_ROOT,
            video_id + "_" + str(start_int) + "_alpamayo.json",
        )
    )


def make_stride_windows(interval_start_s, interval_end_s, max_windows):
    windows = []
    cursor = float(interval_start_s)

    while cursor + CLIP_LENGTH_S <= interval_end_s:
        windows.append(
            {
                "start_s": float(cursor),
                "end_s": float(cursor + CLIP_LENGTH_S),
                "selection_score": None,
            }
        )

        if max_windows > 0 and len(windows) >= max_windows:
            break

        cursor += STRIDE_S

    return windows


def filter_windows_to_interval(windows, interval_start_s, interval_end_s):
    filtered = []

    for window in windows:
        start_s = parse_number(window.get("start_s"), 0.0)
        end_s = parse_number(window.get("end_s"), start_s + CLIP_LENGTH_S)

        if end_s <= start_s:
            continue

        if start_s < interval_start_s:
            continue

        if end_s > interval_end_s:
            continue

        filtered.append(
            {
                "start_s": start_s,
                "end_s": end_s,
                "selection_score": window.get("score", window.get("selection_score")),
            }
        )

    return filtered


def dedupe_windows(windows):
    seen = set()
    deduped = []

    for window in windows:
        start_int = int(round(float(window["start_s"])))

        if start_int in seen:
            continue

        seen.add(start_int)
        deduped.append(window)

    return deduped


def event_windows_for_video(source_video, interval_start_s, interval_end_s):
    if not os.path.isfile(source_video):
        return []

    try:
        from event_candidate_search import select_candidate_windows
    except Exception:
        return []

    try:
        windows = select_candidate_windows(source_video, CLIP_LENGTH_S)
    except Exception as error:
        print("Event candidate search failed:")
        print(source_video)
        print(str(error))
        return []

    windows = filter_windows_to_interval(windows, interval_start_s, interval_end_s)
    windows = dedupe_windows(windows)

    if EVENT_MAX_WINDOWS_PER_VIDEO > 0:
        windows = windows[:EVENT_MAX_WINDOWS_PER_VIDEO]

    return windows


def build_jobs():
    if not os.path.isfile(MAPPING_CSV):
        print("Missing mapping CSV:")
        print(MAPPING_CSV)
        raise SystemExit(1)

    with open(MAPPING_CSV, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    original_row_count = len(rows)

    if CITY_LIMIT > 0:
        selected_rows = rows[:CITY_LIMIT]
        print(
            "Mapping has "
            + str(original_row_count)
            + " rows; using the first "
            + str(len(selected_rows))
            + " cities."
        )
    else:
        selected_rows = rows
        print("Mapping has " + str(original_row_count) + " rows; using all cities.")

    jobs = []
    city_summaries = []
    zero_job_cities = []

    for row_index, row in enumerate(selected_rows):
        city_index = row_index + 1

        if PRINT_PROGRESS_EVERY_N_CITIES > 0:
            if city_index == 1 or city_index % PRINT_PROGRESS_EVERY_N_CITIES == 0:
                print(
                    "Building city "
                    + str(city_index)
                    + "/"
                    + str(len(selected_rows))
                )


        city = str(get_first(row, ["city", "City", "city_name", "location"], "Unknown")).strip()
        country = str(get_first(row, ["country", "Country"], "Unknown")).strip()
        continent = str(get_first(row, ["continent", "Continent"], "Unknown")).strip()

        city_slug = slugify(city)
        country_slug = slugify(country)

        video_ids = parse_video_ids(row)
        intervals_by_video = parse_intervals_for_row(row, video_ids)

        city_job_count = 0
        used_city_footage_s = 0.0
        city_footage_limit_enabled = CITY_FOOTAGE_S > 0

        for video_id in video_ids:
            source_video = source_video_path_for(video_id)
            intervals = intervals_by_video.get(video_id, [])

            for interval in intervals:
                interval_start_s = float(interval["start_s"])
                interval_end_s = float(interval["end_s"])

                if interval_end_s - interval_start_s < CLIP_LENGTH_S:
                    continue

                if city_footage_limit_enabled and used_city_footage_s >= CITY_FOOTAGE_S:
                    break

                effective_interval_end_s = interval_end_s

                if city_footage_limit_enabled:
                    remaining_s = CITY_FOOTAGE_S - used_city_footage_s
                    effective_interval_end_s = min(
                        interval_end_s,
                        interval_start_s + remaining_s,
                    )

                if effective_interval_end_s - interval_start_s < CLIP_LENGTH_S:
                    continue

                selection_method = "stride"

                if USE_EVENT_CANDIDATE_SEARCH:
                    candidate_windows = event_windows_for_video(
                        source_video,
                        interval_start_s,
                        effective_interval_end_s,
                    )

                    if candidate_windows:
                        selection_method = "event_candidate_search"
                    else:
                        candidate_windows = make_stride_windows(
                            interval_start_s,
                            effective_interval_end_s,
                            EVENT_FALLBACK_WINDOWS_PER_VIDEO,
                        )
                        selection_method = "stride_fallback"
                else:
                    candidate_windows = make_stride_windows(
                        interval_start_s,
                        effective_interval_end_s,
                        0,
                    )

                for window in candidate_windows:
                    start_s = float(window["start_s"])
                    end_s = float(window["end_s"])
                    start_int = int(round(start_s))

                    clip_tag = video_id + "_" + str(start_int)

                    job_id = (
                        "city"
                        + str(city_index).zfill(3)
                        + "_"
                        + city_slug
                        + "_"
                        + country_slug
                        + "_"
                        + clip_tag
                    )

                    job = {
                        "job_id": job_id,
                        "city_index": city_index,
                        "city": city,
                        "country": country,
                        "continent": continent,
                        "video_id": video_id,
                        "source_video": source_video,
                        "segment_start_time_s": float(start_s),
                        "segment_end_time_s": float(end_s),
                        "clip_length_s": float(CLIP_LENGTH_S),
                        "clip_video": clip_video_path_for(video_id, start_s),
                        "alpamayo_json": alpamayo_json_path_for(video_id, start_s),
                        "selection_method": selection_method,
                        "selection_score": window.get("selection_score"),
                    }

                    jobs.append(job)
                    city_job_count += 1

                    if ONE_CLIP_PER_CITY:
                        break

                used_city_footage_s += max(0.0, effective_interval_end_s - interval_start_s)

                if ONE_CLIP_PER_CITY and city_job_count > 0:
                    break

            if ONE_CLIP_PER_CITY and city_job_count > 0:
                break

            if city_footage_limit_enabled and used_city_footage_s >= CITY_FOOTAGE_S:
                continue

        if city_job_count == 0:
            zero_job_cities.append(city + ", " + country)

        city_summaries.append(
            {
                "city_index": city_index,
                "city": city,
                "country": country,
                "continent": continent,
                "videos": len(video_ids),
                "jobs": city_job_count,
            }
        )

    return jobs, city_summaries, zero_job_cities, original_row_count, len(selected_rows)


def write_outputs(jobs, city_summaries, zero_job_cities, original_row_count, selected_row_count):
    ensure_dir(WORKFLOW_OUTPUTS)
    ensure_dir(CLIP_ROOT)
    ensure_dir(ALPAMAYO_JSON_ROOT)

    with open(JOBS_JSONL, "w", encoding="utf-8", newline="\n") as handle:
        for job in jobs:
            handle.write(json.dumps(job, ensure_ascii=False) + "\n")

    summary = {
        "project_root": PROJECT_ROOT,
        "mapping_csv": MAPPING_CSV,
        "mapping_rows": original_row_count,
        "cities": selected_row_count,
        "total_jobs": len(jobs),
        "clip_length_s": CLIP_LENGTH_S,
        "stride_s": STRIDE_S,
        "one_clip_per_city": ONE_CLIP_PER_CITY,
        "city_footage_s": CITY_FOOTAGE_S,
        "use_event_candidate_search": USE_EVENT_CANDIDATE_SEARCH,
        "event_max_windows_per_video": EVENT_MAX_WINDOWS_PER_VIDEO,
        "event_fallback_windows_per_video": EVENT_FALLBACK_WINDOWS_PER_VIDEO,
        "print_progress_every_n_cities": PRINT_PROGRESS_EVERY_N_CITIES,
        "video_root": VIDEO_ROOT,
        "clip_root": CLIP_ROOT,
        "alpamayo_json_root": ALPAMAYO_JSON_ROOT,
        "jobs_jsonl": JOBS_JSONL,
        "summary_json": SUMMARY_JSON,
        "zero_job_cities": zero_job_cities,
        "city_summaries": city_summaries,
    }

    with open(SUMMARY_JSON, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print("")
    print("Clip job builder complete")
    print("=========================")
    print("project_root:", PROJECT_ROOT)
    print("mapping_csv:", MAPPING_CSV)
    print("cities:", selected_row_count)
    print("total_jobs:", len(jobs))
    print("clip_length_s:", CLIP_LENGTH_S)
    print("stride_s:", STRIDE_S)
    print("one_clip_per_city:", ONE_CLIP_PER_CITY)
    print("city_footage_s:", CITY_FOOTAGE_S)
    print("use_event_candidate_search:", USE_EVENT_CANDIDATE_SEARCH)
    print("event_max_windows_per_video:", EVENT_MAX_WINDOWS_PER_VIDEO)
    print("event_fallback_windows_per_video:", EVENT_FALLBACK_WINDOWS_PER_VIDEO)
    print("print_progress_every_n_cities:", PRINT_PROGRESS_EVERY_N_CITIES)
    print("video_root:", VIDEO_ROOT)
    print("clip_root:", CLIP_ROOT)
    print("alpamayo_json_root:", ALPAMAYO_JSON_ROOT)
    print("jobs_jsonl:", JOBS_JSONL)
    print("summary_json:", SUMMARY_JSON)

    if zero_job_cities:
        print("")
        print("WARNING: Some cities produced zero jobs:")
        for city_name in zero_job_cities[:50]:
            print("  " + city_name)

        if len(zero_job_cities) > 50:
            print("  ... and " + str(len(zero_job_cities) - 50) + " more")


def main():
    jobs, city_summaries, zero_job_cities, original_row_count, selected_row_count = build_jobs()
    write_outputs(jobs, city_summaries, zero_job_cities, original_row_count, selected_row_count)


if __name__ == "__main__":
    main()
