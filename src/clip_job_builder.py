r"""Build 30 second OptiCarVis clip jobs from the 100 city CROWD mapping.

Input:
    opticarvis/mapping.csv

Output:
    opticarvis/workflow_outputs/clip_jobs.jsonl
    opticarvis/workflow_outputs/clip_jobs_summary.json

Default policy:
    100 cities maximum
    1 hour per city
    30 second clip length
    60 second stride

This script does not run Alpamayo or render videos. It only creates the job list.
"""

import ast
import csv
import json
import os
import re
import sys

from pipeline_common import (
    PROJECT_ROOT,
    MAPPING_CSV,
    WORKFLOW_OUTPUTS,
    VIDEOS_DIR,
    CROWD_CLIPS_DIR,
    ALPAMAYO_JSON_DIR,
    ensure_dir,
    write_json,
    normalise_path,
)


CITY_LIMIT = int(os.environ.get("OPTICARVIS_CITY_LIMIT", "100"))
CITY_FOOTAGE_S = float(os.environ.get("OPTICARVIS_CITY_FOOTAGE_S", "3600"))
CLIP_LENGTH_S = float(os.environ.get("OPTICARVIS_CLIP_LENGTH_S", "30"))
STRIDE_S = float(os.environ.get("OPTICARVIS_STRIDE_S", "60"))

OUTPUT_JSONL = normalise_path(
    os.environ.get(
        "OPTICARVIS_CLIP_JOBS",
        os.path.join(WORKFLOW_OUTPUTS, "clip_jobs.jsonl"),
    )
)

OUTPUT_SUMMARY = normalise_path(
    os.environ.get(
        "OPTICARVIS_CLIP_JOBS_SUMMARY",
        os.path.join(WORKFLOW_OUTPUTS, "clip_jobs_summary.json"),
    )
)

VIDEO_ROOT = normalise_path(
    os.environ.get(
        "OPTICARVIS_VIDEO_ROOT",
        os.environ.get("OPTICARVIS_VIDEOS_DIR", VIDEOS_DIR),
    )
)

CLIP_ROOT = normalise_path(
    os.environ.get(
        "OPTICARVIS_CLIP_ROOT",
        os.environ.get("OPTICARVIS_CROWD_CLIPS_DIR", CROWD_CLIPS_DIR),
    )
)

ALPAMAYO_JSON_ROOT = normalise_path(
    os.environ.get(
        "OPTICARVIS_ALPAMAYO_JSON_ROOT",
        os.environ.get("OPTICARVIS_ALPAMAYO_JSON_DIR", ALPAMAYO_JSON_DIR),
    )
)


def safe_slug(value):
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = text.strip("_")
    return text or "unknown"


def parse_number(value, default=None):
    if value is None:
        return default

    text = str(value).strip()

    if not text:
        return default

    try:
        return float(text)
    except ValueError:
        return default


def parse_video_list(value):
    text = str(value or "").strip()

    if not text:
        return []

    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]

    videos = []
    for item in text.split(","):
        video_id = item.strip().strip("'").strip('"')
        if video_id:
            videos.append(video_id)

    return videos


def parse_nested_number_list(value):
    text = str(value or "").strip()

    if not text:
        return []

    try:
        parsed = ast.literal_eval(text)
    except Exception:
        number = parse_number(text)
        return [[number]] if number is not None else []

    if not isinstance(parsed, list):
        return [[parsed]]

    normalised = []
    for item in parsed:
        if isinstance(item, list):
            normalised.append(item)
        else:
            normalised.append([item])

    return normalised


def get_nested_value(nested, outer_index, inner_index, default=None):
    if outer_index >= len(nested):
        return default

    item = nested[outer_index]

    if not isinstance(item, list):
        item = [item]

    if inner_index >= len(item):
        return default

    return item[inner_index]


def row_intervals(row):
    videos = parse_video_list(row.get("videos", ""))
    starts = parse_nested_number_list(row.get("start_time", ""))
    ends = parse_nested_number_list(row.get("end_time", ""))
    times = parse_nested_number_list(row.get("time_of_day", ""))

    intervals = []

    for video_index, video_id in enumerate(videos):
        start_values = starts[video_index] if video_index < len(starts) else [0]
        time_values = times[video_index] if video_index < len(times) else []

        if not isinstance(start_values, list):
            start_values = [start_values]

        for segment_index, start_value in enumerate(start_values):
            start_s = parse_number(start_value)
            end_s = parse_number(get_nested_value(ends, video_index, segment_index))
            time_code = get_nested_value(times, video_index, segment_index)

            if start_s is None:
                continue

            if end_s is None:
                end_s = start_s + CITY_FOOTAGE_S

            if end_s - start_s < CLIP_LENGTH_S:
                continue

            intervals.append(
                {
                    "video_id": video_id,
                    "start_s": float(start_s),
                    "end_s": float(end_s),
                    "time_of_day": time_code,
                    "duration_s": float(end_s - start_s),
                    "source_video": normalise_path(os.path.join(VIDEO_ROOT, video_id + ".mp4")),
                }
            )

    intervals = sorted(
        intervals,
        key=lambda item: item["duration_s"],
        reverse=True,
    )

    return intervals


def build_jobs_for_city(row, city_index):
    locality = row.get("locality", "")
    country = row.get("country", "")
    continent = row.get("continent", "")

    intervals = row_intervals(row)
    jobs = []
    used_timeline_s = 0.0

    for interval in intervals:
        if used_timeline_s >= CITY_FOOTAGE_S:
            break

        source_start = interval["start_s"]
        source_end = interval["end_s"]
        t = source_start

        while t + CLIP_LENGTH_S <= source_end and used_timeline_s < CITY_FOOTAGE_S:
            start_int = int(round(t))
            clip_tag = interval["video_id"] + "_" + str(start_int)
            city_slug = safe_slug(locality)
            country_slug = safe_slug(country)

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

            clip_video = normalise_path(
                os.path.join(
                    CLIP_ROOT,
                    interval["video_id"]
                    + "_"
                    + str(start_int)
                    + "_"
                    + str(int(round(CLIP_LENGTH_S)))
                    + "s.mp4",
                )
            )

            alpamayo_json = normalise_path(
                os.path.join(
                    ALPAMAYO_JSON_ROOT,
                    interval["video_id"]
                    + "_"
                    + str(start_int)
                    + "_alpamayo.json",
                )
            )

            jobs.append(
                {
                    "job_id": job_id,
                    "city_index": city_index,
                    "locality": locality,
                    "state": row.get("state", ""),
                    "country": country,
                    "iso3": row.get("iso3", ""),
                    "continent": continent,
                    "lat": row.get("lat", ""),
                    "lon": row.get("lon", ""),
                    "traffic_mortality": row.get("traffic_mortality", ""),
                    "traffic_index": row.get("traffic_index", ""),
                    "video_id": interval["video_id"],
                    "source_video": interval["source_video"],
                    "source_interval_start_s": interval["start_s"],
                    "source_interval_end_s": interval["end_s"],
                    "segment_start_time_s": float(start_int),
                    "clip_length_s": CLIP_LENGTH_S,
                    "stride_s": STRIDE_S,
                    "time_of_day": interval["time_of_day"],
                    "clip_video": clip_video,
                    "alpamayo_json": alpamayo_json,
                }
            )

            used_timeline_s += STRIDE_S
            t += STRIDE_S

    return jobs, intervals


def main():
    mapping_csv = normalise_path(sys.argv[1] if len(sys.argv) > 1 else MAPPING_CSV)

    if not os.path.isfile(mapping_csv):
        print("Missing mapping CSV:")
        print(mapping_csv)
        raise SystemExit(1)

    ensure_dir(WORKFLOW_OUTPUTS)
    ensure_dir(CLIP_ROOT)
    ensure_dir(ALPAMAYO_JSON_ROOT)

    with open(mapping_csv, "r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    if len(rows) > CITY_LIMIT:
        print("Mapping has %d rows; using the first %d cities." % (len(rows), CITY_LIMIT))
        rows = rows[:CITY_LIMIT]

    all_jobs = []
    city_summaries = []

    for city_index, row in enumerate(rows, start=1):
        jobs, intervals = build_jobs_for_city(row, city_index)
        all_jobs.extend(jobs)

        city_summaries.append(
            {
                "city_index": city_index,
                "locality": row.get("locality", ""),
                "country": row.get("country", ""),
                "continent": row.get("continent", ""),
                "candidate_source_intervals": len(intervals),
                "generated_jobs": len(jobs),
                "generated_source_timeline_s": len(jobs) * STRIDE_S,
            }
        )

    ensure_dir(os.path.dirname(OUTPUT_JSONL))
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as handle:
        for job in all_jobs:
            handle.write(json.dumps(job, ensure_ascii=False) + "\n")

    summary = {
        "project_root": PROJECT_ROOT,
        "mapping_csv": mapping_csv,
        "output_jsonl": OUTPUT_JSONL,
        "city_count": len(rows),
        "total_jobs": len(all_jobs),
        "city_footage_s": CITY_FOOTAGE_S,
        "clip_length_s": CLIP_LENGTH_S,
        "stride_s": STRIDE_S,
        "expected_jobs_per_city": int(CITY_FOOTAGE_S / STRIDE_S),
        "video_root": VIDEO_ROOT,
        "clip_root": CLIP_ROOT,
        "alpamayo_json_root": ALPAMAYO_JSON_ROOT,
        "cities": city_summaries,
    }

    write_json(OUTPUT_SUMMARY, summary)

    print("")
    print("Clip job builder complete")
    print("=========================")
    print("project_root:", PROJECT_ROOT)
    print("mapping_csv:", mapping_csv)
    print("cities:", len(rows))
    print("total_jobs:", len(all_jobs))
    print("clip_length_s:", CLIP_LENGTH_S)
    print("stride_s:", STRIDE_S)
    print("video_root:", VIDEO_ROOT)
    print("clip_root:", CLIP_ROOT)
    print("alpamayo_json_root:", ALPAMAYO_JSON_ROOT)
    print("jobs_jsonl:", OUTPUT_JSONL)
    print("summary_json:", OUTPUT_SUMMARY)

    weak = [c for c in city_summaries if c["generated_jobs"] == 0]
    if weak:
        print("")
        print("WARNING: Some cities produced zero jobs:")
        for item in weak[:20]:
            print("  %s, %s" % (item["locality"], item["country"]))


if __name__ == "__main__":
    main()
