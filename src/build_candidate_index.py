"""Build the offline OptiCarVis candidate-window Parquet index.

Run from the repository root after source videos have been downloaded:

    python src/build_candidate_index.py

Optional positional video ids limit a validation run:

    python src/build_candidate_index.py 3ai7SUaPoHM

The command demuxes packets, decodes keyframes only, caches their SigLIP2
embeddings, and ranks windows against policy-derived prompts. It does not build
clip jobs or make explanation decisions.
"""

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

import candidate_index
import candidate_semantics


def load_config():
    configured_path = os.path.join(PROJECT_ROOT, "config")
    default_path = os.path.join(PROJECT_ROOT, "default.config")
    path = configured_path if os.path.isfile(configured_path) else default_path

    with open(path, "r", encoding="utf-8-sig") as handle:
        value = json.load(handle)

    return value if isinstance(value, dict) else {}


CONFIG = load_config()


def config_value(key, default, environment_names=None):
    names = list(environment_names or [])
    names.append("OPTICARVIS_" + str(key))

    for name in names:
        value = os.environ.get(name)
        if value is not None and str(value).strip():
            return value

    return CONFIG.get(key, default)


def resolve_project_path(value):
    text = str(value).strip()
    if os.path.isabs(text):
        return os.path.normpath(text)
    return os.path.normpath(os.path.join(PROJECT_ROOT, text))


def config_path(key, default, environment_names=None):
    return resolve_project_path(config_value(key, default, environment_names))


def config_float(key, default):
    value = config_value(key, default)

    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def config_int(key, default):
    value = config_value(key, default)

    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def config_bool(key, default):
    value = config_value(key, default)

    if isinstance(value, bool):
        return value

    text = str(value).strip().lower()

    if text in ("1", "true", "yes", "y", "on"):
        return True
    if text in ("0", "false", "no", "n", "off"):
        return False

    return bool(default)


def get_first(row, names, default=""):
    lower_lookup = {str(key).strip().lower(): key for key in row}

    for name in names:
        source_key = lower_lookup.get(str(name).strip().lower())
        if source_key is None:
            continue

        value = row.get(source_key)
        if value is not None and str(value).strip():
            return value

    return default


def clean_video_id(value):
    text = str(value or "").strip().replace("\\", "/").split("/")[-1]
    text = re.sub(r"\.(mp4|mkv|mov|avi)$", "", text, flags=re.IGNORECASE)
    matches = re.findall(r"[A-Za-z0-9_-]{6,}", text)
    return matches[0] if matches else ""


def parse_video_ids(row):
    value = get_first(
        row,
        ["videos", "video_ids", "youtube_ids", "files", "video_id"],
        "",
    )
    text = str(value or "").strip().strip("[](){}")
    video_ids = []
    seen = set()

    for part in re.split(r"[,;|\s]+", text):
        video_id = clean_video_id(part.strip().strip("'\""))
        if video_id and video_id not in seen:
            seen.add(video_id)
            video_ids.append(video_id)

    return video_ids


def video_records(mapping_csv, video_root, requested_video_ids=None):
    requested = set(requested_video_ids or [])
    records = []
    records_by_video = {}

    with open(mapping_csv, "r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            city = str(
                get_first(
                    row,
                    ["city", "locality", "city_name", "location"],
                    "Unknown",
                )
            ).strip()
            country = str(get_first(row, ["country"], "Unknown")).strip()
            continent = str(get_first(row, ["continent"], "Unknown")).strip()

            video_ids = parse_video_ids(row)
            intervals_by_video = candidate_index.mapping_intervals_for_videos(
                row,
                video_ids,
            )

            for video_id in video_ids:
                if requested and video_id not in requested:
                    continue

                if video_id not in records_by_video:
                    record = {
                        "city": city,
                        "country": country,
                        "continent": continent,
                        "video_id": video_id,
                        "source_video": os.path.join(video_root, video_id + ".mp4"),
                        "intervals": [],
                    }
                    records_by_video[video_id] = record
                    records.append(record)

                record = records_by_video[video_id]
                record["intervals"] = candidate_index.normalise_intervals(
                    record["intervals"] + intervals_by_video.get(video_id, [])
                )

    return records


def summary_path_for(index_path):
    stem, _extension = os.path.splitext(index_path)
    return stem + "_summary.json"


def print_top_candidates(rows, limit=10):
    representatives = [
        row for row in rows if bool(row.get("is_event_representative"))
    ]
    ranked = sorted(
        representatives or rows,
        key=lambda row: (
            str(row["video_id"]),
            int(row["candidate_rank"]),
        ),
    )

    print("")
    print("Highest ranked distinct candidate events")
    print("========================================")

    for row in ranked[: max(0, int(limit))]:
        print(
            "%s start=%0.2f s rank=%d semantic=%0.4f packet=%0.3f event=%s windows=%d"
            % (
                row["video_id"],
                float(row["t_start_s"]),
                int(row["candidate_rank"]),
                float(row.get("semantic_score", float("nan"))),
                float(row.get("packet_candidate_score", row["candidate_score"])),
                row.get("event_cluster_id", "unclustered"),
                int(row.get("event_window_count", 1)),
            )
        )


def main():
    requested_video_ids = [value.strip() for value in sys.argv[1:] if value.strip()]
    index_path = config_path(
        "CANDIDATE_INDEX_PARQUET",
        os.path.join("workflow_outputs", "candidate_index.parquet"),
    )

    if requested_video_ids and not os.environ.get(
        "OPTICARVIS_CANDIDATE_INDEX_PARQUET",
        "",
    ).strip():
        stem, extension = os.path.splitext(index_path)
        index_path = stem + "_subset" + (extension or ".parquet")

    mapping_csv = config_path(
        "mapping",
        "mapping.csv",
        environment_names=["OPTICARVIS_MAPPING_CSV"],
    )
    video_root = config_path(
        "videos",
        "videos",
        environment_names=["OPTICARVIS_VIDEOS_DIR"],
    )
    clip_length_s = config_float("CLIP_LENGTH_S", 30.0)
    window_stride_s = config_float("CANDIDATE_INDEX_STRIDE_S", 5.0)
    event_separation_s = config_float(
        "CANDIDATE_EVENT_SEPARATION_S",
        clip_length_s,
    )
    semantic_enabled = config_bool("CANDIDATE_SEMANTIC_ENABLED", True)
    semantic_prompt_path = config_path(
        "CANDIDATE_SEMANTIC_PROMPTS",
        os.path.join("configs", "candidate_semantic_prompts.json"),
    )
    semantic_cache_dir = config_path(
        "CANDIDATE_SEMANTIC_CACHE_DIR",
        os.path.join("workflow_outputs", "candidate_keyframe_embeddings"),
    )
    semantic_batch_size = config_int("CANDIDATE_SEMANTIC_BATCH_SIZE", 32)
    semantic_top_k = config_int("CANDIDATE_SEMANTIC_TOP_K", 2)
    semantic_scorer = None

    if semantic_enabled:
        semantic_scorer = candidate_semantics.SemanticCandidateScorer(
            prompt_path=semantic_prompt_path,
            cache_dir=semantic_cache_dir,
            batch_size=semantic_batch_size,
            top_k=semantic_top_k,
        )

    print("")
    print("Offline candidate index")
    print("=======================")
    print("index_version:", candidate_index.INDEX_VERSION)
    print("mapping_csv:", mapping_csv)
    print("video_root:", video_root)
    print("Reading mapping and matching local source videos...")

    mapping_records = video_records(mapping_csv, video_root, requested_video_ids)
    mapping_video_count = len(mapping_records)

    if requested_video_ids:
        records = mapping_records
        missing_video_count = sum(
            1 for record in records if not os.path.isfile(record["source_video"])
        )
    else:
        records = [
            record
            for record in mapping_records
            if os.path.isfile(record["source_video"])
        ]
        missing_video_count = mapping_video_count - len(records)

    print("videos_in_mapping:", mapping_video_count)
    print("local_videos_selected:", len(records))
    print("missing_videos_skipped:", missing_video_count)
    print("clip_length_s:", clip_length_s)
    print("window_stride_s:", window_stride_s)
    print("event_separation_s:", event_separation_s)
    print("window_scope: mapping.csv start_time to end_time intervals")
    print("semantic_enabled:", semantic_enabled)

    if semantic_scorer is not None:
        print(
            "semantic_implementation_version:",
            candidate_semantics.IMPLEMENTATION_VERSION,
        )
        print("semantic_model:", semantic_scorer.model_id)
        print("semantic_prompts:", semantic_prompt_path)
        print("semantic_cache:", semantic_cache_dir)

    print("output:", index_path)

    all_rows = []
    indexed_video_ids = []
    missing_video_ids = []
    failed_videos = []

    for number, record in enumerate(records, start=1):
        video_id = record["video_id"]
        source_video = record["source_video"]
        print("[%d/%d] %s" % (number, len(records), video_id))

        if not os.path.isfile(source_video):
            missing_video_ids.append(video_id)
            print("  missing:", source_video)
            continue

        if not record.get("intervals"):
            failed_videos.append(
                {
                    "video_id": video_id,
                    "error_type": "MissingMappingInterval",
                    "error": "No valid start_time and end_time pair in mapping.csv",
                }
            )
            print("  skipped: no valid mapping interval")
            continue

        print(
            "  mapping_intervals:",
            ", ".join(
                "%0.2f-%0.2f s" % (interval["start_s"], interval["end_s"])
                for interval in record["intervals"]
            ),
        )

        try:
            rows = candidate_index.index_video(
                record,
                clip_length_s=clip_length_s,
                window_stride_s=window_stride_s,
            )

            if semantic_scorer is not None and rows:
                semantic_scorer.score_rows(
                    rows,
                    source_video=source_video,
                    video_id=video_id,
                    intervals=record["intervals"],
                )

            candidate_index.attach_event_clusters(
                rows,
                separation_s=event_separation_s,
            )
        except Exception as error:
            failed_videos.append(
                {
                    "video_id": video_id,
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            print("  failed:", type(error).__name__, str(error)[:300])
            continue

        all_rows.extend(rows)
        indexed_video_ids.append(video_id)
        print("  windows:", len(rows))
        print(
            "  event_clusters:",
            sum(bool(row.get("is_event_representative")) for row in rows),
        )

        if semantic_scorer is not None:
            print(
                "  keyframe_embeddings:",
                "cache" if semantic_scorer.last_cache_hit else "computed",
            )

    index_written = False

    if all_rows:
        candidate_index.write_parquet_index(all_rows, index_path)
        index_written = True

        if requested_video_ids:
            print_top_candidates(all_rows)
    else:
        print("")
        print("No valid candidate windows were produced; the existing index was preserved.")

    summary = {
        "index_version": candidate_index.INDEX_VERSION,
        "index_path": index_path,
        "index_written": index_written,
        "mapping_csv": mapping_csv,
        "clip_length_s": clip_length_s,
        "window_stride_s": window_stride_s,
        "event_separation_s": event_separation_s,
        "semantic_enabled": semantic_enabled,
        "semantic_model": (
            semantic_scorer.model_id if semantic_scorer is not None else None
        ),
        "semantic_prompt_path": (
            semantic_prompt_path if semantic_scorer is not None else None
        ),
        "semantic_prompt_hash": (
            semantic_scorer.prompt_hash if semantic_scorer is not None else None
        ),
        "semantic_cache_dir": (
            semantic_cache_dir if semantic_scorer is not None else None
        ),
        "videos_in_mapping": mapping_video_count,
        "videos_selected": len(records),
        "videos_indexed": len(indexed_video_ids),
        "candidate_windows": len(all_rows),
        "candidate_events": sum(
            bool(row.get("is_event_representative")) for row in all_rows
        ),
        "indexed_video_ids": indexed_video_ids,
        "missing_video_count": missing_video_count,
        "missing_video_ids": missing_video_ids,
        "failed_videos": failed_videos,
    }
    summary_path = summary_path_for(index_path)

    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print("")
    print("Index complete")
    print("videos_indexed:", len(indexed_video_ids))
    print("candidate_windows:", len(all_rows))
    print(
        "candidate_events:",
        sum(bool(row.get("is_event_representative")) for row in all_rows),
    )
    print("missing_videos_skipped:", missing_video_count)
    print("failed_videos:", len(failed_videos))
    print("index:", index_path)
    print("summary:", summary_path)

    if not index_written:
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
