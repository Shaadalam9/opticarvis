"""Offline semantic and packet index for candidate clip selection.

The live batch must never scan source videos while it builds clip jobs.  This
module demuxes each source once with ffprobe, aggregates encoded packet sizes
into candidate windows, and writes an auditable Parquet table. A separate
keyframe pass can attach cached semantic scores. Job building only reads the
table.

Semantic similarity is a discovery signal and packet energy is its tie break,
not the explanation gate. Alpamayo and Gemma still decide what the scene means
and whether an explanation is warranted.
"""

import csv
import json
import math
import os
import shutil
import statistics
import subprocess
from bisect import bisect_left


INDEX_VERSION = 4

INDEX_COLUMNS = [
    "index_version",
    "city",
    "country",
    "continent",
    "video_id",
    "source_video",
    "source_size_bytes",
    "source_mtime_ns",
    "mapping_interval_start_s",
    "mapping_interval_end_s",
    "t_start_s",
    "t_end_s",
    "clip_length_s",
    "window_stride_s",
    "packet_count",
    "keyframe_count",
    "packet_bytes",
    "packet_bytes_per_s",
    "mean_packet_size_bytes",
    "peak_packet_size_bytes",
    "packet_candidate_score",
    "packet_candidate_percentile",
    "packet_candidate_rank",
    "semantic_model",
    "semantic_prompt_hash",
    "semantic_keyframe_count",
    "semantic_negative_score",
    "semantic_interaction_region_score",
    "semantic_pedestrian_intent_score",
    "semantic_proximity_risk_score",
    "semantic_critical_conflict_score",
    "semantic_trajectory_uncertainty_score",
    "semantic_unusual_context_score",
    "semantic_score",
    "candidate_score",
    "candidate_percentile",
    "candidate_rank",
    "event_cluster_id",
    "event_start_s",
    "event_end_s",
    "event_representative_start_s",
    "event_representative_rank",
    "event_window_count",
    "is_event_representative",
    "selection_source",
]

REQUIRED_QUERY_COLUMNS = {
    "video_id",
    "t_start_s",
    "t_end_s",
    "clip_length_s",
    "candidate_score",
    "candidate_rank",
}

_INDEX_CACHE = {}


def normalise_intervals(intervals):
    """Return sorted, non overlapping, finite time ranges."""
    cleaned = []

    for interval in intervals or []:
        try:
            if isinstance(interval, dict):
                start_s = float(interval["start_s"])
                end_s = float(interval["end_s"])
            else:
                start_s = float(interval[0])
                end_s = float(interval[1])
        except (KeyError, IndexError, TypeError, ValueError):
            continue

        if not math.isfinite(start_s) or not math.isfinite(end_s):
            continue

        start_s = max(0.0, start_s)
        if end_s <= start_s:
            continue

        cleaned.append((start_s, end_s))

    cleaned.sort()
    merged = []

    for start_s, end_s in cleaned:
        if merged and start_s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end_s))
        else:
            merged.append((start_s, end_s))

    return [
        {"start_s": float(start_s), "end_s": float(end_s)}
        for start_s, end_s in merged
    ]


def _mapping_value(row, names):
    lookup = {str(key).strip().lower(): key for key in row}

    for name in names:
        source_key = lookup.get(str(name).strip().lower())
        if source_key is None:
            continue

        value = row.get(source_key)
        if value is not None and str(value).strip():
            return value

    return None


def _json_time_values(value):
    if isinstance(value, (list, tuple)):
        return list(value)

    if value is None:
        return None

    try:
        parsed = json.loads(str(value).strip())
    except (TypeError, ValueError):
        return None

    return parsed if isinstance(parsed, list) else [parsed]


def _numeric_leaves(value):
    if isinstance(value, (list, tuple)):
        result = []
        for item in value:
            result.extend(_numeric_leaves(item))
        return result

    try:
        number = float(value)
    except (TypeError, ValueError):
        return []

    return [number] if math.isfinite(number) else []


def mapping_intervals_for_videos(row, video_ids):
    """Pair mapping.csv start_time and end_time entries with each video."""
    video_ids = list(video_ids)
    result = {video_id: [] for video_id in video_ids}
    starts = _json_time_values(
        _mapping_value(row, ["start_time", "start_times"])
    )
    ends = _json_time_values(
        _mapping_value(row, ["end_time", "end_times"])
    )

    if starts is None or ends is None:
        return result

    if len(video_ids) == 1 and len(starts) != 1:
        starts = [starts]
    if len(video_ids) == 1 and len(ends) != 1:
        ends = [ends]

    if len(starts) != len(video_ids) or len(ends) != len(video_ids):
        return result

    for index, video_id in enumerate(video_ids):
        start_values = _numeric_leaves(starts[index])
        end_values = _numeric_leaves(ends[index])

        if len(start_values) != len(end_values):
            continue

        pairs = zip(start_values, end_values)
        result[video_id] = normalise_intervals(pairs)

    return result


def resolve_ffprobe_path(configured=None):
    """Resolve ffprobe without assuming one platform or install location."""
    if configured is not None and str(configured).strip():
        return os.path.normpath(str(configured).strip())

    configured_env = os.environ.get("OPTICARVIS_FFPROBE", "").strip()
    if configured_env:
        return os.path.normpath(configured_env)

    discovered = shutil.which("ffprobe")
    if discovered:
        return os.path.normpath(discovered)

    return "ffprobe"


def probe_video_packets(video_path, ffprobe_path=None, intervals=None):
    """Return video packet timestamps, sizes and keyframe flags.

    ffprobe demuxes the encoded stream and does not decode pixels.  Its CSV is
    consumed as a stream so an hour-long source does not first create a large
    intermediate text file.
    """
    video_path = os.path.abspath(str(video_path))

    if not os.path.isfile(video_path):
        raise FileNotFoundError(video_path)

    command = [
        resolve_ffprobe_path(ffprobe_path),
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "packet=pts_time,size,flags",
        "-of",
        "csv=p=0",
    ]

    allowed_intervals = normalise_intervals(intervals)
    if intervals is not None and not allowed_intervals:
        return []

    if allowed_intervals:
        read_intervals = ",".join(
            "%0.6f%%%0.6f" % (interval["start_s"], interval["end_s"])
            for interval in allowed_intervals
        )
        command.extend(["-read_intervals", read_intervals])

    command.append(video_path)

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    packets = []

    try:
        reader = csv.reader(process.stdout)

        for values in reader:
            if len(values) < 2:
                continue

            try:
                pts_time = float(values[0])
                size_bytes = int(values[1])
            except (TypeError, ValueError):
                continue

            if not math.isfinite(pts_time) or pts_time < 0 or size_bytes < 0:
                continue

            if allowed_intervals and not any(
                interval["start_s"] <= pts_time < interval["end_s"]
                for interval in allowed_intervals
            ):
                continue

            flags = values[2] if len(values) >= 3 else ""
            packets.append(
                {
                    "pts_time_s": pts_time,
                    "size_bytes": size_bytes,
                    "is_keyframe": "K" in flags,
                }
            )
    finally:
        if process.stdout is not None:
            process.stdout.close()

    stderr = process.stderr.read() if process.stderr is not None else ""
    return_code = process.wait()

    if process.stderr is not None:
        process.stderr.close()

    if return_code != 0:
        detail = stderr.strip() or "ffprobe returned exit code %d" % return_code
        raise RuntimeError(detail)

    packets.sort(key=lambda packet: packet["pts_time_s"])
    return packets


def robust_scores(values):
    """Return within-video robust z scores for positive energy values."""
    if not values:
        return []

    transformed = [math.log1p(max(0.0, float(value))) for value in values]
    centre = statistics.median(transformed)
    deviations = [abs(value - centre) for value in transformed]
    mad = statistics.median(deviations)

    if mad > 1e-12:
        scale = 1.4826 * mad
        return [(value - centre) / scale for value in transformed]

    if len(transformed) < 2:
        return [0.0]

    scale = statistics.pstdev(transformed)
    if scale <= 1e-12:
        return [0.0 for _ in transformed]

    mean = statistics.fmean(transformed)
    return [(value - mean) / scale for value in transformed]


def aggregate_packet_windows(
    packets,
    clip_length_s=30.0,
    window_stride_s=5.0,
    intervals=None,
):
    """Aggregate packet statistics into fixed candidate windows.

    Every valid window is retained.  Ranking therefore prioritises likely
    events without deleting ordinary-driving windows from the sampling frame.
    """
    clip_length_s = float(clip_length_s)
    window_stride_s = float(window_stride_s)

    if clip_length_s <= 0:
        raise ValueError("clip_length_s must be positive")
    if window_stride_s <= 0:
        raise ValueError("window_stride_s must be positive")
    if not packets:
        return []

    ordered = sorted(packets, key=lambda packet: float(packet["pts_time_s"]))
    times = [float(packet["pts_time_s"]) for packet in ordered]
    sizes = [max(0, int(packet["size_bytes"])) for packet in ordered]
    keyframes = [1 if packet.get("is_keyframe") else 0 for packet in ordered]

    positive_steps = [
        current - previous
        for previous, current in zip(times, times[1:])
        if current > previous
    ]
    tail_s = statistics.median(positive_steps) if positive_steps else 0.0
    duration_s = times[-1] + tail_s

    allowed_intervals = normalise_intervals(intervals)

    if intervals is not None and not allowed_intervals:
        return []

    if not allowed_intervals:
        allowed_intervals = [{"start_s": 0.0, "end_s": duration_s}]

    byte_prefix = [0]
    keyframe_prefix = [0]

    for size_bytes, is_keyframe in zip(sizes, keyframes):
        byte_prefix.append(byte_prefix[-1] + size_bytes)
        keyframe_prefix.append(keyframe_prefix[-1] + is_keyframe)

    windows = []

    for interval in allowed_intervals:
        interval_start_s = interval["start_s"]
        interval_end_s = min(interval["end_s"], duration_s)
        start_s = interval_start_s

        while start_s + clip_length_s <= interval_end_s + 1e-9:
            end_s = start_s + clip_length_s
            left = bisect_left(times, start_s)
            right = bisect_left(times, end_s)

            if right > left:
                packet_bytes = byte_prefix[right] - byte_prefix[left]
                packet_count = right - left
                keyframe_count = keyframe_prefix[right] - keyframe_prefix[left]
                peak_packet_size = max(sizes[left:right])

                windows.append(
                    {
                        "mapping_interval_start_s": round(interval_start_s, 6),
                        "mapping_interval_end_s": round(interval_end_s, 6),
                        "t_start_s": round(start_s, 6),
                        "t_end_s": round(end_s, 6),
                        "clip_length_s": clip_length_s,
                        "window_stride_s": window_stride_s,
                        "packet_count": packet_count,
                        "keyframe_count": keyframe_count,
                        "packet_bytes": packet_bytes,
                        "packet_bytes_per_s": packet_bytes / clip_length_s,
                        "mean_packet_size_bytes": packet_bytes / packet_count,
                        "peak_packet_size_bytes": peak_packet_size,
                    }
                )

            start_s += window_stride_s

    scores = robust_scores([row["packet_bytes_per_s"] for row in windows])
    ranked_indices = sorted(
        range(len(windows)),
        key=lambda index: (-scores[index], windows[index]["t_start_s"]),
    )
    ranks = {index: rank for rank, index in enumerate(ranked_indices, start=1)}
    denominator = max(1, len(windows) - 1)

    for index, row in enumerate(windows):
        rank = ranks[index]
        row["candidate_score"] = round(float(scores[index]), 8)
        row["candidate_percentile"] = round(
            1.0 - ((rank - 1) / denominator),
            8,
        )
        row["candidate_rank"] = rank
        row["packet_candidate_score"] = row["candidate_score"]
        row["packet_candidate_percentile"] = row["candidate_percentile"]
        row["packet_candidate_rank"] = row["candidate_rank"]

    return windows


def index_video(
    video_record,
    clip_length_s=30.0,
    window_stride_s=5.0,
    ffprobe_path=None,
):
    """Build index rows for one source video and attach audit metadata."""
    source_video = os.path.abspath(str(video_record["source_video"]))
    source_stat = os.stat(source_video)
    intervals = normalise_intervals(video_record.get("intervals"))

    if not intervals:
        return []

    packets = probe_video_packets(
        source_video,
        ffprobe_path=ffprobe_path,
        intervals=intervals,
    )
    windows = aggregate_packet_windows(
        packets,
        clip_length_s=clip_length_s,
        window_stride_s=window_stride_s,
        intervals=intervals,
    )

    rows = []

    for window in windows:
        row = {
            "index_version": INDEX_VERSION,
            "city": str(video_record.get("city", "")),
            "country": str(video_record.get("country", "")),
            "continent": str(video_record.get("continent", "")),
            "video_id": str(video_record.get("video_id", "")),
            "source_video": source_video,
            "source_size_bytes": int(source_stat.st_size),
            "source_mtime_ns": int(source_stat.st_mtime_ns),
            "selection_source": "ffprobe_packet_energy",
        }
        row.update(window)
        rows.append(row)

    return rows


def attach_event_clusters(rows, separation_s=None):
    """Attach ranked, representative centred temporal event clusters.

    Sliding windows around one scene often occupy several adjacent ranks. The
    best ranked window becomes the event representative and claims overlapping
    neighbours inside ``separation_s``. This is deliberately non transitive:
    a chain of five second windows cannot merge an entire drive into one event.

    Every input row is retained for audit and negative sampling. Downstream job
    selection can use only rows marked ``is_event_representative``.
    """
    if not rows:
        return rows

    configured_separation_s = None
    if separation_s is not None:
        configured_separation_s = float(separation_s)
        if not math.isfinite(configured_separation_s) or configured_separation_s <= 0:
            raise ValueError("separation_s must be positive")

    partitions = {}

    for index, row in enumerate(rows):
        key = (
            str(row.get("video_id", "")),
            float(row.get("mapping_interval_start_s", 0.0)),
            float(row.get("mapping_interval_end_s", float("inf"))),
        )
        partitions.setdefault(key, []).append(index)

    event_counts = {}

    for partition_key, partition_indices in partitions.items():
        unassigned = set(partition_indices)
        video_id = partition_key[0] or "video"

        while unassigned:
            representative_index = min(
                unassigned,
                key=lambda index: (
                    int(rows[index].get("candidate_rank", 10**12)),
                    float(rows[index].get("t_start_s", 0.0)),
                ),
            )
            representative = rows[representative_index]
            representative_start_s = float(representative["t_start_s"])
            representative_rank = int(representative.get("candidate_rank", 0))
            current_separation_s = configured_separation_s

            if current_separation_s is None:
                current_separation_s = float(representative.get("clip_length_s", 0.0))
                if not math.isfinite(current_separation_s) or current_separation_s <= 0:
                    raise ValueError("clip_length_s must be positive for event clustering")

            cluster_indices = [
                index
                for index in unassigned
                if abs(float(rows[index]["t_start_s"]) - representative_start_s)
                < current_separation_s
            ]
            event_number = event_counts.get(video_id, 0) + 1
            event_counts[video_id] = event_number
            event_cluster_id = "%s_event_%04d" % (
                video_id,
                event_number,
            )
            event_start_s = min(float(rows[index]["t_start_s"]) for index in cluster_indices)
            event_end_s = max(float(rows[index]["t_end_s"]) for index in cluster_indices)
            event_window_count = len(cluster_indices)

            for index in cluster_indices:
                row = rows[index]
                row["event_cluster_id"] = event_cluster_id
                row["event_start_s"] = event_start_s
                row["event_end_s"] = event_end_s
                row["event_representative_start_s"] = representative_start_s
                row["event_representative_rank"] = representative_rank
                row["event_window_count"] = event_window_count
                row["is_event_representative"] = index == representative_index

            unassigned.difference_update(cluster_indices)

    return rows


def write_parquet_index(rows, output_path):
    """Write a complete index atomically so readers never see a partial file."""
    import pandas as pd

    output_path = os.path.abspath(str(output_path))
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    frame = pd.DataFrame(rows, columns=INDEX_COLUMNS)
    temporary_path = output_path + ".tmp"

    try:
        frame.to_parquet(temporary_path, index=False, engine="pyarrow")
        os.replace(temporary_path, output_path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)

    _INDEX_CACHE.pop(output_path, None)
    return output_path


def _load_index_by_video(index_path):
    import pandas as pd

    index_path = os.path.abspath(str(index_path))
    modified_ns = os.stat(index_path).st_mtime_ns
    cached = _INDEX_CACHE.get(index_path)

    if cached is not None and cached[0] == modified_ns:
        return cached[1]

    frame = pd.read_parquet(index_path, engine="pyarrow")
    missing = REQUIRED_QUERY_COLUMNS.difference(frame.columns)

    if missing:
        raise ValueError(
            "Candidate index is missing columns: " + ", ".join(sorted(missing))
        )

    grouped = {}

    for record in frame.to_dict(orient="records"):
        video_id = str(record.get("video_id", ""))
        grouped.setdefault(video_id, []).append(record)

    for video_rows in grouped.values():
        video_rows.sort(
            key=lambda row: (
                int(row.get("candidate_rank", 10**12)),
                float(row.get("t_start_s", 0.0)),
            )
        )

    _INDEX_CACHE[index_path] = (modified_ns, grouped)
    return grouped


def _true_value(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, float) and not math.isfinite(value):
        return False
    return str(value).strip().lower() in ("1", "true", "yes", "y", "on")


def _optional_float(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _optional_int(value):
    try:
        if value is None or (isinstance(value, float) and not math.isfinite(value)):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_string(value):
    if value is None:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    text = str(value).strip()
    return text or None


def select_indexed_windows(
    rows,
    interval_start_s,
    interval_end_s,
    clip_length_s,
    max_windows=0,
    min_start_gap_s=0.0,
):
    """Filter ranked index rows to one valid mapping interval."""
    interval_start_s = float(interval_start_s)
    interval_end_s = float(interval_end_s)
    clip_length_s = float(clip_length_s)
    max_windows = int(max_windows)
    min_start_gap_s = max(0.0, float(min_start_gap_s))
    selected = []
    has_event_clusters = any(
        _true_value(row.get("is_event_representative")) for row in rows
    )
    eligible_rows = [
        row
        for row in rows
        if not has_event_clusters or _true_value(row.get("is_event_representative"))
    ]
    eligible_rows.sort(
        key=lambda row: (
            int(row.get("candidate_rank", 10**12)),
            float(row.get("t_start_s", 0.0)),
        )
    )

    for row in eligible_rows:
        start_s = float(row["t_start_s"])
        end_s = float(row["t_end_s"])
        indexed_length_s = float(row["clip_length_s"])
        semantic_score = row.get("semantic_score")

        try:
            semantic_score = float(semantic_score)
        except (TypeError, ValueError):
            semantic_score = None

        if semantic_score is not None and not math.isfinite(semantic_score):
            semantic_score = None

        if abs(indexed_length_s - clip_length_s) > 1e-6:
            continue
        if start_s < interval_start_s or end_s > interval_end_s:
            continue
        if any(abs(start_s - item["start_s"]) < min_start_gap_s for item in selected):
            continue

        selected.append(
            {
                "start_s": start_s,
                "end_s": end_s,
                "selection_score": float(row["candidate_score"]),
                "selection_rank": int(row["candidate_rank"]),
                "packet_bytes_per_s": float(row.get("packet_bytes_per_s", 0.0)),
                "packet_candidate_score": float(
                    row.get("packet_candidate_score", row["candidate_score"])
                ),
                "packet_candidate_rank": int(
                    row.get("packet_candidate_rank", row["candidate_rank"])
                ),
                "semantic_score": semantic_score,
                "event_cluster_id": _optional_string(
                    row.get("event_cluster_id")
                ),
                "event_start_s": _optional_float(row.get("event_start_s")),
                "event_end_s": _optional_float(row.get("event_end_s")),
                "event_representative_start_s": _optional_float(
                    row.get("event_representative_start_s")
                ),
                "event_representative_rank": _optional_int(
                    row.get("event_representative_rank")
                ),
                "event_window_count": _optional_int(row.get("event_window_count")),
                "selection_source": str(
                    row.get("selection_source", "ffprobe_packet_energy")
                ),
            }
        )

        if max_windows > 0 and len(selected) >= max_windows:
            break

    return selected


def candidate_windows_for_video(
    index_path,
    video_id,
    interval_start_s,
    interval_end_s,
    clip_length_s,
    max_windows=0,
    min_start_gap_s=0.0,
):
    """Read ranked windows for one video without opening the source video."""
    grouped = _load_index_by_video(index_path)
    rows = grouped.get(str(video_id), [])
    return select_indexed_windows(
        rows,
        interval_start_s=interval_start_s,
        interval_end_s=interval_end_s,
        clip_length_s=clip_length_s,
        max_windows=max_windows,
        min_start_gap_s=min_start_gap_s,
    )
