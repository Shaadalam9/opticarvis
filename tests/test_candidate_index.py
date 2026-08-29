"""CPU-only guards for offline candidate indexing and selection."""

import csv
import json
import os
import sys
import tempfile


SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")

if SRC not in sys.path:
    sys.path.insert(0, SRC)

import candidate_index


def load_index_builder():
    import build_candidate_index

    return build_candidate_index


def synthetic_packets():
    packets = []

    for second in range(70):
        size_bytes = 100
        if 40 <= second < 43:
            size_bytes = 10000

        packets.append(
            {
                "pts_time_s": float(second),
                "size_bytes": size_bytes,
                "is_keyframe": second % 10 == 0,
            }
        )

    return packets


def test_packet_spike_ranks_windows_without_dropping_baselines():
    rows = candidate_index.aggregate_packet_windows(
        synthetic_packets(),
        clip_length_s=10.0,
        window_stride_s=5.0,
    )

    assert len(rows) == 13
    assert sorted(row["candidate_rank"] for row in rows) == list(range(1, 14))

    best = min(rows, key=lambda row: row["candidate_rank"])
    assert best["t_start_s"] <= 40.0 < best["t_end_s"]

    quiet_rows = [row for row in rows if row["t_end_s"] <= 40.0]
    assert quiet_rows, "ordinary-driving windows must remain in the index"


def test_ranked_selection_respects_interval_length_and_gap():
    rows = candidate_index.aggregate_packet_windows(
        synthetic_packets(),
        clip_length_s=10.0,
        window_stride_s=5.0,
    )

    selected = candidate_index.select_indexed_windows(
        rows,
        interval_start_s=10.0,
        interval_end_s=65.0,
        clip_length_s=10.0,
        max_windows=3,
        min_start_gap_s=10.0,
    )

    assert len(selected) == 3
    assert all(10.0 <= row["start_s"] for row in selected)
    assert all(row["end_s"] <= 65.0 for row in selected)

    starts = [row["start_s"] for row in selected]
    for left_index, left in enumerate(starts):
        for right in starts[left_index + 1:]:
            assert abs(left - right) >= 10.0


def test_event_clustering_preserves_windows_and_marks_distinct_representatives():
    rows = [
        {
            "video_id": "test_video",
            "mapping_interval_start_s": 0.0,
            "mapping_interval_end_s": 120.0,
            "t_start_s": start_s,
            "t_end_s": start_s + 30.0,
            "clip_length_s": 30.0,
            "candidate_score": score,
            "candidate_rank": rank,
            "packet_candidate_score": score,
            "packet_candidate_rank": rank,
        }
        for start_s, score, rank in [
            (0.0, 0.1, 4),
            (5.0, 0.3, 2),
            (10.0, 0.5, 1),
            (15.0, 0.2, 3),
            (35.0, 0.1, 5),
            (60.0, 0.0, 6),
        ]
    ]

    candidate_index.attach_event_clusters(rows, separation_s=30.0)

    assert len(rows) == 6, "clustering must retain every raw index window"
    representatives = [row for row in rows if row["is_event_representative"]]
    assert [row["t_start_s"] for row in representatives] == [10.0, 60.0]
    assert representatives[0]["event_window_count"] == 5
    assert representatives[0]["event_start_s"] == 0.0
    assert representatives[0]["event_end_s"] == 65.0
    assert len({row["event_cluster_id"] for row in rows}) == 2


def test_event_index_selection_returns_only_representatives():
    rows = [
        {
            "video_id": "test_video",
            "mapping_interval_start_s": 0.0,
            "mapping_interval_end_s": 120.0,
            "t_start_s": start_s,
            "t_end_s": start_s + 30.0,
            "clip_length_s": 30.0,
            "candidate_score": score,
            "candidate_rank": rank,
            "packet_candidate_score": score,
            "packet_candidate_rank": rank,
        }
        for start_s, score, rank in [
            (0.0, 0.1, 4),
            (5.0, 0.3, 2),
            (10.0, 0.5, 1),
            (15.0, 0.2, 3),
            (35.0, 0.1, 5),
            (60.0, 0.0, 6),
        ]
    ]
    candidate_index.attach_event_clusters(rows, separation_s=30.0)

    selected = candidate_index.select_indexed_windows(
        rows,
        interval_start_s=0.0,
        interval_end_s=120.0,
        clip_length_s=30.0,
        max_windows=10,
        min_start_gap_s=0.0,
    )

    assert [row["start_s"] for row in selected] == [10.0, 60.0]
    assert all(row["event_cluster_id"] for row in selected)
    assert [row["event_window_count"] for row in selected] == [5, 1]


def test_packet_windows_are_created_only_inside_mapping_intervals():
    rows = candidate_index.aggregate_packet_windows(
        synthetic_packets(),
        clip_length_s=10.0,
        window_stride_s=5.0,
        intervals=[
            {"start_s": 10.0, "end_s": 35.0},
            {"start_s": 45.0, "end_s": 65.0},
        ],
    )

    assert [row["t_start_s"] for row in rows] == [
        10.0,
        15.0,
        20.0,
        25.0,
        45.0,
        50.0,
        55.0,
    ]
    assert all(
        row["mapping_interval_start_s"] <= row["t_start_s"]
        and row["t_end_s"] <= row["mapping_interval_end_s"]
        for row in rows
    )


def test_four_parameter_space_has_fourteen_iteration_budget():
    path = os.path.join(SRC, "..", "configs", "bo_render_space.json")

    with open(path, "r", encoding="utf-8") as handle:
        space = json.load(handle)

    parameters = set(space["parameters"])
    assert parameters == {
        "mask_alpha",
        "trajectory_alpha",
        "background_dim_alpha",
        "palette_id",
    }
    assert 2 * (len(parameters) + 1) + 4 == 14


def test_mapping_video_records_can_be_filtered_to_local_sources():
    builder = load_index_builder()
    directory = tempfile.mkdtemp()
    mapping_path = os.path.join(directory, "mapping.csv")
    local_video = os.path.join(directory, "localVideo1.mp4")

    with open(mapping_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "locality",
                "country",
                "continent",
                "videos",
                "start_time",
                "end_time",
            ]
        )
        writer.writerow(
            [
                "Test City",
                "Test Country",
                "Europe",
                "[localVideo1,missingVid2]",
                "[[15,90],[25]]",
                "[[55,140],[70]]",
            ]
        )

    with open(local_video, "wb") as handle:
        handle.write(b"test")

    records = builder.video_records(mapping_path, directory)
    local_records = [
        record for record in records if os.path.isfile(record["source_video"])
    ]

    assert len(records) == 2
    assert [record["video_id"] for record in local_records] == ["localVideo1"]
    assert records[0]["intervals"] == [
        {"start_s": 15.0, "end_s": 55.0},
        {"start_s": 90.0, "end_s": 140.0},
    ]
    assert records[1]["intervals"] == [
        {"start_s": 25.0, "end_s": 70.0}
    ]


def test_empty_index_can_be_written_with_a_stable_schema():
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        return

    directory = tempfile.mkdtemp()
    path = os.path.join(directory, "candidate_index.parquet")
    candidate_index.write_parquet_index([], path)

    import pandas as pd

    frame = pd.read_parquet(path, engine="pyarrow")
    assert list(frame.columns) == candidate_index.INDEX_COLUMNS
    assert "event_cluster_id" in frame.columns
    assert "is_event_representative" in frame.columns


def test_written_index_can_be_queried_without_source_video():
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        return

    rows = candidate_index.aggregate_packet_windows(
        synthetic_packets(),
        clip_length_s=10.0,
        window_stride_s=5.0,
    )

    for row in rows:
        row.update(
            {
                "index_version": candidate_index.INDEX_VERSION,
                "city": "Test City",
                "country": "Test Country",
                "continent": "Test Continent",
                "video_id": "test_video",
                "source_video": "/source/does/not/need/to/exist.mp4",
                "source_size_bytes": 123,
                "source_mtime_ns": 456,
                "selection_source": "ffprobe_packet_energy",
            }
        )

    directory = tempfile.mkdtemp()
    path = os.path.join(directory, "candidate_index.parquet")
    candidate_index.write_parquet_index(rows, path)
    selected = candidate_index.candidate_windows_for_video(
        path,
        "test_video",
        interval_start_s=0.0,
        interval_end_s=70.0,
        clip_length_s=10.0,
        max_windows=2,
        min_start_gap_s=10.0,
    )

    assert len(selected) == 2
    assert all("selection_score" in row for row in selected)


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
