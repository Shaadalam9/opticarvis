"""Convert a saved Alpamayo inference JSON into an offline WHEN policy CSV.

The video_id / segment_start_time_s are read from the JSON payload when present
(scripts/infer_crowd_clip.py writes them); the CONFIG defaults below are only
used for payloads that lack them (e.g. the NVIDIA benchmark sample from
scripts/infer_inspect_output.py).
"""

import json
import os
import sys

from alpamayo_policy_io import build_policy_row, write_policy_csv


# =============================================================================
# CONFIG — edit and run (defaults apply only when the JSON has no metadata)
# =============================================================================

INPUT_JSON = "alpamayo_inference_output.json"
OUTPUT_CSV = "alpamayo_offline_when.csv"

DEFAULT_VIDEO_ID = "benchmark_sample"
DEFAULT_SEGMENT_START_TIME_S = 0.0
WHEN_START_LOCAL_S = 0.0
WHEN_END_LOCAL_S = 3.0
MODEL_SOURCE = "oom_free_alpamayo_r1"


def main():
    if not os.path.isfile(INPUT_JSON):
        print(f"Input JSON not found: {INPUT_JSON}", file=sys.stderr)
        sys.exit(1)

    with open(INPUT_JSON, "r", encoding="utf-8") as input_file:
        payload = json.load(input_file)

    video_id = str(payload.get("video_id") or DEFAULT_VIDEO_ID)
    segment_start = payload.get("segment_start_time_s", DEFAULT_SEGMENT_START_TIME_S)
    if "video_id" not in payload:
        print(
            f"WARNING: {INPUT_JSON} has no video_id metadata; using fallback "
            f"'{DEFAULT_VIDEO_ID}'. policy_demo.py matches rows by exact video_id "
            "+ segment_start_time_s, so this CSV will not match a real CROWD clip.",
            file=sys.stderr,
        )

    row = build_policy_row(
        payload,
        video_id=video_id,
        segment_start_time_s=segment_start,
        when_start_local_s=WHEN_START_LOCAL_S,
        when_end_local_s=WHEN_END_LOCAL_S,
        model_source=MODEL_SOURCE,
    )
    write_policy_csv(OUTPUT_CSV, [row])

    print("Saved:", OUTPUT_CSV)
    print("Video id:", video_id, "| segment start:", segment_start)
    print("Reasoning:", row["reasoning_trace"])
    print("Meta action:", row["meta_action"])
    print("Reason:", row["explanation_reason"])


if __name__ == "__main__":
    main()
