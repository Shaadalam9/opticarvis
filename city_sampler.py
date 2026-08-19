"""
City sampling for OptiCarVis using Mark Colley's `lpm-sampling` package.

Repository:
    https://github.com/M-Colley/lpm-sampling

Install:
    pip install lpm-sampling

Design:
    1. Start from the full city frame.
    2. Keep only qualifying footage:
           vehicle_type == 0
           time_of_day == 0
    3. Require at least MIN_FOOTAGE_HOURS of qualifying footage.
    4. Use population_locality ** ALPHA as the size measure.
    5. Use lpm_sampling.pi_from_size() to obtain fixed size inclusion probabilities.
    6. Convert latitude/longitude with lpm_sampling.unit_sphere_xyz().
    7. Draw the sample with lpm_sampling.lpm2() or lpm_sampling.lpm1().
    8. Save a clean sample_manifest.csv and a separate design_manifest.csv.

There is NO one city per country constraint.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from lpm_sampling import (
        lpm1,
        lpm2,
        nearest_distance_km,
        pi_from_size,
        spatial_balance,
        unit_sphere_xyz,
    )
except ImportError as exc:
    raise ImportError(
        "The 'lpm-sampling' package is required. Install it with:\n"
        "    pip install lpm-sampling\n"
        "or directly from Mark's repository:\n"
        "    pip install git+https://github.com/M-Colley/lpm-sampling.git"
    ) from exc


# ============================================================
# SETTINGS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_CSV = PROJECT_ROOT / "docs" / "mapping_original.csv"
OUTPUT_DIR = PROJECT_ROOT / "lpm_city_sample"

SAMPLE_SIZE = 100
SEED = 20260818

# Population damping.
# 1.00 = strict population proportional to size
# 0.75 = moderate damping
# 0.50 = stronger damping
ALPHA = 0.75

# "lpm2" is the practical default.
# Change to "lpm1" for mutual nearest neighbour pairing.
METHOD = "lpm2"

POPULATION_COLUMN = "population_locality"

VEHICLE_TYPE_COLUMN = "vehicle_type"
TIME_OF_DAY_COLUMN = "time_of_day"
START_TIME_COLUMN = "start_time"
END_TIME_COLUMN = "end_time"

REQUIRED_VEHICLE_TYPE = 0
REQUIRED_TIME_OF_DAY = 0

MIN_FOOTAGE_HOURS = 1.0

REMOVE_ZERO_ZERO = True


# Columns not wanted in the clean final sample manifest.
COLUMNS_TO_REMOVE_FROM_SAMPLE_MANIFEST = [
    "source_row",
    "upload_date",
    "channel",
    "footage_hours_used",
    "size_measure",
    "pi",
    "design_weight",
    "sphere_x",
    "sphere_y",
    "sphere_z",
]


# ============================================================
# MEDIA HELPERS
# ============================================================

def parse_json_list(value):
    """Parse a JSON encoded list. Return an empty list when parsing fails."""

    if pd.isna(value):
        return []

    try:
        parsed = json.loads(str(value))
    except (json.JSONDecodeError, TypeError, ValueError):
        return []

    return parsed if isinstance(parsed, list) else []


def parse_video_ids(value):
    """
    Parse the dataset's videos field.

    It contains values such as:
        [video_a,video_b,video_c]
    """

    if pd.isna(value):
        return []

    text = str(value).strip()

    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]

    if not text.strip():
        return []

    return [
        item.strip()
        for item in text.split(",")
        if item.strip()
    ]


def qualifying_intervals(row):
    """
    Return qualifying intervals grouped by video.

    A qualifying interval must satisfy both:
        vehicle_type == 0
        time_of_day == 0
    """

    vehicle_types = parse_json_list(
        row[VEHICLE_TYPE_COLUMN]
    )
    time_of_day = parse_json_list(
        row[TIME_OF_DAY_COLUMN]
    )
    starts = parse_json_list(
        row[START_TIME_COLUMN]
    )
    ends = parse_json_list(
        row[END_TIME_COLUMN]
    )

    videos = parse_video_ids(
        row.get("videos", "[]")
    )

    output = []

    for video_index, (
        vehicle,
        day_group,
        start_group,
        end_group,
    ) in enumerate(
        zip(
            vehicle_types,
            time_of_day,
            starts,
            ends,
        )
    ):
        try:
            vehicle = int(vehicle)
        except (TypeError, ValueError):
            continue

        if vehicle != REQUIRED_VEHICLE_TYPE:
            continue

        if not all(
            isinstance(group, list)
            for group in [
                day_group,
                start_group,
                end_group,
            ]
        ):
            continue

        intervals = []

        for day_value, start, end in zip(
            day_group,
            start_group,
            end_group,
        ):
            try:
                day_value = int(day_value)
                start_value = float(start)
                end_value = float(end)
            except (TypeError, ValueError):
                continue

            if day_value != REQUIRED_TIME_OF_DAY:
                continue

            if (
                not np.isfinite(start_value)
                or not np.isfinite(end_value)
                or end_value < start_value
            ):
                continue

            intervals.append(
                (
                    start_value,
                    end_value,
                    start,
                    end,
                )
            )

        if not intervals:
            continue

        video_id = (
            videos[video_index]
            if video_index < len(videos)
            else ""
        )

        output.append(
            {
                "video": video_id,
                "vehicle_type": REQUIRED_VEHICLE_TYPE,
                "intervals": intervals,
            }
        )

    return output


def merged_duration_seconds(intervals):
    """Calculate duration after merging overlapping intervals."""

    if not intervals:
        return 0.0

    numeric_intervals = sorted(
        (float(start), float(end))
        for start, end, _, _ in intervals
    )

    merged_start, merged_end = numeric_intervals[0]
    total = 0.0

    for start, end in numeric_intervals[1:]:
        if start <= merged_end:
            merged_end = max(
                merged_end,
                end,
            )
        else:
            total += merged_end - merged_start
            merged_start, merged_end = start, end

    total += merged_end - merged_start

    return total


def qualifying_footage_hours(row):
    """
    Total qualifying footage hours for one city.

    Durations are derived from start_time/end_time, assumed to be seconds.
    Overlapping intervals within the same video are merged.
    """

    total_seconds = 0.0

    for video_record in qualifying_intervals(row):
        total_seconds += merged_duration_seconds(
            video_record["intervals"]
        )

    return total_seconds / 3600.0


def filter_media_for_export(row):
    """
    Keep only media records satisfying vehicle_type == 0 and time_of_day == 0.

    This changes only the exported clean sample manifest.
    """

    records = qualifying_intervals(row)

    videos = []
    vehicle_types = []
    time_groups = []
    start_groups = []
    end_groups = []

    for record in records:
        videos.append(
            record["video"]
        )
        vehicle_types.append(
            REQUIRED_VEHICLE_TYPE
        )

        days = []
        starts = []
        ends = []

        for _, _, original_start, original_end in record["intervals"]:
            days.append(
                REQUIRED_TIME_OF_DAY
            )
            starts.append(
                original_start
            )
            ends.append(
                original_end
            )

        time_groups.append(days)
        start_groups.append(starts)
        end_groups.append(ends)

    return {
        "videos": "[" + ",".join(videos) + "]",
        VEHICLE_TYPE_COLUMN: json.dumps(
            vehicle_types,
            separators=(",", ":"),
        ),
        TIME_OF_DAY_COLUMN: json.dumps(
            time_groups,
            separators=(",", ":"),
        ),
        START_TIME_COLUMN: json.dumps(
            start_groups,
            separators=(",", ":"),
        ),
        END_TIME_COLUMN: json.dumps(
            end_groups,
            separators=(",", ":"),
        ),
    }


# ============================================================
# FRAME PREPARATION
# ============================================================

def prepare_frame(df):
    """
    Apply all eligibility criteria BEFORE assigning inclusion probabilities.
    """

    required_columns = {
        "locality",
        "country",
        "lat",
        "lon",
        POPULATION_COLUMN,
        VEHICLE_TYPE_COLUMN,
        TIME_OF_DAY_COLUMN,
        START_TIME_COLUMN,
        END_TIME_COLUMN,
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    frame = df.copy()

    frame["lat"] = pd.to_numeric(
        frame["lat"],
        errors="coerce",
    )
    frame["lon"] = pd.to_numeric(
        frame["lon"],
        errors="coerce",
    )
    frame[POPULATION_COLUMN] = pd.to_numeric(
        frame[POPULATION_COLUMN],
        errors="coerce",
    )

    frame = frame.dropna(
        subset=[
            "locality",
            "country",
            "lat",
            "lon",
            POPULATION_COLUMN,
        ]
    ).copy()

    frame = frame[
        frame["lat"].between(-90.0, 90.0)
        & frame["lon"].between(-180.0, 180.0)
    ].copy()

    if REMOVE_ZERO_ZERO:
        frame = frame[
            ~(
                np.isclose(frame["lat"], 0.0)
                & np.isclose(frame["lon"], 0.0)
            )
        ].copy()

    # Population must be positive because it defines the PPS size measure.
    frame = frame[
        frame[POPULATION_COLUMN] > 0
    ].copy()

    # Calculate qualifying footage BEFORE sampling.
    frame["footage_hours_used"] = frame.apply(
        qualifying_footage_hours,
        axis=1,
    )

    frame = frame[
        frame["footage_hours_used"]
        >= MIN_FOOTAGE_HOURS
    ].copy()

    frame = frame.reset_index().rename(
        columns={"index": "source_row"}
    )

    if len(frame) < SAMPLE_SIZE:
        raise ValueError(
            f"Only {len(frame)} eligible cities remain, "
            f"but SAMPLE_SIZE={SAMPLE_SIZE}."
        )

    return frame


# ============================================================
# LPM SAMPLING USING M-COLLEY/lpm-sampling
# ============================================================

def draw_lpm_sample(frame):
    """
    Calculate inclusion probabilities and perform the LPM draw.

    The LPM implementation itself comes from the `lpm-sampling` package.
    """

    population = frame[
        POPULATION_COLUMN
    ].to_numpy(dtype=float)

    # Population damping is applied before PPS probabilities.
    size_measure = np.power(
        population,
        ALPHA,
    )

    # Mark's package performs the iterative certainty unit capping.
    pi, certainty_indices = pi_from_size(
        size_measure,
        n=SAMPLE_SIZE,
    )

    # Mark's package converts degrees to unit sphere x/y/z coordinates.
    xyz = unit_sphere_xyz(
        frame["lat"].to_numpy(dtype=float),
        frame["lon"].to_numpy(dtype=float),
    )

    rng = np.random.default_rng(
        SEED
    )

    if METHOD.lower() == "lpm2":
        selected_indices = lpm2(
            pi,
            xyz,
            rng=rng,
        )
    elif METHOD.lower() == "lpm1":
        selected_indices = lpm1(
            pi,
            xyz,
            rng=rng,
        )
    else:
        raise ValueError(
            "METHOD must be 'lpm1' or 'lpm2'."
        )

    if len(selected_indices) != SAMPLE_SIZE:
        raise RuntimeError(
            f"LPM selected {len(selected_indices)} cities, "
            f"expected {SAMPLE_SIZE}."
        )

    return (
        selected_indices,
        pi,
        certainty_indices,
        xyz,
        size_measure,
    )


# ============================================================
# OUTPUT
# ============================================================

def make_clean_sample_manifest(sample):
    """
    Create the user facing CSV.

    Statistical design information is intentionally kept in design_manifest.csv
    rather than discarded.
    """

    clean_sample = sample.copy()

    # Replace media fields with only vehicle type 0 / time of day 0 footage.
    for row_index in clean_sample.index:
        filtered_media = filter_media_for_export(
            clean_sample.loc[row_index]
        )

        for column, value in filtered_media.items():
            if column in clean_sample.columns:
                clean_sample.at[
                    row_index,
                    column,
                ] = value

    # Remove unwanted internal/statistical columns.
    clean_sample = clean_sample.drop(
        columns=[
            column
            for column in COLUMNS_TO_REMOVE_FROM_SAMPLE_MANIFEST
            if column in clean_sample.columns
        ]
    )

    # The source CSV's original id is not wanted.
    if "id" in clean_sample.columns:
        clean_sample = clean_sample.drop(
            columns=["id"]
        )

    # Fresh simple ID: 1, 2, 3, ...
    clean_sample.insert(
        0,
        "id",
        np.arange(
            1,
            len(clean_sample) + 1,
        ),
    )

    return clean_sample


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(
        INPUT_CSV
    )

    frame = prepare_frame(
        df
    )

    (
        selected_indices,
        pi,
        certainty_indices,
        xyz,
        size_measure,
    ) = draw_lpm_sample(
        frame
    )

    # Keep design information on the complete eligible frame.
    frame["size_measure"] = size_measure
    frame["pi"] = pi

    frame["design_weight"] = np.where(
        pi > 0,
        1.0 / pi,
        np.inf,
    )

    frame["sphere_x"] = xyz[:, 0]
    frame["sphere_y"] = xyz[:, 1]
    frame["sphere_z"] = xyz[:, 2]

    sample = frame.iloc[
        selected_indices
    ].copy()

    # Keep the design manifest in deterministic source frame order.
    sample = sample.sort_index().reset_index(
        drop=True
    )

    clean_sample = make_clean_sample_manifest(
        sample
    )

    eligible_frame_path = (
        OUTPUT_DIR
        / "eligible_frame_with_pi.csv"
    )

    design_manifest_path = (
        OUTPUT_DIR
        / "design_manifest.csv"
    )

    sample_manifest_path = (
        "mapping.csv"
    )

    diagnostics_path = (
        OUTPUT_DIR
        / "diagnostics.txt"
    )

    country_counts_path = (
        OUTPUT_DIR
        / "selected_cities_by_country.csv"
    )

    frame.to_csv(
        eligible_frame_path,
        index=False,
    )

    sample.to_csv(
        design_manifest_path,
        index=False,
    )

    clean_sample.to_csv(
        sample_manifest_path,
        index=False,
    )

    country_counts = (
        clean_sample["country"]
        .value_counts()
        .rename_axis("country")
        .reset_index(name="selected_city_count")
    )

    country_counts.to_csv(
        country_counts_path,
        index=False,
    )

    # Diagnostics supplied by Mark's package.
    balance_score = spatial_balance(
        pi,
        xyz,
        selected_indices,
    )

    nearest_km = nearest_distance_km(
        xyz,
        xyz[selected_indices],
    )

    population = frame[
        POPULATION_COLUMN
    ].to_numpy(dtype=float)

    population_weighted_nearest_km = float(
        np.average(
            nearest_km,
            weights=population,
        )
    )

    sampled_weights = (
        1.0
        / pi[selected_indices]
    )

    effective_n = float(
        sampled_weights.sum() ** 2
        / np.sum(sampled_weights ** 2)
    )

    diagnostics = [
        "LPM CITY SAMPLING DIAGNOSTICS",
        "=============================",
        "Implementation: M-Colley/lpm-sampling",
        f"Input rows: {len(df)}",
        f"Eligible rows: {len(frame)}",
        f"Eligible countries: {frame['country'].nunique()}",
        f"Required vehicle type: {REQUIRED_VEHICLE_TYPE}",
        f"Required time of day: {REQUIRED_TIME_OF_DAY}",
        f"Minimum qualifying footage hours: {MIN_FOOTAGE_HOURS}",
        f"Sample size: {SAMPLE_SIZE}",
        f"Method: {METHOD.upper()}",
        f"Seed: {SEED}",
        f"Population column: {POPULATION_COLUMN}",
        f"Alpha: {ALPHA}",
        f"sum(pi): {pi.sum():.12f}",
        f"Certainty cities: {len(certainty_indices)}",
        f"Selected cities: {len(selected_indices)}",
        f"Selected countries: {clean_sample['country'].nunique()}",
        f"Spatial balance: {balance_score:.6f}",
        (
            "Population weighted distance to nearest selected city (km): "
            f"{population_weighted_nearest_km:.2f}"
        ),
        f"Kish effective sample size: {effective_n:.2f}",
        f"Minimum sampled design weight: {sampled_weights.min():.2f}",
        f"Maximum sampled design weight: {sampled_weights.max():.2f}",
    ]

    diagnostics_path.write_text(
        "\n".join(diagnostics),
        encoding="utf-8",
    )

    print(
        "\n".join(diagnostics)
    )

    print()
    print(
        f"Clean sample manifest: {sample_manifest_path}"
    )
    print(
        f"Design manifest: {design_manifest_path}"
    )
    print(
        f"Eligible frame: {eligible_frame_path}"
    )
    print(
        f"Country counts: {country_counts_path}"
    )
    print(
        f"Diagnostics: {diagnostics_path}"
    )


if __name__ == "__main__":
    main()
