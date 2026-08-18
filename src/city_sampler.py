r"""Draw a spatially balanced, population-weighted sample of cities.

The study needs a set of cities that is (a) spread over the inhabited world,
(b) weighted toward where people actually live, and (c) a real probability
sample, so that country-level analyses carry design weights and defensible
standard errors instead of the "we picked these" of a convenience set.

Selecting one city per country cannot do this. It forces India to a single
arbitrary city -- Patna rather than Delhi -- while giving Luxembourg the same
allocation, and it produces no inclusion probabilities at all.

The sampler is the LOCAL PIVOTAL METHOD, from the lpm-sampling package
(github.com/M-Colley/lpm-sampling; Grafstrom, Lundstrom & Schelin 2012) -- it
was written for this study and then extracted, so this module is the
study-facing layer: frame eligibility, imputation, diagnostics and the design
manifest. Each city gets an inclusion probability proportional to its
population; the sampler repeatedly makes a city and its nearest neighbour
compete for their combined probability, so selecting one pushes probability out
of its neighbourhood and the sample spreads itself. Every city's inclusion
probability comes out exactly as assigned, so 1/pi is a valid design weight.

Three details matter more than the algorithm choice:

1. ELIGIBILITY IS APPLIED TO THE FRAME, BEFORE THE DRAW. Restricting to cities
   with enough footage after sampling would destroy the probabilities that make
   the weights meaningful. Filter first; the sample then represents the filtered
   frame, and the write-up says so.

2. COORDINATES ARE UNIT-SPHERE x/y/z, NEVER RAW DEGREES. Longitude is circular:
   in degree space Suva at +178 and Apia at -172 are 350 degrees apart when they
   are neighbours, so a degree-space sampler believes the Pacific rim is empty
   and happily selects both. Degrees are also anisotropic -- a degree of
   longitude is 111 km at the equator and 48 km at Reykjavik. Chord distance on
   the unit sphere is monotone in great-circle distance, with no seam and no
   pole degeneracy.

3. PROBABILITIES ARE CAPPED BY ITERATION, NOT BY CLIPPING. A megacity whose
   raw pi exceeds 1 becomes a certainty unit and its excess is redistributed;
   that can push another city over 1, so it repeats until stable. Clipping once
   silently breaks sum(pi) == n.

Usage:

    python city_sampler.py --frame cities.csv --n 150 --seed 20260818 \
        --out workflow_outputs/city_sample/

The frame CSV needs locality, country, lat, lon and a population column; the
column names are all options. --min-footage-hours filters on a footage column
when the frame carries one.
"""

import argparse
import csv
import hashlib
import math
import os
import sys

import numpy as np

SRC = os.path.dirname(os.path.abspath(__file__))

if SRC not in sys.path:
    sys.path.insert(0, SRC)

from lpm_sampling import lpm, pi_from_size, spatial_balance, systematic_pps  # noqa: E402

EARTH_RADIUS_KM = 6371.0088

# Population damping. alpha = 1 is strict probability-proportional-to-size;
# alpha < 1 pulls the design toward equal probability, which reaches more
# distinct countries and narrows the weight range at some cost in efficiency
# for population totals. Whatever is used is recorded in the manifest.
DEFAULT_ALPHA = 1.0


def unit_sphere_xyz(lat_deg, lon_deg):
    """Lat/lon in degrees -> x, y, z on the unit sphere.

    Euclidean (chord) distance between these points is a strictly increasing
    function of great-circle distance: chord = 2*sin(great_circle / 2). So a
    sampler that only compares distances behaves identically to one working in
    great-circle km, without the antimeridian seam or the pole degeneracy.
    """
    lat = np.radians(np.asarray(lat_deg, dtype=np.float64))
    lon = np.radians(np.asarray(lon_deg, dtype=np.float64))
    cos_lat = np.cos(lat)

    return np.stack([cos_lat * np.cos(lon), cos_lat * np.sin(lon), np.sin(lat)], axis=1)


def great_circle_km(a_xyz, b_xyz):
    """Great-circle distance (km) between unit-sphere points, pairwise rows."""
    dot = np.clip(np.sum(a_xyz * b_xyz, axis=-1), -1.0, 1.0)

    return EARTH_RADIUS_KM * np.arccos(dot)


def nearest_selected_km(frame_xyz, selected_xyz, chunk=2048):
    """For every frame city, great-circle km to the nearest selected city."""
    out = np.empty(len(frame_xyz), dtype=np.float64)

    for start in range(0, len(frame_xyz), chunk):
        block = frame_xyz[start:start + chunk]
        dot = np.clip(block @ selected_xyz.T, -1.0, 1.0)
        out[start:start + len(block)] = EARTH_RADIUS_KM * np.arccos(dot.max(axis=1))

    return out


def read_frame(path, columns):
    """Load the frame CSV, keeping every original row for the manifest."""
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise SystemExit("Frame %s has no rows." % path)

    missing = [c for c in (columns["lat"], columns["lon"], columns["population"])
               if c not in rows[0]]

    if missing:
        raise SystemExit(
            "Frame %s lacks required column(s): %s\nAvailable: %s"
            % (path, ", ".join(missing), ", ".join(sorted(rows[0]))))

    return rows


def to_float(value, default=float("nan")):
    try:
        text = str(value).strip()
        return float(text) if text else default
    except (TypeError, ValueError):
        return default


def frame_version(path):
    digest = hashlib.sha256()

    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)

    return digest.hexdigest()[:16]


def build_frame(rows, columns, min_footage_hours):
    """Return (kept_rows, population, xyz, report).

    Eligibility runs HERE, before any probability is computed, so the design is
    a design for the filtered frame. Rows with unusable coordinates are dropped
    and counted; rows with missing or zero population are kept but flagged --
    zero is this project's missing sentinel, and dropping them would silently
    remove small cities from the frame entirely.
    """
    kept = []
    dropped_coords = 0
    dropped_footage = 0
    missing_population = 0

    for row in rows:
        lat = to_float(row.get(columns["lat"]))
        lon = to_float(row.get(columns["lon"]))

        if not (math.isfinite(lat) and math.isfinite(lon)) or abs(lat) > 90.0 or abs(lon) > 180.0:
            dropped_coords += 1
            continue

        if min_footage_hours > 0.0 and columns["footage"]:
            hours = to_float(row.get(columns["footage"]), 0.0)

            if not math.isfinite(hours) or hours < min_footage_hours:
                dropped_footage += 1
                continue

        population = to_float(row.get(columns["population"]), 0.0)

        if not math.isfinite(population) or population <= 0.0:
            missing_population += 1
            population = 0.0

        row = dict(row)
        row["_lat"] = lat
        row["_lon"] = lon
        row["_population_raw"] = population
        kept.append(row)

    if not kept:
        raise SystemExit("No frame rows survived eligibility filtering.")

    population = np.array([r["_population_raw"] for r in kept], dtype=np.float64)

    # Impute rather than drop: a zero population would make a city structurally
    # impossible to sample, quietly shrinking the frame the design describes.
    positive = population[population > 0.0]
    fill = float(np.percentile(positive, 10.0)) if len(positive) else 1.0
    imputed = population <= 0.0
    population = np.where(imputed, fill, population)

    for row, was_imputed in zip(kept, imputed):
        row["_population_imputed"] = int(bool(was_imputed))
        row["_population"] = float(fill) if was_imputed else row["_population_raw"]

    xyz = unit_sphere_xyz([r["_lat"] for r in kept], [r["_lon"] for r in kept])

    report = {
        "frame_rows": len(rows),
        "eligible": len(kept),
        "dropped_bad_coords": dropped_coords,
        "dropped_footage": dropped_footage,
        "population_imputed": int(imputed.sum()),
        "imputation_value": fill,
    }

    return kept, population, xyz, report


def draw_sample(population, xyz, n, alpha, seed, method="lpm2"):
    """Inclusion probabilities and the drawn sample."""
    sizes = np.power(np.maximum(population, 0.0), alpha)
    pi, certainty = pi_from_size(sizes, n)
    rng = np.random.default_rng(seed)
    selected = lpm(pi, xyz, method=method, rng=rng)

    return pi, selected, certainty


def coverage_curve(distance_km, population, radii=(50, 100, 250, 500, 1000)):
    """Share of frame population within X km of a selected city."""
    total = float(population.sum())

    return [(r, float(population[distance_km <= r].sum()) / total) for r in radii]


def diagnose(kept, population, xyz, pi, selected, label):
    """Everything a reviewer needs to judge the sample, measured not asserted."""
    selected_xyz = xyz[selected]
    distance = nearest_selected_km(xyz, selected_xyz)

    # The study-critical number for the participant-matching plan: how far a
    # randomly chosen PERSON (not city) is from the nearest rendered city.
    weighted_km = float((population * distance).sum() / population.sum())

    countries = {r.get("country", r.get("iso3", "")) for r in kept}
    picked_countries = {kept[i].get("country", kept[i].get("iso3", "")) for i in selected}
    weights = 1.0 / pi[selected]

    return {
        "label": label,
        "n": int(len(selected)),
        "spatial_balance_B": float(spatial_balance(pi, xyz, selected)),
        "participant_km_mean": weighted_km,
        "city_km_median": float(np.median(distance)),
        "countries_selected": len(picked_countries),
        "countries_in_frame": len(countries),
        "weight_min": float(weights.min()),
        "weight_max": float(weights.max()),
        "n_eff": float(weights.sum() ** 2 / (weights ** 2).sum()),
        "certainty_units": int((pi >= 1.0 - 1e-9).sum()),
        "coverage": coverage_curve(distance, population),
        "distance_km": distance,
    }


def print_report(report, results, alpha, seed, method):
    print("")
    print("Frame")
    print("-----")
    print("  rows in file            : %d" % report["frame_rows"])
    print("  eligible (frame sampled): %d" % report["eligible"])
    print("  dropped, bad coordinates: %d" % report["dropped_bad_coords"])
    print("  dropped, footage filter : %d" % report["dropped_footage"])
    print("  population imputed      : %d (filled with %.0f)"
          % (report["population_imputed"], report["imputation_value"]))
    print("  alpha %.2f | seed %s | method %s" % (alpha, seed, method))

    print("")
    print("Design comparison (lower spatial balance B is better)")
    print("----------------------------------------------------")
    header = "  %-28s %8s %14s %12s %10s" % (
        "design", "n", "balance B", "person-km", "countries")
    print(header)
    print("  " + "-" * (len(header) - 2))

    for item in results:
        balance = "%10.4f" % item["spatial_balance_B"]

        if item.get("balance_se"):
            balance = "%7.4f+-%.4f" % (item["spatial_balance_B"], item["balance_se"])

        print("  %-28s %8d %14s %12.1f %10d"
              % (item["label"], item["n"], balance,
                 item["participant_km_mean"], item["countries_selected"]))

    main = results[0]
    print("")
    print("Population within X km of a selected city (%s)" % main["label"])
    print("----------------------------------------------")

    for radius, share in main["coverage"]:
        print("  %5d km : %5.1f%%" % (radius, 100.0 * share))

    print("")
    print("Design weights: min %.1f, max %.1f, effective n %.1f of %d, certainty units %d"
          % (main["weight_min"], main["weight_max"], main["n_eff"], main["n"],
             main["certainty_units"]))


def write_manifest(path, kept, population, pi, selected, meta):
    """The statistical object: one row per sampled city, weights included."""
    fields = ["sample_id", "locality", "country", "iso3", "continent", "lat", "lon",
              "population_used", "population_raw", "population_imputed",
              "pi", "design_weight", "certainty", "alpha", "seed", "method",
              "frame_version", "frame_eligible_rows"]

    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()

        for rank, index in enumerate(selected, start=1):
            row = kept[index]
            writer.writerow({
                "sample_id": rank,
                "locality": row.get("locality", row.get("city", "")),
                "country": row.get("country", ""),
                "iso3": row.get("iso3", ""),
                "continent": row.get("continent", ""),
                "lat": "%.6f" % row["_lat"],
                "lon": "%.6f" % row["_lon"],
                "population_used": "%.0f" % population[index],
                "population_raw": "%.0f" % row["_population_raw"],
                "population_imputed": row["_population_imputed"],
                "pi": "%.8f" % pi[index],
                "design_weight": "%.4f" % (1.0 / pi[index]),
                "certainty": int(pi[index] >= 1.0 - 1e-9),
                "alpha": meta["alpha"],
                "seed": meta["seed"],
                "method": meta["method"],
                "frame_version": meta["frame_version"],
                "frame_eligible_rows": meta["eligible"],
            })


def compare_designs(kept, population, xyz, pi, args):
    """Mean spatial balance and matching distance over repeated draws.

    The spatially unaware baseline is randomised systematic pps: it honours the
    same inclusion probabilities and ignores the coordinates, which isolates
    exactly what the spatial method buys.
    """
    trials = {"LPM x%d (mean)" % args.replicates: [], "systematic pps x%d (mean)" % args.replicates: []}
    keys = list(trials)

    for replicate in range(args.replicates):
        rng_lpm = np.random.default_rng(args.seed + 1000 + replicate)
        rng_sys = np.random.default_rng(args.seed + 5000 + replicate)
        trials[keys[0]].append(lpm(pi, xyz, method=args.method, rng=rng_lpm))
        trials[keys[1]].append(systematic_pps(pi, rng=rng_sys))

    summary = []

    for label, draws in trials.items():
        scored = [diagnose(kept, population, xyz, pi, draw, label) for draw in draws]
        summary.append({
            "label": label,
            "n": scored[0]["n"],
            "spatial_balance_B": float(np.mean([s["spatial_balance_B"] for s in scored])),
            "balance_se": float(np.std([s["spatial_balance_B"] for s in scored], ddof=1)
                                / math.sqrt(len(scored))),
            "participant_km_mean": float(np.mean([s["participant_km_mean"] for s in scored])),
            "countries_selected": int(np.mean([s["countries_selected"] for s in scored])),
            "coverage": scored[0]["coverage"],
            "weight_min": scored[0]["weight_min"],
            "weight_max": scored[0]["weight_max"],
            "n_eff": scored[0]["n_eff"],
            "certainty_units": scored[0]["certainty_units"],
        })

    return summary


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Spatially balanced, population-weighted sample of cities.")
    parser.add_argument("--frame", required=True, help="CSV of candidate cities")
    parser.add_argument("--n", type=int, required=True, help="cities to select")
    parser.add_argument("--seed", type=int, required=True, help="reproducibility seed")
    parser.add_argument("--out", default="workflow_outputs/city_sample",
                        help="output directory for the manifest")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA,
                        help="population damping exponent (1.0 = strict PPS)")
    parser.add_argument("--method", default="lpm2", choices=("lpm1", "lpm2"))
    parser.add_argument("--lat-column", default="lat")
    parser.add_argument("--lon-column", default="lon")
    parser.add_argument("--population-column", default="population_locality")
    parser.add_argument("--footage-column", default="",
                        help="column of usable footage hours, if the frame has one")
    parser.add_argument("--min-footage-hours", type=float, default=0.0,
                        help="eligibility threshold, applied BEFORE the draw")
    parser.add_argument("--replicates", type=int, default=20,
                        help="draws per design for the comparison table (0 to skip)")

    return parser.parse_args(argv[1:])


def main(argv=None):
    args = parse_args(argv or sys.argv)
    columns = {
        "lat": args.lat_column,
        "lon": args.lon_column,
        "population": args.population_column,
        "footage": args.footage_column,
    }

    rows = read_frame(args.frame, columns)
    kept, population, xyz, report = build_frame(rows, columns, args.min_footage_hours)

    if args.n >= len(kept):
        raise SystemExit("Asked for %d cities from a frame of %d eligible rows."
                         % (args.n, len(kept)))

    pi, selected, _certainty = draw_sample(
        population, xyz, args.n, args.alpha, args.seed, args.method)

    results = [diagnose(kept, population, xyz, pi, selected, "LPM (the drawn sample)")]

    # A single draw of each design proves nothing -- the balance index varies
    # substantially between draws. Replicate both and compare the means.
    if args.replicates > 1:
        results.extend(compare_designs(kept, population, xyz, pi, args))

    print_report(report, results, args.alpha, args.seed, args.method)

    os.makedirs(args.out, exist_ok=True)
    manifest = os.path.join(args.out, "sample_manifest.csv")
    meta = {"alpha": args.alpha, "seed": args.seed, "method": args.method,
            "frame_version": frame_version(args.frame), "eligible": report["eligible"]}
    write_manifest(manifest, kept, population, pi, selected, meta)

    print("")
    print("Manifest:", manifest)
    print("Design weights (1/pi) are in the manifest and are REQUIRED for unbiased")
    print("estimation; a city dropped after the draw invalidates every one of them.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
