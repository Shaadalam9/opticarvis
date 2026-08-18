r"""Guard the city sampling design.

The point of drawing cities with the Local Pivotal Method rather than picking
them is that the result is a PROBABILITY SAMPLE: every city's inclusion
probability is exactly what was assigned, so 1/pi is a valid design weight and
country-level estimates have defensible standard errors. These tests pin the
properties that claim rests on, plus the two geometry traps that silently break
a global design.

CPU only, no network. Standalone:

    .venv/bin/python tests/test_city_sampler.py

or under pytest.
"""

import os
import sys

import numpy as np

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")

if SRC not in sys.path:
    sys.path.insert(0, SRC)

import city_sampler as CS      # noqa: E402
import lpm_sampling as LPM     # noqa: E402


def clustered_frame(n_points=600, seed=3):
    """A frame with real clustering, which is what spatial balance is about."""
    rng = np.random.default_rng(seed)
    centres = rng.uniform(-60.0, 60.0, size=(6, 2))
    lat = []
    lon = []

    for centre in centres:
        lat.extend(rng.normal(centre[0], 3.0, n_points // 8))
        lon.extend(rng.normal(centre[1], 3.0, n_points // 8))

    lat.extend(rng.uniform(-60.0, 60.0, n_points - len(lat)))
    lon.extend(rng.uniform(-180.0, 180.0, n_points - len(lon)))
    lat = np.clip(np.array(lat), -89.0, 89.0)
    lon = ((np.array(lon) + 180.0) % 360.0) - 180.0

    return lat, lon


def test_inclusion_probabilities_are_respected():
    """The defining property: E[selected] == pi, per city.

    If this drifts, every design weight in the manifest is wrong and the
    "probability sample" claim in the paper is false.
    """
    rng = np.random.default_rng(11)
    lat, lon = clustered_frame(200, seed=5)
    xyz = CS.unit_sphere_xyz(lat, lon)
    sizes = rng.gamma(2.0, 50000.0, size=len(lat))
    pi, _certainty = LPM.pi_from_size(sizes, 30)

    replicates = 4000
    counts = np.zeros(len(pi))

    for _ in range(replicates):
        counts[LPM.lpm(pi, xyz, rng=rng)] += 1.0

    frequency = counts / replicates
    se = np.sqrt(np.maximum(pi * (1.0 - pi), 1e-12) / replicates)
    worst = float(np.max(np.abs(frequency - pi) / se))

    assert worst < 5.0, (
        "inclusion probabilities are off by up to %.2f Monte Carlo SE -- the "
        "sampler is biased and the design weights would be fiction" % worst)


def test_sample_size_and_capping_are_exact():
    """sum(pi) == n must hold exactly, including with certainty units."""
    sizes = np.array([5e7, 4e7, 1e5, 2e5, 3e5, 1e4, 5e4, 8e4], dtype=float)
    pi, certainty = LPM.pi_from_size(sizes, 4)

    assert abs(pi.sum() - 4.0) < 1e-9, "pi must sum to n, got %.9f" % pi.sum()
    assert pi.max() <= 1.0 + 1e-12, "an inclusion probability exceeded 1"
    assert len(certainty) >= 1, "the two megacities should become certainty units"

    lat, lon = clustered_frame(80, seed=9)
    xyz = CS.unit_sphere_xyz(lat[:8], lon[:8])

    for seed in range(20):
        selected = LPM.lpm(pi, xyz, rng=np.random.default_rng(seed))
        assert len(selected) == 4, "sample size drifted to %d" % len(selected)

        for unit in certainty:
            assert unit in selected, "a certainty unit was not selected"


def test_antimeridian_neighbours_stay_neighbours():
    """Cities either side of the date line are neighbours, not opposites.

    In raw degrees Suva (+178) and Apia (-172) are 350 apart, so a degree-space
    sampler treats the Pacific rim as empty on both sides and will select both.
    Unit-sphere chord distance has no seam.
    """
    suva = CS.unit_sphere_xyz([-18.14], [178.44])
    apia = CS.unit_sphere_xyz([-13.83], [-171.77])
    london = CS.unit_sphere_xyz([51.51], [-0.13])

    across_dateline = float(CS.great_circle_km(suva[0], apia[0]))
    to_london = float(CS.great_circle_km(suva[0], london[0]))

    assert 1000.0 < across_dateline < 1600.0, (
        "Suva to Apia should be ~1200 km, got %.0f km" % across_dateline)
    assert across_dateline < to_london, "the date line was treated as a real gap"

    degree_gap = abs(178.44 - (-171.77))
    assert degree_gap > 350.0, "sanity: raw degrees really are misleading here"


def test_great_circle_matches_known_distance():
    london = CS.unit_sphere_xyz([51.5074], [-0.1278])
    paris = CS.unit_sphere_xyz([48.8566], [2.3522])
    km = float(CS.great_circle_km(london[0], paris[0]))

    assert abs(km - 344.0) < 6.0, "London-Paris came out %.1f km, expected ~344" % km


def test_high_latitude_is_not_stretched():
    """One degree of longitude is ~48 km at Reykjavik, not 111 km.

    Treating degrees as isotropic over-weights east-west separation with
    latitude, which biases spread in exactly the places city density changes.
    """
    a = CS.unit_sphere_xyz([64.1466], [-21.9426])
    b = CS.unit_sphere_xyz([64.1466], [-20.9426])
    km = float(CS.great_circle_km(a[0], b[0]))

    assert 40.0 < km < 56.0, "one degree of longitude at 64N came out %.1f km" % km


def test_lpm_spreads_better_than_a_spatially_blind_design():
    """LPM must beat randomised systematic pps on Voronoi balance.

    Averaged over replicates: a single draw of either design proves nothing,
    because the balance index varies substantially between draws.
    """
    lat, lon = clustered_frame(800, seed=17)
    xyz = CS.unit_sphere_xyz(lat, lon)
    rng = np.random.default_rng(23)
    sizes = rng.gamma(2.0, 30000.0, size=len(lat))
    pi, _ = LPM.pi_from_size(sizes, 60)

    balanced = []
    blind = []

    for replicate in range(12):
        balanced.append(LPM.spatial_balance(
            pi, xyz, LPM.lpm(pi, xyz, rng=np.random.default_rng(100 + replicate))))
        blind.append(LPM.spatial_balance(
            pi, xyz, LPM.systematic_pps(pi, rng=np.random.default_rng(500 + replicate))))

    assert np.mean(balanced) < 0.75 * np.mean(blind), (
        "LPM balance %.4f vs blind %.4f -- the spatial method is not spreading"
        % (np.mean(balanced), np.mean(blind)))


def test_eligibility_is_applied_before_the_draw():
    """Filtering after sampling would make every design weight wrong.

    The frame the probabilities are computed over must already exclude
    ineligible cities, so pi sums to n over the ELIGIBLE frame.
    """
    rows = [{"locality": "c%d" % i, "country": "X", "lat": "%.3f" % (10.0 + i * 0.1),
             "lon": "%.3f" % (20.0 + i * 0.1), "population_locality": "%d" % (1000 * (i + 1)),
             "footage_hours": "2.0" if i % 2 == 0 else "0.25"}
            for i in range(40)]
    columns = {"lat": "lat", "lon": "lon", "population": "population_locality",
               "footage": "footage_hours"}

    kept, population, xyz, report = CS.build_frame(rows, columns, min_footage_hours=1.0)

    assert report["dropped_footage"] == 20, "expected half the frame filtered out"
    assert len(kept) == 20 and len(population) == 20 and len(xyz) == 20

    pi, selected, _ = CS.draw_sample(population, xyz, 5, 1.0, seed=1)

    assert abs(pi.sum() - 5.0) < 1e-9, "pi must sum to n over the eligible frame"
    assert len(selected) == 5

    for row in kept:
        assert float(row["footage_hours"]) >= 1.0, "an ineligible city entered the frame"


def test_zero_population_is_imputed_not_dropped():
    """Zero is this project's missing sentinel; pi = 0 would make a city
    structurally unsamplable and silently shrink the frame."""
    rows = [{"locality": "c%d" % i, "lat": "%.2f" % (5.0 + i), "lon": "%.2f" % (5.0 + i),
             "population_locality": "0" if i < 3 else "%d" % (10000 * i)}
            for i in range(12)]
    columns = {"lat": "lat", "lon": "lon", "population": "population_locality", "footage": ""}

    kept, population, _xyz, report = CS.build_frame(rows, columns, min_footage_hours=0.0)

    assert len(kept) == 12, "rows with zero population must be kept"
    assert report["population_imputed"] == 3
    assert float(population.min()) > 0.0, "an imputed population is still zero"


if __name__ == "__main__":
    failures = 0

    for name, test in sorted(globals().items()):
        if name.startswith("test_") and callable(test):
            try:
                test()
                print("PASS  %s" % name)
            except AssertionError as error:
                failures += 1
                print("FAIL  %s\n      %s" % (name, error))

    raise SystemExit(1 if failures else 0)
