# Choosing the cities

The study needs cities that are spread over the inhabited world, weighted
toward where people actually live, and drawn in a way that survives a reviewer
asking "why these?". This document explains the design, the references behind
it, and how to run it.

## Why not one city per country

The one-per-country rule forces India to a single arbitrary city — Patna rather
than Delhi — while giving Luxembourg the same allocation. It also produces no
inclusion probabilities, so there are no design weights and no defensible
standard errors for any country-level analysis. Large countries needing several
cities and small ones needing at most one is exactly what a
probability-proportional-to-size design does on its own.

## The method: Local Pivotal Method (LPM)

Each city gets an inclusion probability `pi_i` proportional to its population,
scaled so the probabilities sum to the sample size. The sampler then repeatedly
picks a city and its **nearest neighbour** and lets them compete for their
combined probability: one moves toward selection, the other toward rejection,
in a way that leaves each city's inclusion probability unchanged in
expectation. Because the competitors are always spatial neighbours, selecting
one pushes probability *out of its neighbourhood*, and the sample spreads
itself over the map. When every probability has resolved to 0 or 1, the 1s are
the sample.

The result is a strict probability design: `pi_i` comes out exactly as
assigned, so `1/pi_i` is a valid Horvitz–Thompson design weight.

**LPM1** requires the two competitors to be *mutual* nearest neighbours (better
spread, more work); **LPM2** just takes the nearest neighbour of the chosen
city. `--method` selects between them; LPM2 is the default.

### The alternative: GRTS

GRTS (Generalized Random Tessellation Stratified) reaches a similar goal by
recursively tessellating the region into randomised quadrants, flattening that
hierarchy into a one-dimensional address that preserves proximity, and taking a
systematic sample along it. It is the established method in environmental
survey work and is what the US EPA's `spsurvey` implements. For a finite list
of points on a sphere with unequal probabilities, LPM needs less machinery — it
requires only a distance metric — which is why it is what this repo implements.

## References

- Grafström, A., Lundström, N.L.P. & Schelin, L. (2012). Spatially balanced
  sampling through the pivotal method. *Biometrics* 68(2), 514–520.
  doi:10.1111/j.1541-0420.2011.01699.x — **the primary citation for LPM1/LPM2.**
- Stevens, D.L. Jr. & Olsen, A.R. (2004). Spatially balanced sampling of natural
  resources. *Journal of the American Statistical Association* 99(465),
  262–278. doi:10.1198/016214504000000250 — GRTS, and the source of the Voronoi
  spatial-balance measure used in the diagnostics.
- Grafström, A. & Lundström, N.L.P. (2013). Why well spread probability samples
  are balanced. *Open Journal of Statistics* 3(1), 36–41.
  doi:10.4236/ojs.2013.31005 — why spreading in an auxiliary space also
  balances on it. (Modest venue; cite the Biometrics paper as the anchor.)
- Deville, J.-C. & Tillé, Y. (2004). Efficient balanced sampling: the cube
  method. *Biometrika* 91(4), 893–912. doi:10.1093/biomet/91.4.893 — the general
  balanced-sampling framework LPM is a spatial cousin of.
- Dumelle, M., Kincaid, T., Olsen, A.R. & Weber, M. (2023). spsurvey: spatial
  sampling design and analysis in R. *Journal of Statistical Software* 105(3).
  doi:10.18637/jss.v105.i03 — software citation for GRTS.

### Reference implementations

- **R, `BalancedSampling`** (AGPL-3) — <https://github.com/envisim/BalancedSampling>,
  by the LPM authors. `lpm1()`, `lpm2()`, `sb()` for spatial balance,
  `getPips()` for probabilities proportional to size. This is the implementation
  to cross-check against if a reviewer asks.
- **R, `spsurvey`** (GPL-3, US EPA) — <https://github.com/USEPA/spsurvey>.
  `grts()`, `sp_balance()`.
- **Python** — there was no maintained package for either method when this was
  written: PyPI has no `pygrts`, the only credible Python GRTS code is a
  single-author GitHub project, and searches for a Python LPM return unrelated
  projects. So the implementation was written here and then **extracted into its
  own package**, [`lpm-sampling`](https://github.com/M-Colley/lpm-sampling)
  (MIT, numpy only), which this repo now depends on. It ships its own tests and
  a harness that compares it against the R reference on a shared frame.

## Three things the implementation must get right

**Eligibility is applied to the frame, before the draw.** Restricting to cities
with enough footage *after* sampling would destroy the probabilities the
weights depend on. Filter first; the sample then represents the filtered frame,
and the write-up says so. If a sampled city later turns out to have no usable
footage, do **not** substitute a convenient neighbour — that converts the
probability sample into a convenience sample and every weight becomes fiction.

**Coordinates are unit-sphere x/y/z, never raw degrees.** Longitude is
circular: in degree space Suva (+178) and Apia (−172) are 350° apart when they
are 1,200 km neighbours, so a degree-space sampler believes the Pacific rim is
empty on both sides and cheerfully selects both. Degrees are also anisotropic —
one degree of longitude is 111 km at the equator and 48 km at Reykjavik.
Chord distance on the unit sphere is monotone in great-circle distance, with no
seam and no pole degeneracy. Both traps are pinned by tests.

**Probabilities are capped by iteration, not by clipping.** A megacity whose
raw probability exceeds 1 becomes a certainty unit and its excess is
redistributed, which can push another city over 1 — so it repeats until stable.
Clipping once silently breaks `sum(pi) == n`.

## The population-damping trade-off

`--alpha` raises population to a power before computing probabilities.
`alpha = 1` is strict PPS; lower values pull toward equal probability. Measured
on an 8,856-city frame, n = 150:

| alpha | person-km to nearest city | effective n | weight range |
|---|---|---|---|
| 1.00 | 290 | 17.9 | 1 – 1951 |
| 0.75 | 301 | 54.0 | 2 – 730 |
| 0.50 | 307 | 103.3 | 10 – 252 |
| 0.25 | 316 | 134.4 | 19 – 121 |

Strict PPS gives the best participant-matching distance but an effective sample
size of only 18 out of 150 — most of the statistical power is spent on a few
megacities. `alpha` between 0.5 and 0.75 buys back three to six times the
effective sample size for a 4–6% worse matching distance, which is usually the
better trade for a study that also wants to say something about countries.

## Running it

```bash
.venv/bin/python src/city_sampler.py \
    --frame path/to/cities.csv --n 150 --seed 20260818 --alpha 0.75 \
    --population-column population_locality \
    --footage-column footage_hours --min-footage-hours 1.0 \
    --out workflow_outputs/city_sample
```

It prints the frame report, a replicated comparison against a spatially unaware
design with the same probabilities, the population-coverage curve, and the
weight diagnostics; it writes `sample_manifest.csv` with `pi` and
`design_weight` per city. Keep that manifest: it is the statistical object,
and `mapping.csv` cannot carry it.

Measured on a clustered 8,856-city frame (n = 150, 10 draws each): Voronoi
balance 0.150 ± 0.004 for LPM against 0.444 ± 0.022 for randomised systematic
pps, and 291 km against 378 km of population-weighted distance to the nearest
selected city. The whole draw takes about 3 seconds.

## What this does not fix

Footage availability is not random: dashcam uploads correlate with income,
YouTube penetration, language and road type — the very axes the study cares
about. A perfect design over "cities with footage" still generalises only to
cities with footage. State that as a limitation rather than hoping a reviewer
misses it.

The explanation gate is a second, unmodelled filter: a city can pass
eligibility, be sampled, and still produce no video because the gate declines
every window. Track how many sampled cities yield a render, and report it.
