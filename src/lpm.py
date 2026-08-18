"""Local Pivotal Method (LPM) spatially balanced sampling -- pure numpy.

Reference
---------
Grafstrom, A., Lundstrom, N.L.P. & Schelin, L. (2012),
"Spatially balanced sampling through the pivotal method", Biometrics 68(2), 514-520.

What the method does
--------------------
Every unit i = 1..N carries an inclusion probability ``pi[i]`` in [0, 1], with
``sum(pi) == n`` (the sample size), and a coordinate ``coords[i]`` in R^d.

The pivotal method repeatedly picks two units and lets them *fight over* their
combined probability mass: one of them moves toward 1 (selected), the other
toward 0 (rejected), in a way that leaves each unit's inclusion probability
unchanged in expectation.  The randomisation runs until every pi is 0 or 1,
at which point the units with pi == 1 are the sample.

The *local* pivotal method makes the two competing units spatial NEIGHBOURS.
Because one neighbour's gain is the other's loss, probability mass is pushed
*out of* the neighbourhood of whichever unit wins, so the realised sample
spreads out over the coordinate space instead of clumping.

    LPM1: i and j must be mutual nearest neighbours (better spread, more work).
    LPM2: j is simply i's nearest neighbour (cheaper, still well spread).

Design notes
------------
* Only numpy is used.  Nearest neighbours are found by brute force over the
  *active* units (those with 0 < pi < 1), which shrinks by at least one unit
  per iteration, so the whole run is O(N^2 * d) worst case with a small
  constant.  For the frame sizes this is written for (N up to ~1e4) that is
  fast enough and keeps the code auditable; swap in a k-d tree only if you
  measure a need.
* Active units are held in a compacted array with swap-removal, so no
  per-iteration boolean masking or fancy indexing is needed.
* All randomness goes through a ``numpy.random.Generator`` so runs are
  reproducible via ``rng=np.random.default_rng(seed)``.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "lpm",
    "lpm1",
    "lpm2",
    "pi_from_size",
    "spatial_balance",
    "systematic_pps",
]

# Probabilities within EPS of 0 or 1 are snapped to 0 or 1.  Needed because the
# pair update produces values like ``pi_i + pi_j - 1`` that can land at 1e-17
# instead of exactly 0; without snapping the algorithm would never terminate.
# The induced bias is bounded by EPS and is far below any Monte Carlo error.
EPS = 1e-12


# --------------------------------------------------------------------------
# inclusion probabilities
# --------------------------------------------------------------------------

def pi_from_size(sizes, n, *, eps=1e-12):
    """Inclusion probabilities proportional to size, capped at 1.

    Starts from ``pi = n * sizes / sum(sizes)`` and then applies the standard
    iterative capping: any unit whose pi would exceed 1 becomes a *certainty*
    unit (pi = 1) and its excess is redistributed proportionally among the
    remaining units, repeated until no further unit exceeds 1.

    Parameters
    ----------
    sizes : array_like, shape (N,)
        Non-negative size measure.  Units with size 0 get pi = 0.
    n : int
        Target sample size.  Must satisfy ``0 <= n <= (# units with size > 0)``.

    Returns
    -------
    pi : ndarray, shape (N,), float
        Inclusion probabilities, ``sum(pi) == n`` (up to float error).
    certainty : ndarray, int
        Indices of the units that were capped to 1.

    Raises
    ------
    ValueError
        If sizes are negative, or n exceeds the number of positive-size units.
    """
    sizes = np.asarray(sizes, dtype=float).ravel()
    if np.any(sizes < 0) or not np.all(np.isfinite(sizes)):
        raise ValueError("sizes must be finite and non-negative")
    n = int(n)
    N = sizes.size
    if n < 0:
        raise ValueError("n must be non-negative")

    pi = np.zeros(N, dtype=float)
    if n == 0:
        return pi, np.empty(0, dtype=int)

    positive = sizes > 0
    n_pos = int(positive.sum())
    if n > n_pos:
        raise ValueError(
            f"n={n} exceeds the number of units with positive size ({n_pos})"
        )

    certainty = np.zeros(N, dtype=bool)
    while True:
        free = positive & ~certainty
        n_free = n - int(certainty.sum())
        if n_free <= 0:
            # All of the sample is taken by certainty units.
            pi[free] = 0.0
            break
        total = sizes[free].sum()
        pi_free = n_free * sizes[free] / total
        over = pi_free > 1.0 + eps
        if not np.any(over):
            pi[free] = np.minimum(pi_free, 1.0)
            break
        idx_free = np.flatnonzero(free)
        certainty[idx_free[over]] = True

    pi[certainty] = 1.0
    pi[~positive] = 0.0
    pi = np.clip(pi, 0.0, 1.0)
    return pi, np.flatnonzero(certainty)


# --------------------------------------------------------------------------
# the pivotal pair update
# --------------------------------------------------------------------------

def _pivot(pi_i, pi_j, rng):
    """One pivotal update of a pair; returns the new ``(pi_i, pi_j)``.

    Deville & Tille's pivotal step, as used by Grafstrom et al. (2012)::

        pi_i + pi_j < 1:   -> (pi_i + pi_j, 0)  w.p. pi_i / (pi_i + pi_j)
                           -> (0, pi_i + pi_j)  w.p. pi_j / (pi_i + pi_j)
        pi_i + pi_j >= 1:  -> (1, pi_i + pi_j - 1)  w.p. (1 - pi_j) / (2 - pi_i - pi_j)
                           -> (pi_i + pi_j - 1, 1)  w.p. (1 - pi_i) / (2 - pi_i - pi_j)

    Both branches are martingale steps -- E[new pi_i] == pi_i and
    E[new pi_j] == pi_j -- and the pair total ``pi_i + pi_j`` is conserved
    exactly.  At least one of the two returned values is exactly 0 or 1, which
    is what makes the outer loop terminate in at most N iterations.

    NOTE on the first branch: the winner takes the mass with probability equal
    to *its own* share, not its partner's.  Pairing ``pi_j / (pi_i + pi_j)``
    with the outcome ``(pi_i + pi_j, 0)`` is a natural transcription slip and
    it silently destroys the design: the sample size and the total expected
    mass both stay correct, so only a per-unit Monte Carlo check catches it.
    See lpm_validate.py test 2.
    """
    total = pi_i + pi_j
    if total < 1.0:
        # One of the two is knocked out; the survivor absorbs the whole mass.
        if rng.random() < pi_i / total:
            return total, 0.0
        return 0.0, total
    # One of the two is locked in at 1; the other keeps the leftover.
    if rng.random() < (1.0 - pi_j) / (2.0 - total):
        return 1.0, total - 1.0
    return total - 1.0, 1.0


# --------------------------------------------------------------------------
# the sampler
# --------------------------------------------------------------------------

def lpm(pi, coords, method="lpm2", rng=None, return_mask=False):
    """Draw a spatially balanced sample by the local pivotal method.

    Parameters
    ----------
    pi : array_like, shape (N,)
        Inclusion probabilities in [0, 1].  Their sum must be (very close to)
        an integer; that integer is the realised sample size.  Values already
        equal to 0 or 1 are respected: 0s are never selected, 1s always are.
    coords : array_like, shape (N, d) or (N,)
        Coordinates used for the nearest-neighbour search.  Scale them
        yourself if the dimensions are not comparable -- the method uses plain
        Euclidean distance.
    method : {"lpm2", "lpm1"}
        ``"lpm2"`` (default): j is i's nearest active neighbour.
        ``"lpm1"``: i and j must be *mutual* nearest neighbours, found by
        walking the nearest-neighbour chain from a random start.
    rng : numpy.random.Generator, optional
        Source of randomness; defaults to ``np.random.default_rng()``.
    return_mask : bool
        If True return a boolean mask of length N instead of an index array.

    Returns
    -------
    ndarray
        Sorted indices of the selected units (or a boolean mask).

    Raises
    ------
    ValueError
        If pi is outside [0, 1], its sum is not near-integer, or the shapes of
        pi and coords disagree.
    """
    if rng is None:
        rng = np.random.default_rng()
    method = method.lower()
    if method not in ("lpm1", "lpm2"):
        raise ValueError("method must be 'lpm1' or 'lpm2'")

    p = np.array(pi, dtype=float).ravel()          # working copy, mutated
    N = p.size
    coords = np.asarray(coords, dtype=float)
    if coords.ndim == 1:
        coords = coords.reshape(-1, 1)
    if coords.shape[0] != N:
        raise ValueError(
            f"coords has {coords.shape[0]} rows but pi has {N} entries"
        )
    if np.any(~np.isfinite(p)) or np.any(p < -EPS) or np.any(p > 1.0 + EPS):
        raise ValueError("pi must be finite and lie in [0, 1]")
    p = np.clip(p, 0.0, 1.0)

    target = p.sum()
    n = int(round(target))
    if abs(target - n) > 1e-6:
        raise ValueError(f"sum(pi) = {target!r} is not an integer sample size")

    # --- active set: units with 0 < pi < 1, held compacted with swap-removal.
    act_idx = np.flatnonzero((p > EPS) & (p < 1.0 - EPS))   # unit ids
    m = act_idx.size
    act = np.empty(N, dtype=np.intp)
    act[:m] = act_idx
    C = np.empty((max(N, 1), coords.shape[1]), dtype=float)  # coords, compacted
    C[:m] = coords[act_idx]

    def drop(s):
        """Remove slot s from the active set (swap in the last active unit)."""
        nonlocal m
        m -= 1
        if s != m:
            act[s] = act[m]
            C[s] = C[m]

    def nearest(s, prefer=-1):
        """Slot of the nearest other active unit to slot ``s``.

        Ties are broken uniformly at random, except that if ``prefer`` (a slot)
        is among the tied minima it wins.  That preference is what guarantees
        the LPM1 nearest-neighbour chain terminates when coordinates are
        duplicated -- otherwise a tie could cycle forever.
        """
        d = C[:m] - C[s]
        d2 = np.einsum("ij,ij->i", d, d)
        d2[s] = np.inf
        best = d2.min()
        if prefer >= 0 and d2[prefer] <= best:
            return prefer
        tied = np.flatnonzero(d2 == best)
        if tied.size == 1:
            return int(tied[0])
        return int(tied[rng.integers(tied.size)])

    while m > 0:
        if m == 1:
            # A lone survivor with fractional pi.  Since sum(pi) is an integer
            # and every retired unit is exactly 0 or 1, its value must be
            # within float drift of 0 or 1; snap it.  The coin-flip fallback
            # keeps the draw unbiased if it ever is genuinely fractional.
            u = int(act[0])
            if p[u] < 1e-9:
                p[u] = 0.0
            elif p[u] > 1.0 - 1e-9:
                p[u] = 1.0
            else:
                p[u] = 1.0 if rng.random() < p[u] else 0.0
            drop(0)
            break

        si = int(rng.integers(m))
        if method == "lpm2":
            sj = nearest(si)
        else:
            # Walk the nearest-neighbour chain until a mutual pair is found.
            sj = nearest(si)
            while True:
                sk = nearest(sj, prefer=si)
                if sk == si:
                    break
                si, sj = sj, sk

        ui, uj = int(act[si]), int(act[sj])
        p[ui], p[uj] = _pivot(p[ui], p[uj], rng)

        # Snap and retire whichever unit(s) reached a boundary.  Drop the
        # higher slot first so the swap-removal cannot invalidate the other.
        finished = []
        for s, u in ((si, ui), (sj, uj)):
            if p[u] <= EPS:
                p[u] = 0.0
                finished.append(s)
            elif p[u] >= 1.0 - EPS:
                p[u] = 1.0
                finished.append(s)
        for s in sorted(finished, reverse=True):
            drop(s)

    mask = p >= 0.5
    if return_mask:
        return mask
    return np.flatnonzero(mask)


def lpm2(pi, coords, rng=None, return_mask=False):
    """LPM2: pair each unit with its nearest active neighbour."""
    return lpm(pi, coords, method="lpm2", rng=rng, return_mask=return_mask)


def lpm1(pi, coords, rng=None, return_mask=False):
    """LPM1: pair only mutual nearest neighbours (tighter spread than LPM2)."""
    return lpm(pi, coords, method="lpm1", rng=rng, return_mask=return_mask)


# --------------------------------------------------------------------------
# baseline sampler for comparison
# --------------------------------------------------------------------------

def systematic_pps(pi, rng=None, return_mask=False):
    """Randomised systematic pps sampling -- the spatially *unaware* baseline.

    Permutes the units at random, walks the cumulative sum of pi and takes the
    units straddling ``u, u+1, ..., u+n-1`` for ``u ~ Uniform(0, 1)``.  Gives
    exactly n units and respects the inclusion probabilities, but ignores the
    coordinates entirely -- which is exactly what LPM is meant to improve on.
    """
    if rng is None:
        rng = np.random.default_rng()
    p = np.asarray(pi, dtype=float).ravel()
    N = p.size
    n = int(round(p.sum()))
    order = rng.permutation(N)
    cum = np.cumsum(p[order])
    marks = rng.random() + np.arange(n, dtype=float)
    picks = order[np.searchsorted(cum, marks, side="right")]
    mask = np.zeros(N, dtype=bool)
    mask[picks] = True
    if return_mask:
        return mask
    return np.flatnonzero(mask)


# --------------------------------------------------------------------------
# spatial balance
# --------------------------------------------------------------------------

def spatial_balance(pi, coords, selected, *, chunk=4096):
    """Voronoi spatial balance B (Stevens & Olsen); lower is better.

    Every unit of the frame is assigned to its nearest SELECTED unit.  For each
    selected unit i, ``v_i`` is the sum of the inclusion probabilities of the
    units assigned to it.  A perfectly balanced sample has every ``v_i == 1``
    (each selected unit "represents" exactly one unit's worth of probability),
    so::

        B = mean_i (v_i - 1)^2

    Parameters
    ----------
    pi : array_like, shape (N,)
    coords : array_like, shape (N, d) or (N,)
    selected : array_like
        Indices of the selected units, or a boolean mask of length N.
    chunk : int
        Rows of the frame processed per block, to bound memory at O(chunk * n).

    Returns
    -------
    float
        B, or ``nan`` if nothing is selected.
    """
    p = np.asarray(pi, dtype=float).ravel()
    coords = np.asarray(coords, dtype=float)
    if coords.ndim == 1:
        coords = coords.reshape(-1, 1)
    sel = np.asarray(selected)
    if sel.dtype == bool:
        sel = np.flatnonzero(sel)
    sel = sel.astype(np.intp).ravel()
    if sel.size == 0:
        return float("nan")

    S = coords[sel]
    v = np.zeros(sel.size, dtype=float)
    for a in range(0, coords.shape[0], chunk):
        block = coords[a:a + chunk]
        # squared Euclidean distance, block x selected
        d2 = (
            np.einsum("ij,ij->i", block, block)[:, None]
            - 2.0 * block @ S.T
            + np.einsum("ij,ij->i", S, S)[None, :]
        )
        owner = np.argmin(d2, axis=1)
        v += np.bincount(owner, weights=p[a:a + chunk], minlength=sel.size)
    return float(np.mean((v - 1.0) ** 2))
