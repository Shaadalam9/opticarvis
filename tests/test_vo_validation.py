r"""Guard the VO heading cross-check (ENGINEERING.md 3a, automated).

The VO yaw estimate acquires a steady drift on cameras whose calibration does
not match the flat-ground constants -- real turns still read correctly, so the
sign checks pass, while the future path curls sideways on straights and the
ribbon bends where the car is not going. Measured on Fvt6rD9tt1c_22: +225 deg
accumulated over 30 s, +14 deg in a window the scene pans the other way. The
cross-check rejects such tracks; a rejected track renders a straight in-lane
ribbon, the documented conservative fallback.

Synthetic profiles only -- no video, no models. Standalone:

    .venv/bin/python tests/test_vo_validation.py

or under pytest.
"""

import os
import sys

import numpy as np

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")

if SRC not in sys.path:
    sys.path.insert(0, SRC)

import ego_trajectory as ET  # noqa: E402

FPS = 30.0
WINDOW = int(ET.VALIDATION_WINDOW_S * FPS)


def psi_from_window_deltas(deltas_deg):
    """Per-frame heading (radians) accumulating each window's delta linearly."""
    psi = [0.0]

    for delta in deltas_deg:
        step = np.radians(delta) / WINDOW

        for _ in range(WINDOW):
            psi.append(psi[-1] + step)

    return np.asarray(psi)


def pans_from_window_sums(sums_px):
    pans = []

    for total in sums_px:
        pans.extend([total / WINDOW] * WINDOW)

    return np.asarray(pans)


def test_agreeing_turns_pass():
    """Genuine turns, matching signs, quiet straights: the track is kept."""
    pan_sums = [+2.0, +200.0, -5.0, -150.0]
    vo_deltas = [+0.5, +40.0, -0.5, -30.0]

    valid, reasons = ET.validate_heading_against_pan(
        psi_from_window_deltas(vo_deltas), pans_from_window_sums(pan_sums), FPS)

    assert valid, "clean agreement must pass, got: %r" % reasons


def test_sign_contradiction_rejects():
    """A strong scene pan one way with VO heading the other way is fatal."""
    pan_sums = [+2.0, -130.0]
    vo_deltas = [+0.5, +14.0]

    valid, reasons = ET.validate_heading_against_pan(
        psi_from_window_deltas(vo_deltas), pans_from_window_sums(pan_sums), FPS)

    assert not valid, "a sign contradiction must reject the track"
    assert any("pans" in reason for reason in reasons)


def test_sustained_drift_rejects():
    """Heading accumulating while the scene is quiet is drift, not turning."""
    pan_sums = [+5.0, -3.0, +8.0]
    vo_deltas = [+15.0, +18.0, +12.0]

    valid, reasons = ET.validate_heading_against_pan(
        psi_from_window_deltas(vo_deltas), pans_from_window_sums(pan_sums), FPS)

    assert not valid, "sustained drift on quiet scenes must reject the track"
    assert sum("drifts" in reason for reason in reasons) >= 2


def test_single_drift_window_is_tolerated():
    """One noisy window must not throw away an otherwise consistent track."""
    pan_sums = [+2.0, +200.0, +8.0]
    vo_deltas = [+0.5, +40.0, +12.0]

    valid, _reasons = ET.validate_heading_against_pan(
        psi_from_window_deltas(vo_deltas), pans_from_window_sums(pan_sums), FPS)

    assert valid, "a single drift window is noise, not bias"


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
