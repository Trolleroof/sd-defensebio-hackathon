"""Frozen input-quality rules (heat-quality-v1).

Rules mask a cell; they never replace it with a guessed value. Ranges are wide
physiological plausibility bounds chosen before any model comparison, not
clinical limits. The rectal spike rule targets isolated single-minute dropouts
inside a slowly varying series.
"""
from __future__ import annotations

VERSION = "heat-quality-v1"
RANGES = {
    "hr_bpm": (30.0, 230.0),
    "rectal_c": (35.5, 42.0),
    "device_skin_c": (25.0, 42.0),
    "work_rate_w": (0.0, 600.0),
    "ambient_c": (20.0, 50.0),
    "ambient_rh_pct": (0.0, 100.0),
}
SPIKE_FIELDS = {"rectal_c": 1.0}
RULES = {
    "non_numeric": "cell is not stored as a number -> masked",
    "out_of_range": {"rule": "value outside plausibility range -> masked", "ranges": RANGES},
    "isolated_spike": {"rule": "differs from both adjacent recorded minutes by more than the "
                               "threshold in the same direction -> masked",
                       "threshold_c": SPIKE_FIELDS},
    "missing": "absent cells produce no record; nothing is interpolated or carried forward",
}


def _num(point):
    return point["cell"].number if point else None


def check(field, cell, minute, points):
    """Return (value or None, flags) for one source cell."""
    if cell.kind != "n" or cell.number is None:
        return None, ["non_numeric"]
    value = cell.number
    flags = []
    lo, hi = RANGES[field]
    if not lo <= value <= hi:
        flags.append("out_of_range")
    if field in SPIKE_FIELDS:
        prev, nxt = _num(points.get(minute - 1)), _num(points.get(minute + 1))
        thr = SPIKE_FIELDS[field]
        if prev is not None and nxt is not None:
            dp, dn = value - prev, value - nxt
            if abs(dp) > thr and abs(dn) > thr and (dp > 0) == (dn > 0):
                flags.append("isolated_spike")
    return (None if flags else value), flags
