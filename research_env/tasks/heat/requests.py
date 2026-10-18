"""Build forecast requests from the prefix of a person's record.

``build_request`` receives the full prepared rows only so that the trusted
runner has one place to call; the first thing it does is discard every row
stamped after the cutoff, and nothing later in the function reads the
discarded rows. ``test_prefix_only`` perturbs post-cutoff data and checks the
request is byte-identical.
"""
from __future__ import annotations

import hashlib
import json

HORIZONS_MIN = (5, 10, 20)
CUTOFFS_MIN = tuple(range(5, 56))      # every minute 5..55 inclusive
PROTOCOL_MINUTES = 60
TIERS = ("wearable", "calibrated")
CALIBRATION_MINUTE = 1
CONTEXT_FIELDS = ("sex", "age_yr", "height_cm", "mass_kg", "bmi_kg_m2")
HISTORY_FIELDS = ("hr_bpm", "device_skin_c", "work_rate_w", "ambient_c", "ambient_rh_pct")


def opaque_id(secret: str, *parts) -> str:
    return hashlib.sha256(json.dumps([secret, *parts]).encode()).hexdigest()[:16]


def requested_horizons(cutoff: int) -> list[int]:
    """Horizons inside the known 60-minute protocol.

    Requested horizons depend only on the cutoff and the protocol length, never
    on whether a reference value exists, so request shape does not reveal
    evaluator eligibility masks (for example a masked rectal minute or an early
    end of recording).
    """
    return [h for h in HORIZONS_MIN if cutoff + h <= PROTOCOL_MINUTES]


def build_request(request_id: str, cutoff: int, input_rows, context: dict, tier: str,
                  reference_rows=None) -> dict:
    """Return the request a participant artifact sees for one cutoff.

    ``input_rows`` are this person's prepared input rows; ``reference_rows`` are
    used only in the calibrated tier and only for the fixed calibration minute.
    """
    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    prefix = [r for r in input_rows if r["minute"] <= cutoff]
    history = {f: [] for f in HISTORY_FIELDS}
    for row in sorted(prefix, key=lambda r: (r["field"], r["minute"])):
        if row["value"] is not None:          # masked cells are absent, not filled
            history[row["field"]].append([row["minute"], row["value"]])
    request = {
        "request_id": request_id,
        "tier": tier,
        "cutoff_min": cutoff,
        "horizons_min": requested_horizons(cutoff),
        "history": history,
        "context": {f: context[f]["value"] for f in CONTEXT_FIELDS},
        "known_protocol": {"exercise": "constant-load cycling", "duration_min": PROTOCOL_MINUTES,
                           "environment": "heat chamber; measured ambient values appear in history"},
    }
    if tier == "calibrated":
        cal = [r for r in (reference_rows or []) if r["minute"] == CALIBRATION_MINUTE]
        if len(cal) != 1 or not cal[0]["eligible"]:
            raise ValueError("calibration reference unavailable")
        if CALIBRATION_MINUTE > cutoff:
            raise ValueError("calibration minute after cutoff")
        request["calibration"] = {"rectal_c_at_minute": CALIBRATION_MINUTE,
                                  "rectal_c": cal[0]["rectal_c"]}
    return request


def build_targets(request_id: str, cutoff: int, reference_rows) -> list[dict]:
    """Evaluator-owned targets for one request, with frozen eligibility."""
    by_minute = {r["minute"]: r for r in reference_rows}
    out = []
    for h in requested_horizons(cutoff):
        ref = by_minute.get(cutoff + h)
        eligible = bool(ref and ref["eligible"])
        out.append({"request_id": request_id, "horizon_min": h, "target_minute": cutoff + h,
                    "rectal_c": ref["rectal_c"] if eligible else None, "eligible": eligible,
                    "ineligible_reason": None if eligible else
                    ("no reference recorded" if ref is None else ",".join(ref["flags"]))})
    return out
