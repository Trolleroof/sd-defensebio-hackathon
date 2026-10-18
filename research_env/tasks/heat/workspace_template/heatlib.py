"""Participant helpers: rebuild runner-identical requests from training people.

``training_requests`` produces exactly the request objects the trusted runner
sends at inference (checked by the builder's tests), so models can be fit on
the same representation they will see. Only numpy and the standard library are
needed. You may copy, change or ignore this file.
"""
from __future__ import annotations

import json
from pathlib import Path

HORIZONS_MIN = (5, 10, 20)
CUTOFFS_MIN = tuple(range(5, 56))
PROTOCOL_MINUTES = 60
CALIBRATION_MINUTE = 1
HISTORY_FIELDS = ("hr_bpm", "device_skin_c", "work_rate_w", "ambient_c", "ambient_rh_pct")
CONTEXT_FIELDS = ("sex", "age_yr", "height_cm", "mass_kg", "bmi_kg_m2")


def load_training(train_manifest):
    return json.loads(Path(train_manifest).read_text())


def requested_horizons(cutoff):
    return [h for h in HORIZONS_MIN if cutoff + h <= PROTOCOL_MINUTES]


def request_at(person, cutoff, tier):
    """Request for one training person at one cutoff (same shape as inference)."""
    history = {f: [] for f in HISTORY_FIELDS}
    for row in sorted((r for r in person["inputs"] if r["minute"] <= cutoff),
                      key=lambda r: (r["field"], r["minute"])):
        if row["value"] is not None:
            history[row["field"]].append([row["minute"], row["value"]])
    req = {"request_id": f"{person['person']}-{tier}-{cutoff}", "tier": tier, "cutoff_min": cutoff,
           "horizons_min": requested_horizons(cutoff), "history": history,
           "context": dict(person["context"]),
           "known_protocol": {"exercise": "constant-load cycling", "duration_min": PROTOCOL_MINUTES,
                              "environment": "heat chamber; measured ambient values appear in history"}}
    if tier == "calibrated":
        cal = [r for r in person["reference"] if r["minute"] == CALIBRATION_MINUTE and r["eligible"]]
        req["calibration"] = {"rectal_c_at_minute": CALIBRATION_MINUTE, "rectal_c": cal[0]["rectal_c"]}
    return req


def training_requests(train, tier=None):
    """Yield (person_id, request, {horizon: target or None}) for every cutoff."""
    tier = tier or train["tier"]
    for person in train["people"]:
        ref = {r["minute"]: r["rectal_c"] for r in person["reference"] if r["eligible"]}
        for c in CUTOFFS_MIN:
            req = request_at(person, c, tier)
            yield person["person"], req, {h: ref.get(c + h) for h in req["horizons_min"]}


def last(series, default=None):
    return series[-1][1] if series else default


def last_time(series, default=None):
    return series[-1][0] if series else default


def slope(series, window):
    """Least-squares slope (units/min) over points within ``window`` minutes of the last one."""
    if len(series) < 2:
        return 0.0
    t_end = series[-1][0]
    pts = [(t, v) for t, v in series if t >= t_end - window]
    if len(pts) < 2:
        return 0.0
    mt = sum(t for t, _ in pts) / len(pts)
    mv = sum(v for _, v in pts) / len(pts)
    den = sum((t - mt) ** 2 for t, _ in pts)
    return sum((t - mt) * (v - mv) for t, v in pts) / den if den else 0.0


def features(req, h):
    """A plain causal feature vector; every value is computable at the cutoff."""
    hist, ctx, c = req["history"], req["context"], req["cutoff_min"]
    skin = hist["device_skin_c"]
    hr = hist["hr_bpm"]
    work = hist["work_rate_w"]
    mass = ctx["mass_kg"]
    hr_last = last(hr, 150.0)
    f = [
        1.0, c, h, c + h, (c + h) ** 0.5,
        last(skin, 35.5), slope(skin, 5), slope(skin, 15),
        hr_last, 0.0 if hr else 1.0, (c - last_time(hr, 0)) if hr else c,
        sum(v for _, v in hr) / len(hr) if hr else 150.0,
        last(work, 150.0), last(work, 150.0) / mass,
        last(hist["ambient_c"], 36.0), last(hist["ambient_rh_pct"], 20.0),
        1.0 if ctx["sex"] == "M" else 0.0, ctx["age_yr"], ctx["height_cm"], mass, ctx["bmi_kg_m2"],
    ]
    if req["tier"] == "calibrated":
        f.append(req["calibration"]["rectal_c"])
    return f
