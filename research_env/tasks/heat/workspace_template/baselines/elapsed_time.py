"""Mandatory elapsed-time-only baseline.

Predicts the training-population mean rectal temperature at the target minute,
ignoring every measurement. In the calibrated tier it predicts the measured
minute-1 rectal value plus the population mean rise from minute 1. Intervals
are training-residual 5th/95th percentiles per horizon.
"""
import json
from pathlib import Path

import numpy as np

import heatlib


def train(train_manifest, calibration_manifest, output_dir, seed):
    data = heatlib.load_training(train_manifest)
    tier = data["tier"]
    curves = {}
    for p in data["people"]:
        for r in p["reference"]:
            if r["eligible"]:
                curves.setdefault(r["minute"], []).append(r["rectal_c"])
    mean_curve = {m: float(np.mean(v)) for m, v in curves.items()}
    rise = {m: mean_curve[m] - mean_curve[heatlib.CALIBRATION_MINUTE] for m in mean_curve}
    params = {"tier": tier, "mean_curve": mean_curve, "rise": rise}
    resid = {h: [] for h in heatlib.HORIZONS_MIN}
    for _, req, targets in heatlib.training_requests(data):
        for rec in _predict(params, req, None):
            y = targets[rec["horizon_min"]]
            if y is not None:
                resid[rec["horizon_min"]].append(y - rec["mean_c"])
    params["q"] = {str(h): [float(np.quantile(v, 0.05)), float(np.quantile(v, 0.95))] for h, v in resid.items()}
    Path(output_dir, "params.json").write_text(json.dumps(params))
    return {"params": "params.json"}


def _curve(params, minute, key):
    table = {int(k): v for k, v in params[key].items()}
    if minute in table:
        return table[minute]
    nearest = min(table, key=lambda m: abs(m - minute))
    return table[nearest]


def _predict(params, req, q):
    out = []
    for h in req["horizons_min"]:
        m = req["cutoff_min"] + h
        if req["tier"] == "calibrated":
            mean = req["calibration"]["rectal_c"] + _curve(params, m, "rise")
        else:
            mean = _curve(params, m, "mean_curve")
        lo, hi = q[str(h)] if q else (0.0, 0.0)
        out.append({"horizon_min": h, "mean_c": mean, "lower_c": mean + min(lo, 0.0),
                    "upper_c": mean + max(hi, 0.0)})
    return out


def predict(artifact_dir, request, _cache={}):
    if artifact_dir not in _cache:
        _cache[artifact_dir] = json.loads(Path(artifact_dir, "params.json").read_text())
    params = _cache[artifact_dir]
    return _predict(params, request, params["q"])
