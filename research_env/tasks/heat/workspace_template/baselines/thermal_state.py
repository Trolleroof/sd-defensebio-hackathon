"""Simple first-order thermal state-space baseline.

Latent core temperature evolves once per minute:

    T[t+1] = T[t] + a + b * W[t] / mass - k * (T[t] - Tskin[t])

W is work rate and Tskin the device skin temperature, each carried forward from
its last available value (no future values are used; after the cutoff both are
held at their last value). T[1] is a fitted population constant in the
wearable tier and the revealed minute-1 rectal value in the calibrated tier.
Parameters are fitted by damped Gauss-Newton on open-loop simulations of the
training people. Intervals: training forecast residual 5th/95th percentiles.
"""
import json
from pathlib import Path

import numpy as np

import heatlib


def _carry(series, minutes, default):
    out, j, cur = [], 0, (series[0][1] if series else default)
    for m in minutes:
        while j < len(series) and series[j][0] <= m:
            cur = series[j][1]; j += 1
        out.append(cur)
    return np.array(out)


def _simulate(theta, t1, work_per_kg, skin, steps):
    a, b, k = theta[1], theta[2], theta[3]
    T = np.empty(steps)
    T[0] = t1
    for i in range(steps - 1):
        T[i + 1] = T[i] + a + b * work_per_kg[i] - k * (T[i] - skin[i])
    return T


def _drivers(req, last_minute):
    """Driver arrays for minutes 1..last_minute using only request history."""
    minutes = list(range(1, last_minute + 1))
    hist = req["history"]
    work = _carry(hist["work_rate_w"], minutes, 150.0) / req["context"]["mass_kg"]
    skin = _carry(hist["device_skin_c"], minutes, 35.5)
    return work, skin


def _forecast(theta, req):
    c = req["cutoff_min"]
    hs = req["horizons_min"]
    last = c + max(hs) if hs else c
    work, skin = _drivers(req, last)
    t1 = req["calibration"]["rectal_c"] if req["tier"] == "calibrated" else theta[0]
    T = _simulate(theta, t1, work, skin, last)
    return {h: float(T[c + h - 1]) for h in hs}


def train(train_manifest, calibration_manifest, output_dir, seed):
    data = heatlib.load_training(train_manifest)
    tier = data["tier"]
    series = []
    for p in data["people"]:
        ref = {r["minute"]: r["rectal_c"] for r in p["reference"] if r["eligible"]}
        last = max(ref)
        full = heatlib.request_at(p, last, tier)
        work, skin = _drivers(full, last)
        t1 = full["calibration"]["rectal_c"] if tier == "calibrated" else None
        idx = np.array(sorted(ref)) - 1
        series.append((work, skin, last, t1, idx, np.array([ref[m] for m in sorted(ref)])))

    def residuals(theta):
        out = []
        for work, skin, last, t1, idx, y in series:
            T = _simulate(theta, theta[0] if t1 is None else t1, work, skin, last)
            out.append(T[idx] - y)
        return np.concatenate(out)

    theta = np.array([37.0, 0.02, 0.0, 0.0])
    lam = 1e-2
    cost = float(np.sum(residuals(theta) ** 2))
    for _ in range(200):
        r = residuals(theta)
        J = np.empty((len(r), 4))
        for j in range(4):
            d = np.zeros(4); d[j] = 1e-6 * max(1.0, abs(theta[j]))
            J[:, j] = (residuals(theta + d) - r) / d[j]
        step = np.linalg.solve(J.T @ J + lam * np.diag(np.diag(J.T @ J) + 1e-12), -J.T @ r)
        cand = theta + step
        new = float(np.sum(residuals(cand) ** 2))
        if new < cost:
            theta, cost, lam = cand, new, lam / 3
            if np.max(np.abs(step)) < 1e-9:
                break
        else:
            lam *= 5
    resid = {h: [] for h in heatlib.HORIZONS_MIN}
    for _, req, targets in heatlib.training_requests(data):
        for h, mean in _forecast(theta, req).items():
            if targets[h] is not None:
                resid[h].append(targets[h] - mean)
    params = {"tier": tier, "theta": theta.tolist(), "train_sse": cost,
              "q": {str(h): [float(np.quantile(v, 0.05)), float(np.quantile(v, 0.95))] for h, v in resid.items()}}
    Path(output_dir, "params.json").write_text(json.dumps(params))
    return {"params": "params.json"}


def predict(artifact_dir, request, _cache={}):
    if artifact_dir not in _cache:
        _cache[artifact_dir] = json.loads(Path(artifact_dir, "params.json").read_text())
    params = _cache[artifact_dir]
    means = _forecast(np.array(params["theta"]), request)
    out = []
    for h in request["horizons_min"]:
        lo, hi = params["q"][str(h)]
        out.append({"horizon_min": h, "mean_c": means[h], "lower_c": means[h] + min(lo, 0.0),
                    "upper_c": means[h] + max(hi, 0.0)})
    return out
