"""Regularized lag-feature regression baseline.

One ridge model per horizon on standardized ``heatlib.features``. The penalty
is chosen by leave-one-person-out error on training people; intervals are the
5th/95th percentiles of the corresponding out-of-person residuals.
"""
import json
from pathlib import Path

import numpy as np

import heatlib

LAMBDAS = (0.1, 1.0, 10.0, 100.0, 1000.0)


def _design(data):
    rows = {h: ([], [], []) for h in heatlib.HORIZONS_MIN}
    for pid, req, targets in heatlib.training_requests(data):
        for h in req["horizons_min"]:
            if targets[h] is not None:
                X, y, g = rows[h]
                X.append(heatlib.features(req, h)); y.append(targets[h]); g.append(pid)
    return {h: (np.array(X), np.array(y), np.array(g)) for h, (X, y, g) in rows.items()}


def _fit(X, y, lam):
    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-9] = 1.0
    Z = (X - mu) / sd
    Z[:, 0] = 1.0
    P = lam * np.eye(Z.shape[1]); P[0, 0] = 0.0
    w = np.linalg.solve(Z.T @ Z + P, Z.T @ y)
    return {"mu": mu.tolist(), "sd": sd.tolist(), "w": w.tolist()}


def _apply(m, x):
    z = (np.asarray(x) - np.asarray(m["mu"])) / np.asarray(m["sd"])
    z[0] = 1.0
    return float(z @ np.asarray(m["w"]))


def _lopo(X, y, g, lam):
    resid = []
    for p in np.unique(g):
        tr = g != p
        m = _fit(X[tr], y[tr], lam)
        resid.extend(y[~tr] - np.array([_apply(m, x) for x in X[~tr]]))
    return np.array(resid)


def train(train_manifest, calibration_manifest, output_dir, seed):
    data = heatlib.load_training(train_manifest)
    models = {}
    for h, (X, y, g) in _design(data).items():
        best = min(((float(np.mean(np.abs(_lopo(X, y, g, lam)))), lam) for lam in LAMBDAS))
        resid = _lopo(X, y, g, best[1])
        models[str(h)] = {**_fit(X, y, best[1]), "lambda": best[1], "lopo_mae": best[0],
                          "q": [float(np.quantile(resid, 0.05)), float(np.quantile(resid, 0.95))]}
    Path(output_dir, "params.json").write_text(json.dumps({"tier": data["tier"], "models": models}))
    return {"params": "params.json"}


def predict(artifact_dir, request, _cache={}):
    if artifact_dir not in _cache:
        _cache[artifact_dir] = json.loads(Path(artifact_dir, "params.json").read_text())
    models = _cache[artifact_dir]["models"]
    out = []
    for h in request["horizons_min"]:
        m = models[str(h)]
        mean = _apply(m, heatlib.features(request, h))
        out.append({"horizon_min": h, "mean_c": mean, "lower_c": mean + min(m["q"][0], 0.0),
                    "upper_c": mean + max(m["q"][1], 0.0)})
    return out
