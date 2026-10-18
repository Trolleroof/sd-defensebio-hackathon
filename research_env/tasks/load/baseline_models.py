"""Self-contained baseline artifacts for the load-carriage task.

This file is copied verbatim into each artifact directory as ``predict.py``,
so it imports only the standard library and numpy. Fitted parameters are JSON
(no pickle). Shared interface:

    train(train_manifest, calibration_manifest, output_dir, seed) -> artifact_manifest
    predict(artifact_dir, request) -> {"mean_w", "lower_w", "upper_w"}

``train_manifest`` = {"records_path", "track", "model", "hyperparameters"}.
Training rows carry an opaque ``person_key`` used only for grouped
cross-validation; it is never a model feature.

Intervals are split-conformal: absolute residuals from person-grouped
cross-validation on the training people, quantile at ceil((n+1)*0.9)/n.
"""
from __future__ import annotations

import hashlib
import json
import math
import shutil
from pathlib import Path

import numpy as np

MODELS = ("mean_by_load", "ridge", "gbm", "pandolf", "pandolf_recal", "calibration_offset")
DEFAULTS = {
    "mean_by_load": {},
    "ridge": {"alphas": [0.01, 0.1, 1.0, 10.0, 100.0], "use_sex": False,
              "features": ["mass", "stature", "age", "load", "mass_x_load", "load2_per_mass"]},
    "gbm": {"n_estimators": 100, "learning_rate": 0.05, "max_depth": 2, "min_leaf": 6, "use_sex": False},
    "pandolf": {"terrain_factor": 1.0},
    "pandolf_recal": {"terrain_factor": 1.0},
    "calibration_offset": {},
}
CV_FOLDS = 4


# --------------------------------------------------------------------------- features

def _x(row: dict, names: list[str], use_sex: bool) -> list[float]:
    m, h, a, L = row["body_mass_kg"], row["stature_cm"], row["age_years"], row["load_kg"]
    table = {"mass": m, "stature": h, "age": a, "load": L, "mass_x_load": m * L,
             "load2_per_mass": L * L / m, "total_mass": m + L, "load_frac": L / m}
    out = [float(table[n]) for n in names]
    if use_sex:
        out.append(1.0 if row["sex"] == "M" else 0.0)
    return out


def pandolf_watts(body_mass_kg, load_kg, speed_m_s, grade_fraction, terrain_factor=1.0):
    """Pandolf, Givoni & Goldman (1977), J Appl Physiol 43:577. Gross metabolic rate in W.

    M = 1.5 W + 2.0 (W + L)(L / W)^2 + eta (W + L)(1.5 V^2 + 0.35 V G),
    W body mass kg, L load kg, V speed m/s, G grade in percent, eta = 1.0 for a treadmill.
    """
    W, L, V, G = body_mass_kg, load_kg, speed_m_s, 100.0 * grade_fraction
    return 1.5 * W + 2.0 * (W + L) * (L / W) ** 2 + terrain_factor * (W + L) * (1.5 * V * V + 0.35 * V * G)


# --------------------------------------------------------------------------- tiny boosted trees

def _fit_tree(X, r, depth, min_leaf):
    n, d = X.shape
    if depth == 0 or n < 2 * min_leaf:
        return {"value": float(r.mean())}
    best = None
    base = ((r - r.mean()) ** 2).sum()
    for j in range(d):
        order = np.argsort(X[:, j], kind="stable")
        xs, rs = X[order, j], r[order]
        csum = np.cumsum(rs)
        csq = np.cumsum(rs * rs)
        tot, totsq = csum[-1], csq[-1]
        for i in range(min_leaf - 1, n - min_leaf):
            if xs[i] == xs[i + 1]:
                continue
            nl, nr = i + 1, n - i - 1
            sse = (csq[i] - csum[i] ** 2 / nl) + ((totsq - csq[i]) - (tot - csum[i]) ** 2 / nr)
            if best is None or sse < best[0] - 1e-12:
                best = (sse, j, 0.5 * (xs[i] + xs[i + 1]))
    if best is None or best[0] >= base - 1e-12:
        return {"value": float(r.mean())}
    _, j, thr = best
    mask = X[:, j] <= thr
    return {"feature": int(j), "threshold": float(thr),
            "left": _fit_tree(X[mask], r[mask], depth - 1, min_leaf),
            "right": _fit_tree(X[~mask], r[~mask], depth - 1, min_leaf)}


def _tree_value(tree, x):
    while "value" not in tree:
        tree = tree["left"] if x[tree["feature"]] <= tree["threshold"] else tree["right"]
    return tree["value"]


# --------------------------------------------------------------------------- model fitting

GBM_FEATURES = ["mass", "stature", "age", "load"]


def _fit(model: str, rows: list[dict], hp: dict) -> dict:
    y = np.array([r["reference_w"] for r in rows], dtype=float)
    if model == "mean_by_load":
        means = {}
        for load in sorted({r["load_kg"] for r in rows}):
            means[f"{load:g}"] = float(np.mean([r["reference_w"] for r in rows if r["load_kg"] == load]))
        loads = np.array([float(k) for k in means])
        vals = np.array(list(means.values()))
        slope, intercept = np.polyfit(loads, vals, 1) if len(loads) > 1 else (0.0, float(vals[0]))
        return {"means": means, "extrapolate": {"intercept": float(intercept), "slope": float(slope)}}
    if model == "ridge":
        names, use_sex = list(hp["features"]), bool(hp["use_sex"])
        X = np.array([_x(r, names, use_sex) for r in rows])
        alpha = hp.get("alpha")
        if alpha is None:  # inner grouped CV over the alpha grid, training people only
            alpha = min(hp["alphas"], key=lambda a: _cv_mae(rows, "ridge", dict(hp, alpha=a)))
        mu, sd = X.mean(0), X.std(0)
        sd[sd == 0] = 1.0
        Z = (X - mu) / sd
        Z1 = np.hstack([np.ones((len(Z), 1)), Z])
        pen = alpha * np.eye(Z1.shape[1])
        pen[0, 0] = 0.0
        beta = np.linalg.solve(Z1.T @ Z1 + pen, Z1.T @ y)
        return {"features": names, "use_sex": use_sex, "alpha": float(alpha),
                "mu": mu.tolist(), "sd": sd.tolist(), "beta": beta.tolist()}
    if model == "gbm":
        use_sex = bool(hp["use_sex"])
        X = np.array([_x(r, GBM_FEATURES, use_sex) for r in rows])
        base = float(y.mean())
        pred = np.full(len(y), base)
        trees = []
        for _ in range(int(hp["n_estimators"])):
            tree = _fit_tree(X, y - pred, int(hp["max_depth"]), int(hp["min_leaf"]))
            trees.append(tree)
            pred += hp["learning_rate"] * np.array([_tree_value(tree, x) for x in X])
        return {"use_sex": use_sex, "base": base, "learning_rate": float(hp["learning_rate"]), "trees": trees}
    if model == "pandolf":
        return {"terrain_factor": float(hp["terrain_factor"])}
    if model == "pandolf_recal":
        p = np.array([pandolf_watts(r["body_mass_kg"], r["load_kg"], r["speed_m_s"], r["grade_fraction"],
                                    hp["terrain_factor"]) for r in rows])
        slope, intercept = np.polyfit(p, y, 1)
        return {"terrain_factor": float(hp["terrain_factor"]), "slope": float(slope), "intercept": float(intercept)}
    if model == "calibration_offset":
        unloaded = {r["person_key"]: r["reference_w"] for r in rows if r["load_kg"] == 0.0}
        deltas: dict[str, list[float]] = {}
        for r in rows:
            if r["load_kg"] != 0.0 and r["person_key"] in unloaded:
                deltas.setdefault(f"{r['load_kg']:g}", []).append(r["reference_w"] - unloaded[r["person_key"]])
        return {"delta_w": {k: float(np.mean(v)) for k, v in deltas.items()}}
    raise ValueError(f"unknown model {model}")


def _point(model: str, params: dict, req: dict) -> float:
    if model == "mean_by_load":
        key = f"{req['load_kg']:g}"
        if key in params["means"]:
            return params["means"][key]
        e = params["extrapolate"]
        return e["intercept"] + e["slope"] * req["load_kg"]
    if model == "ridge":
        x = (np.array(_x(req, params["features"], params["use_sex"])) - params["mu"]) / params["sd"]
        return float(params["beta"][0] + x @ np.array(params["beta"][1:]))
    if model == "gbm":
        x = _x(req, GBM_FEATURES, params["use_sex"])
        return params["base"] + params["learning_rate"] * sum(_tree_value(t, x) for t in params["trees"])
    if model in ("pandolf", "pandolf_recal"):
        p = pandolf_watts(req["body_mass_kg"], req["load_kg"], req["speed_m_s"], req["grade_fraction"],
                          params["terrain_factor"])
        return p if model == "pandolf" else params["intercept"] + params["slope"] * p
    if model == "calibration_offset":
        cal = req.get("calibration")
        if not cal:
            raise ValueError("calibration_offset requires a calibration measurement")
        return cal["measured_w"] + params["delta_w"].get(f"{req['load_kg']:g}", 0.0)
    raise ValueError(f"unknown model {model}")


def _calibration_view(row: dict, rows: list[dict]) -> dict:
    """Training-time analogue of a calibration request: reveal the same person's 0 kg value."""
    req = dict(row)
    unloaded = [r["reference_w"] for r in rows if r["person_key"] == row["person_key"] and r["load_kg"] == 0.0]
    req["calibration"] = {"load_kg": 0.0, "measured_w": unloaded[0]} if unloaded else None
    return req


def _folds(rows: list[dict], seed: int = 0) -> list[set]:
    keys = sorted({r["person_key"] for r in rows},
                  key=lambda k: hashlib.sha256(f"{seed}:{k}".encode()).hexdigest())
    return [set(keys[i::CV_FOLDS]) for i in range(CV_FOLDS)]


def _cv_residuals(rows: list[dict], model: str, hp: dict, seed: int = 0) -> list[float]:
    resid = []
    for held in _folds(rows, seed):
        fit_rows = [r for r in rows if r["person_key"] not in held]
        test_rows = [r for r in rows if r["person_key"] in held]
        params = _fit(model, fit_rows, hp)
        for r in test_rows:
            req = _calibration_view(r, test_rows) if model == "calibration_offset" else r
            if model == "calibration_offset" and (r["load_kg"] == 0.0 or req["calibration"] is None):
                continue
            resid.append(r["reference_w"] - _point(model, params, req))
    return resid


def _cv_mae(rows, model, hp):
    return float(np.mean(np.abs(_cv_residuals(rows, model, hp))))


def _conformal_q90(abs_resid: list[float]) -> float:
    a = np.sort(np.abs(np.asarray(abs_resid, dtype=float)))
    n = len(a)
    k = min(n, int(math.ceil((n + 1) * 0.9)))
    return float(a[k - 1])


# --------------------------------------------------------------------------- interface

def _read_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def train(train_manifest, calibration_manifest, output_dir, seed):
    model = train_manifest["model"]
    hp = dict(DEFAULTS[model], **(train_manifest.get("hyperparameters") or {}))
    rows = [r for r in _read_jsonl(train_manifest["records_path"]) if r.get("reference_w") is not None]
    params = _fit(model, rows, hp)
    resid = _cv_residuals(rows, model, dict(hp, alpha=params.get("alpha")), seed)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {"model": model, "hyperparameters": hp, "params": params, "seed": seed,
               "interval": {"method": "split-conformal on person-grouped CV residuals",
                            "q90_w": _conformal_q90(resid), "n_residuals": len(resid),
                            "cv_mae_w": float(np.mean(np.abs(resid)))}}
    (out / "params.json").write_text(json.dumps(payload, indent=1, sort_keys=True))
    shutil.copyfile(__file__, out / "predict.py")
    manifest = {
        "model": model,
        "track": train_manifest.get("track"),
        "seed": seed,
        "uses_sex": bool(hp.get("use_sex", False)),
        "uses_calibration": model == "calibration_offset",
        "n_training_rows": len(rows),
        "n_training_people": len({r["person_key"] for r in rows}),
        "files": {name: hashlib.sha256((out / name).read_bytes()).hexdigest()
                  for name in ("params.json", "predict.py")},
    }
    (out / "artifact_manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))
    return manifest


_CACHE: dict = {}


def predict(artifact_dir, request):
    key = str(artifact_dir)
    if key not in _CACHE:
        _CACHE[key] = json.loads((Path(artifact_dir) / "params.json").read_text())
    art = _CACHE[key]
    mean = float(_point(art["model"], art["params"], request))
    q = art["interval"]["q90_w"]
    return {"mean_w": mean, "lower_w": mean - q, "upper_w": mean + q}
