"""Compact gradient boosting baseline (numpy only).

Per horizon: squared-error boosting of depth-2 regression trees on
``heatlib.features``, candidate splits at 16 feature quantiles. Intervals use
5th/95th percentiles of person-grouped out-of-fold residuals.
"""
import json
from pathlib import Path

import numpy as np

import heatlib

ROUNDS, RATE, DEPTH, MIN_LEAF, N_THRESH, FOLDS = 150, 0.05, 2, 10, 16, 7


def _best_split(X, r, idx, thresholds):
    best = None
    rs = r[idx]
    total, n = rs.sum(), len(idx)
    base = total * total / n
    for j, ths in enumerate(thresholds):
        xj = X[idx, j]
        for t in ths:
            left = xj <= t
            nl = int(left.sum())
            if nl < MIN_LEAF or n - nl < MIN_LEAF:
                continue
            sl = rs[left].sum()
            gain = sl * sl / nl + (total - sl) ** 2 / (n - nl) - base
            if best is None or gain > best[0]:
                best = (gain, j, float(t))
    return best


def _grow(X, r, idx, thresholds, depth):
    if depth == DEPTH or len(idx) < 2 * MIN_LEAF:
        return {"v": float(r[idx].mean())}
    split = _best_split(X, r, idx, thresholds)
    if split is None:
        return {"v": float(r[idx].mean())}
    _, j, t = split
    left = idx[X[idx, j] <= t]
    right = idx[X[idx, j] > t]
    return {"j": j, "t": t, "l": _grow(X, r, left, thresholds, depth + 1),
            "r": _grow(X, r, right, thresholds, depth + 1)}


def _tree(node, x):
    while "v" not in node:
        node = node["l"] if x[node["j"]] <= node["t"] else node["r"]
    return node["v"]


def _fit(X, y):
    thresholds = [np.unique(np.quantile(X[:, j], np.linspace(0.05, 0.95, N_THRESH))) for j in range(X.shape[1])]
    f0 = float(y.mean())
    pred = np.full(len(y), f0)
    trees = []
    idx = np.arange(len(y))
    for _ in range(ROUNDS):
        tree = _grow(X, y - pred, idx, thresholds, 0)
        pred += RATE * np.array([_tree(tree, x) for x in X])
        trees.append(tree)
    return {"f0": f0, "trees": trees}


def _apply(m, x):
    return m["f0"] + RATE * sum(_tree(t, x) for t in m["trees"])


def train(train_manifest, calibration_manifest, output_dir, seed):
    data = heatlib.load_training(train_manifest)
    rows = {h: ([], [], []) for h in heatlib.HORIZONS_MIN}
    for pid, req, targets in heatlib.training_requests(data):
        for h in req["horizons_min"]:
            if targets[h] is not None:
                rows[h][0].append(heatlib.features(req, h)); rows[h][1].append(targets[h]); rows[h][2].append(pid)
    rng = np.random.default_rng(seed)
    people = sorted({p["person"] for p in data["people"]})
    fold_of = {p: i % FOLDS for i, p in enumerate(rng.permutation(people))}
    models = {}
    for h, (X, y, g) in rows.items():
        X, y = np.array(X), np.array(y)
        folds = np.array([fold_of[p] for p in g])
        resid = []
        for k in range(FOLDS):
            m = _fit(X[folds != k], y[folds != k])
            resid.extend(y[folds == k] - np.array([_apply(m, x) for x in X[folds == k]]))
        models[str(h)] = {**_fit(X, y), "q": [float(np.quantile(resid, 0.05)), float(np.quantile(resid, 0.95))],
                          "oof_mae": float(np.mean(np.abs(resid)))}
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
