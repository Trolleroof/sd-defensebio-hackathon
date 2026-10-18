"""Numerical baselines, computed by the evaluator on the same windows the agent is scored on.

- persistence:            hold the last observed heart rate
- kinetic_personal:       first-order kinetics fitted on the target's support data only
- kinetic_population:     same model fitted on reference recordings with the transfer block removed
- kinetic_pop_calibrated: population dynamics; personal offset and gain refitted on support
- ridge_auto:             automatic per-horizon ridge on history + future-covariate features,
                          trained on reference cohort + support (an ordinary fitting workflow, no LLM)
"""

from __future__ import annotations

import importlib.util
from functools import lru_cache
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import TaskConfig
from .episodes import Split, Windows, reference_visible_windows, support_windows

_spec = importlib.util.spec_from_file_location(
    "pb_kinetic", Path(__file__).parent / "workspace_template" / "baseline" / "kinetic.py")
kinetic = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(kinetic)


def _x(w: Windows) -> np.ndarray:
    return np.stack([w.fut[f] for f in kinetic.FEATURES], axis=-1)


def _hr0(w: Windows) -> np.ndarray:
    return w.hist["hr_ecg"][:, -1]


@lru_cache(maxsize=8)
def _reference(cfg_path: str, subjects: tuple[str, ...]):
    from .config import load_task
    cfg = load_task(cfg_path)
    return Windows.concat([reference_visible_windows(cfg, s) for s in subjects])


def reference_windows(cfg: TaskConfig) -> Windows:
    return _reference(str(cfg.path), tuple(cfg.reference_subjects))


@lru_cache(maxsize=8)
def _population_params(cfg_path: str, subjects: tuple[str, ...]) -> np.ndarray:
    ref = _reference(cfg_path, subjects)
    return kinetic.fit(_hr0(ref), _x(ref), ref.target)


def fit_all(cfg: TaskConfig, split: Split) -> dict:
    """Fit every baseline for one target participant; returns name -> predictor(Windows)->(N,F)."""
    sup = support_windows(cfg, split)
    ref = reference_windows(cfg)
    k = len(kinetic.FEATURES)

    p_personal = kinetic.fit(_hr0(sup), _x(sup), sup.target)
    p_pop = _population_params(str(cfg.path), tuple(cfg.reference_subjects))

    # Population dynamics with personal offset and gain on the movement terms.
    def calib_sim(q, hr0, x):
        p = p_pop.copy()
        p[0] = q[0]
        p[1:1 + k] = p_pop[1:1 + k] * q[1]
        return kinetic.simulate(p, hr0, x)

    from scipy.optimize import least_squares
    q = least_squares(lambda q: (calib_sim(q, _hr0(sup), _x(sup)) - sup.target).ravel(),
                      [p_pop[0], 1.0], bounds=([20, 0.2], [200, 5.0]), loss="soft_l1", f_scale=5.0).x

    ridge = _fit_ridge(cfg, ref, sup)

    return {
        "persistence": lambda w: np.repeat(_hr0(w)[:, None], w.target.shape[1], axis=1),
        "kinetic_personal": lambda w: kinetic.simulate(p_personal, _hr0(w), _x(w)),
        "kinetic_population": lambda w: kinetic.simulate(p_pop, _hr0(w), _x(w)),
        "kinetic_pop_calibrated": lambda w: calib_sim(q, _hr0(w), _x(w)),
        "ridge_auto": ridge,
    }


def _ridge_features(cfg: TaskConfig, w: Windows) -> list[np.ndarray]:
    """Per-horizon-step design matrices."""
    hr = w.hist["hr_ecg"]
    base = [hr[:, -1], hr[:, -5:].mean(1), hr.mean(1), hr[:, -1] - hr[:, -10]]
    num = [c for c in cfg.input_channels if c != "activity"]
    hist_x = [w.hist[c].mean(1) for c in num]
    fut = np.stack([w.fut[c] for c in num], -1)  # (N,F,k)
    cum = np.cumsum(fut, 1) / (np.arange(fut.shape[1]) + 1)[None, :, None]
    ew = np.zeros_like(fut)
    acc = fut[:, 0]
    for i in range(fut.shape[1]):
        acc = acc + (fut[:, i] - acc) * (2.0 / 30.0)
        ew[:, i] = acc
    out = []
    for i in range(fut.shape[1]):
        out.append(np.column_stack(base + hist_x + [fut[:, i], cum[:, i], ew[:, i]]))
    return out


def _fit_ridge(cfg: TaskConfig, ref: Windows, sup: Windows):
    tr = Windows.concat([ref, sup])
    weights = np.r_[np.ones(len(ref)), np.full(len(sup), 3.0)]
    feats = _ridge_features(cfg, tr)
    hr0 = tr.hist["hr_ecg"][:, -1]
    models = []
    for i, X in enumerate(feats):
        m = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        m.fit(X, tr.target[:, i] - hr0, ridge__sample_weight=weights)
        models.append(m)

    def predict(w: Windows) -> np.ndarray:
        fs = _ridge_features(cfg, w)
        h0 = w.hist["hr_ecg"][:, -1]
        return np.column_stack([m.predict(X) + h0 for m, X in zip(models, fs)])

    return predict
