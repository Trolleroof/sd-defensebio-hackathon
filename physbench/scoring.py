"""Metrics. Primary: phase-balanced MAE (mean of exertion-phase and rest/recovery-phase MAE)."""

from __future__ import annotations

import math

import numpy as np

from .data.dalia import ACTIVITY_NAMES

HORIZON_BINS_S = [(0, 30), (30, 60), (60, 120)]


def validate_predictions(pred, n: int, f: int, hr_range) -> tuple[bool, str]:
    if not isinstance(pred, np.ndarray):
        return False, "predictions are not an array"
    if pred.shape != (n, f):
        return False, f"expected shape {(n, f)}, got {pred.shape}"
    if not np.all(np.isfinite(pred)):
        return False, f"{int((~np.isfinite(pred)).sum())} non-finite predictions"
    lo, hi = hr_range
    bad = int(((pred < lo) | (pred > hi)).sum())
    if bad:
        return False, f"{bad} predictions outside plausible range [{lo}, {hi}] bpm"
    return True, "ok"


def score(pred: np.ndarray, y: np.ndarray, exertion: np.ndarray, activity: np.ndarray, step_s: float) -> dict:
    err = np.abs(pred - y)
    out = {
        "n_windows": int(len(y)),
        "mae_all": float(err.mean()),
        "bias": float((pred - y).mean()),
        "mae_exertion": float(err[exertion].mean()) if exertion.any() else None,
        "mae_rest": float(err[~exertion].mean()) if (~exertion).any() else None,
    }
    phases = [v for v in (out["mae_exertion"], out["mae_rest"]) if v is not None]
    out["phase_balanced_mae_bpm"] = float(np.mean(phases))
    lead = (np.arange(y.shape[1]) + 1) * step_s
    out["mae_by_horizon_s"] = {
        f"{a}-{b}": float(err[:, (lead > a) & (lead <= b)].mean()) for a, b in HORIZON_BINS_S
    }
    out["mae_by_activity"] = {
        ACTIVITY_NAMES[a]: float(err[activity == a].mean()) for a in np.unique(activity)
    }
    return out


def improvement(agent_mae: float, baseline_mae: float, min_denominator: float = 1.0) -> float:
    """1 - agent/baseline. Positive = better than baseline."""
    return 1.0 - agent_mae / max(baseline_mae, min_denominator)


def bootstrap_ci(values, stat=np.mean, n_boot: int = 10000, alpha: float = 0.05, seed: int = 0):
    v = np.asarray([x for x in values if x is not None and not (isinstance(x, float) and math.isnan(x))], float)
    if len(v) == 0:
        return (None, None, None)
    rng = np.random.default_rng(seed)
    boots = stat(v[rng.integers(0, len(v), (n_boot, len(v)))], axis=1)
    return float(stat(v)), float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def wilson_ci(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (None, None, None)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)
