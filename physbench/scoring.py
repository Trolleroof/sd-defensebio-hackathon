"""Metrics. Primary: phase-balanced MAE (mean of exertion-phase and rest/recovery-phase MAE)."""

from __future__ import annotations

import math

import numpy as np

from .data.dalia import ACTIVITY_NAMES

HORIZON_BINS_S = [(0, 30), (30, 60), (60, 120)]


def validate_predictions(pred, n: int, f: int, hr_range) -> tuple[bool, str]:
    if not isinstance(pred, np.ndarray):
        return False, "predictions are not an array"
    if pred.ndim != 2 or pred.shape != (n, f):
        return False, (
            f"point forecasts only, shape {(n, f)}; got {getattr(pred, 'shape', None)}. "
            "Intervals, quantiles, and extra columns are rejected and are not scored."
        )
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


ELO_BASE, ELO_SCALE = 1500.0, 400.0
# Margin temperatures: a gap of one tau maps to an outcome of ~0.88 for the leader.
ACC_TAU = 0.10    # improvement-vs-anchor gap (0.10 = 10 points)
JUDGE_TAU = 0.25  # judge-score gap on the 0-1 scale
ACC_WEIGHT = 0.7  # combined outcome = ACC_WEIGHT * accuracy + (1 - ACC_WEIGHT) * judge


def _soft_win(diff: float, tau: float) -> float:
    """Margin-aware outcome in [0, 1]: 0.5 for a tie, approaching 1 as the lead grows."""
    return 0.5 + 0.5 * math.tanh(diff / tau)


def matchup_outcome(a: dict, b: dict, acc_weight: float = ACC_WEIGHT) -> dict:
    """Soft outcome of episode a vs episode b on the same subject, from a's side.

    Accuracy: only valid submissions that pass integrity are scored; a scored episode beats an
    unscored one outright, two unscored episodes tie, otherwise the improvement gap sets the margin.
    Judge: the judge-score gap sets the margin; an integrity failure loses outright, and the judge
    game is dropped when either judge call failed.
    """
    oka, okb = bool((a.get("criteria") or {}).get("integrity")), bool((b.get("criteria") or {}).get("integrity"))
    ia = a.get("improvement_vs_anchor") if oka else None
    ib = b.get("improvement_vs_anchor") if okb else None
    if ia is None and ib is None:
        acc = 0.5
    elif ib is None:
        acc = 1.0
    elif ia is None:
        acc = 0.0
    else:
        acc = _soft_win(ia - ib, ACC_TAU)
    ja, jb = (a.get("judge") or {}).get("judge_score"), (b.get("judge") or {}).get("judge_score")
    if ja is None or jb is None:
        judge = None
    elif oka != okb:
        judge = float(oka)
    else:
        judge = _soft_win(ja - jb, JUDGE_TAU)
    combined = acc if judge is None else acc_weight * acc + (1 - acc_weight) * judge
    return {"accuracy": acc, "judge": judge, "combined": combined}


def matchups(results: list[dict], models: list[str], subjects: list[str]) -> list[tuple]:
    """Every episode vs every other model's episode on the same subject: (subject, a, b, outcomes, weight).

    Weight 1/(n_a * n_b) so each subject counts once per model pair regardless of repeats.
    """
    out = []
    for s in subjects:
        by_model = {m: [r for r in results if r.get("model") == m and r.get("subject") == s
                        and "harness_error" not in r and not r.get("infra_error")] for m in models}
        for i, a in enumerate(models):
            for b in models[i + 1:]:
                ra, rb = by_model[a], by_model[b]
                for x in ra:
                    for y in rb:
                        out.append((s, a, b, matchup_outcome(x, y), 1.0 / (len(ra) * len(rb))))
    return out


def fit_bradley_terry(games: list[tuple[str, str, float, float]], models: list[str],
                      l2: float = 0.01, iters: int = 100) -> dict:
    """Bradley-Terry with fractional outcomes, fitted by Newton's method.

    games: (a, b, outcome for a in [0, 1], weight). Maximises
    sum w * [o log p + (1 - o) log(1 - p)], p = sigmoid(theta_a - theta_b), with a small L2 prior
    so a model that wins every game still gets a finite rating. Returns Elo ratings (mean ELO_BASE).
    """
    idx = {m: k for k, m in enumerate(models)}
    theta = np.zeros(len(models))
    for _ in range(iters):
        grad, hess = -l2 * theta, -l2 * np.eye(len(models))
        for a, b, o, w in games:
            i, j = idx[a], idx[b]
            p = 1.0 / (1.0 + math.exp(theta[j] - theta[i]))
            g, h = w * (o - p), w * p * (1 - p)
            grad[i] += g
            grad[j] -= g
            hess[i, i] -= h
            hess[j, j] -= h
            hess[i, j] += h
            hess[j, i] += h
        step = np.linalg.lstsq(hess, grad, rcond=None)[0]
        theta -= step
        if np.abs(step).max() < 1e-9:
            break
    theta -= theta.mean()
    return {m: float(ELO_BASE + ELO_SCALE / math.log(10) * theta[idx[m]]) for m in models}


def elo_ratings(results: list[dict], models: list[str], subjects: list[str], n_boot: int = 1000,
                seed: int = 0) -> dict:
    """Elo per outcome kind ('accuracy', 'judge', 'combined'), each (point, lo, hi).

    Intervals resample subjects with replacement, since subjects are the independent cases.
    """
    games = matchups(results, models, subjects)

    def fit(kind, subset):
        g = [(a, b, o[kind], w) for s, a, b, o, w in subset if o[kind] is not None]
        return fit_bradley_terry(g, models) if g else None

    by_subject = {s: [x for x in games if x[0] == s] for s in subjects}
    rng = np.random.default_rng(seed)
    out = {}
    for kind in ("accuracy", "judge", "combined"):
        point = fit(kind, games)
        if point is None:
            out[kind] = {m: (None, None, None) for m in models}
            continue
        boots = []
        for _ in range(n_boot):
            sample = [x for s in rng.choice(subjects, len(subjects)) for x in by_subject[s]]
            e = fit(kind, sample)
            if e is not None:
                boots.append([e[m] for m in models])
        boots = np.asarray(boots)
        out[kind] = {m: (point[m], float(np.quantile(boots[:, k], 0.025)), float(np.quantile(boots[:, k], 0.975)))
                     for k, m in enumerate(models)}
    return out


def elo_win_prob(r_i: float, r_j: float) -> float:
    return 1.0 / (1.0 + 10 ** ((r_j - r_i) / ELO_SCALE))


def wilson_ci(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (None, None, None)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)
