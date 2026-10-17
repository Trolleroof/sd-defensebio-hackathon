"""Conventional baseline: first-order heart-rate kinetics driven by movement.

    HR_ss(t) = a + b . x(t)                       (steady state set by movement load)
    dHR/dt   = (HR_ss(t) - HR(t)) / tau            (tau_up when rising, tau_down when falling)

x(t) = movement features. Parameters are fitted by least squares on
multi-step forecasts, starting each forecast from the last observed HR.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

FEATURES = ["acc_wrist_enmo", "acc_chest_enmo", "acc_chest_std"]
STEP_S = 2.0
HISTORY_N = 30  # 60 s
HORIZON_N = 60  # 120 s


def simulate(params: np.ndarray, hr0: np.ndarray, x: np.ndarray) -> np.ndarray:
    """hr0: (N,), x: (N, F, k) -> predicted HR (N, F)."""
    k = x.shape[2]
    a, b = params[0], params[1:1 + k]
    tau_up, tau_down = np.exp(params[1 + k]), np.exp(params[2 + k])
    target = a + x @ b
    h = hr0.astype(float).copy()
    out = np.empty(x.shape[:2])
    for i in range(x.shape[1]):
        d = target[:, i] - h
        tau = np.where(d > 0, tau_up, tau_down)
        h = h + STEP_S / tau * d
        out[:, i] = h
    return out


def initial_params(hr: np.ndarray, k: int) -> np.ndarray:
    return np.r_[np.nanpercentile(hr, 20), np.full(k, 20.0), np.log(30.0), np.log(60.0)]


def fit(hr0, x, y, init=None, fixed: dict | None = None) -> np.ndarray:
    """Least-squares fit. ``fixed`` maps parameter index -> value to hold constant."""
    k = x.shape[2]
    p0 = np.asarray(init if init is not None else initial_params(y, k), float)
    fixed = fixed or {}
    free = [i for i in range(len(p0)) if i not in fixed]

    def full(q):
        p = p0.copy()
        p[free] = q
        for i, v in fixed.items():
            p[i] = v
        return p

    lo = np.r_[20.0, np.full(k, -200.0), np.log(4.0), np.log(4.0)][free]
    hi = np.r_[200.0, np.full(k, 2000.0), np.log(900.0), np.log(900.0)][free]
    res = least_squares(lambda q: (simulate(full(q), hr0, x) - y).ravel(), np.clip(p0[free], lo + 1e-6, hi - 1e-6),
                        bounds=(lo, hi), loss="soft_l1", f_scale=5.0)
    return full(res.x)


def windows_from_frame(df: pd.DataFrame, stride: int = 5, segment_col: str | None = "segment"):
    """Cut (history, horizon) training windows from a time-ordered frame.

    Windows never cross a segment boundary. Returns (hr_hist (N,H), x_fut (N,F,k), y_fut (N,F)).
    """
    groups = df.groupby(segment_col) if segment_col and segment_col in df else [(0, df)]
    H, F = HISTORY_N, HORIZON_N
    hh, xx, yy = [], [], []
    for _, g in groups:
        hr = g["hr_ecg"].to_numpy(float)
        x = g[FEATURES].to_numpy(float)
        for s in range(0, len(g) - H - F + 1, stride):
            if np.isfinite(hr[s:s + H + F]).all():
                hh.append(hr[s:s + H])
                xx.append(x[s + H:s + H + F])
                yy.append(hr[s + H:s + H + F])
    if not hh:
        return np.zeros((0, H)), np.zeros((0, F, len(FEATURES))), np.zeros((0, F))
    return np.array(hh), np.array(xx), np.array(yy)
