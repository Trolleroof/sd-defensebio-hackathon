"""Fit the conventional kinetic baseline on this participant's support data.

    python baseline/fit_baseline.py      # writes baseline/params.json
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

import kinetic

HERE = Path(__file__).resolve().parent
df = pd.read_csv(HERE.parent / "data" / "target" / "support.csv")
hh, xx, yy = kinetic.windows_from_frame(df, stride=5)
params = kinetic.fit(hh[:, -1], xx, yy)
pred = kinetic.simulate(params, hh[:, -1], xx)
print(f"support windows: {len(yy)}  in-sample MAE: {np.abs(pred - yy).mean():.2f} bpm")
print("params:", dict(zip(["a"] + [f"b_{f}" for f in kinetic.FEATURES] + ["log_tau_up", "log_tau_down"], np.round(params, 3))))
(HERE / "params.json").write_text(json.dumps({"params": params.tolist(), "features": kinetic.FEATURES}))
