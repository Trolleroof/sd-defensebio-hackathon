"""Local smoke test: run submission/predict.py on windows cut from YOUR support data.

    python tools/check_submission.py [--stride 15]

This is in-sample for anything fitted on support data, so it is a format and
sanity check, not an estimate of hidden performance.
"""

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--stride", type=int, default=15)
args = ap.parse_args()

channels = json.loads((ROOT / "docs" / "channels.json").read_text())["input_channels"]
df = pd.read_csv(ROOT / "data" / "target" / "support.csv")
spec = importlib.util.spec_from_file_location("sub", ROOT / "submission" / "predict.py")
mod = importlib.util.module_from_spec(spec)
sys.path.insert(0, str(ROOT / "submission"))
spec.loader.exec_module(mod)

H, F = 30, 60
errs, n, t0 = [], 0, time.time()
for _, g in df.groupby("segment"):
    for s in range(0, len(g) - H - F + 1, args.stride):
        w = g.iloc[s:s + H + F]
        hist = {c: w[c].to_numpy(float)[:H] for c in ["t", "hr_ecg"] + channels}
        fut = {c: w[c].to_numpy(float)[H:] for c in ["t"] + channels}
        p = np.asarray(mod.predict(hist, fut), float).reshape(-1)
        assert p.shape == (F,), f"predict returned {p.shape}, expected ({F},)"
        assert np.isfinite(p).all(), "non-finite prediction"
        errs.append(np.abs(p - w["hr_ecg"].to_numpy(float)[H:]).mean())
        n += 1
print(f"OK: {n} support windows, mean abs error {np.mean(errs):.2f} bpm (in-sample), {time.time() - t0:.1f}s")
