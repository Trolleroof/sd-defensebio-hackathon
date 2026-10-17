"""Submission contract (copy this shape into submission/predict.py).

predict(history, future) -> np.ndarray of shape (len(future["t"]),), heart rate in bpm.

history: dict of 1-D arrays over the last 60 s (30 samples at 0.5 Hz):
         "t", "hr_ecg", and every input channel.
future:  dict of 1-D arrays over the next 120 s (60 samples):
         "t" and every input channel. No heart rate.

Load any fitted artifacts relative to this file; only the submission/
folder is available when the evaluator runs your code.
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import kinetic  # noqa: E402

_P = json.loads((HERE / "params.json").read_text())
PARAMS = np.array(_P["params"])


def predict(history: dict, future: dict) -> np.ndarray:
    x = np.stack([np.asarray(future[f], float) for f in kinetic.FEATURES], axis=-1)[None]
    hr0 = np.array([float(history["hr_ecg"][-1])])
    return kinetic.simulate(PARAMS, hr0, x)[0]
