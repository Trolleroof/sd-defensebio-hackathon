"""PPG-DaLiA adapter: raw subject pickles -> common recording format.

Common recording format (one ``.npz`` per subject, 0.5 Hz grid):

- ``t``: seconds since recording start. Sample ``i`` sits at the END of
  the 8 s ECG window used for the reference heart rate, so every channel
  value at ``t[i]`` is computed only from data at or before ``t[i]``.
- one array per channel in :data:`CHANNELS` (NaN where unavailable).

The ECG-derived heart rate (``hr_ecg``) is the independent reference
supplied by the dataset authors. Wrist channels come from the Empatica E4,
chest channels from the RespiBAN.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .safe_pickle import load

STEP_S = 2.0
WINDOW_S = 8.0

ACTIVITY_NAMES = {
    0: "transient",
    1: "sitting",
    2: "stairs",
    3: "table_soccer",
    4: "cycling",
    5: "driving",
    6: "lunch",
    7: "walking",
    8: "working",
}

# name -> (unit, source device, description)
CHANNELS = {
    "hr_ecg": ("bpm", "RespiBAN chest ECG", "Reference heart rate from ECG R-peaks over the preceding 8 s (dataset label)"),
    "acc_wrist_enmo": ("g", "Empatica E4 wrist ACC 32 Hz", "Mean of max(|a|-1 g, 0) over the preceding 2 s"),
    "acc_wrist_std": ("g", "Empatica E4 wrist ACC 32 Hz", "Std of |a| over the preceding 2 s"),
    "acc_chest_enmo": ("g", "RespiBAN chest ACC 700 Hz", "Mean of max(|a|-1 g, 0) over the preceding 2 s"),
    "acc_chest_std": ("g", "RespiBAN chest ACC 700 Hz", "Std of |a| over the preceding 2 s"),
    "activity": ("id", "protocol annotation 4 Hz", "Most frequent activity id over the preceding 2 s (see ACTIVITY_NAMES)"),
    "eda_wrist": ("uS", "Empatica E4 wrist EDA 4 Hz", "Mean skin conductance over the preceding 2 s"),
    "temp_wrist": ("degC", "Empatica E4 wrist skin temperature 4 Hz", "Mean skin temperature over the preceding 2 s"),
    "resp_chest_std": ("a.u.", "RespiBAN chest respiration 700 Hz", "Std of respiration belt signal over the preceding 8 s"),
}


def _bin_reduce(x: np.ndarray, fs: float, t_end: np.ndarray, span_s: float, fn) -> np.ndarray:
    out = np.full(len(t_end), np.nan)
    for i, te in enumerate(t_end):
        a, b = int(round((te - span_s) * fs)), int(round(te * fs))
        a = max(a, 0)
        b = min(b, len(x))
        if b > a:
            out[i] = fn(x[a:b])
    return out


def _enmo(mag):
    return float(np.mean(np.maximum(mag - 1.0, 0.0)))


def _mode(a):
    vals, counts = np.unique(a.astype(int), return_counts=True)
    return float(vals[np.argmax(counts)])


def convert_subject(pkl: Path, out_dir: Path) -> dict:
    d = load(pkl)
    sid = str(d["subject"]).strip()
    hr = np.asarray(d["label"], dtype=float)
    t = WINDOW_S + STEP_S * np.arange(len(hr))

    wrist, chest = d["signal"]["wrist"], d["signal"]["chest"]
    wmag = np.linalg.norm(np.asarray(wrist["ACC"], float), axis=1)
    cmag = np.linalg.norm(np.asarray(chest["ACC"], float), axis=1)

    ch = {
        "hr_ecg": hr,
        "acc_wrist_enmo": _bin_reduce(wmag, 32, t, STEP_S, _enmo),
        "acc_wrist_std": _bin_reduce(wmag, 32, t, STEP_S, np.std),
        "acc_chest_enmo": _bin_reduce(cmag, 700, t, STEP_S, _enmo),
        "acc_chest_std": _bin_reduce(cmag, 700, t, STEP_S, np.std),
        "activity": _bin_reduce(np.asarray(d["activity"], float).ravel(), 4, t, STEP_S, _mode),
        "eda_wrist": _bin_reduce(np.asarray(wrist["EDA"], float).ravel(), 4, t, STEP_S, np.mean),
        "temp_wrist": _bin_reduce(np.asarray(wrist["TEMP"], float).ravel(), 4, t, STEP_S, np.mean),
        "resp_chest_std": _bin_reduce(np.asarray(chest["Resp"], float).ravel(), 700, t, WINDOW_S, np.std),
    }
    q = d.get("questionnaire", {})
    meta = {
        "subject": sid,
        "n_samples": int(len(t)),
        "duration_s": float(t[-1]),
        "age": int(q.get("AGE", -1)),
        "gender": str(q.get("Gender", "")).strip(),
        "height_cm": float(q.get("HEIGHT", np.nan)),
        "weight_kg": float(q.get("WEIGHT", np.nan)),
        "sport_level": int(q.get("SPORT", -1)),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / f"{sid}.npz", t=t, **ch)
    (out_dir / f"{sid}.json").write_text(json.dumps(meta, indent=2))
    return meta


def convert_all(raw_root: Path, out_dir: Path, workers: int = 15) -> list[dict]:
    pkls = sorted(raw_root.glob("S*/S*.pkl"), key=lambda p: int(p.stem[1:]))
    with ProcessPoolExecutor(max_workers=workers) as ex:
        metas = list(ex.map(convert_subject, pkls, [out_dir] * len(pkls)))
    manifest = {
        "dataset": "PPG-DaLiA (UCI id 495, CC BY 4.0)",
        "grid_step_s": STEP_S,
        "sample_time_convention": "t[i] is the end of the 8 s ECG window; features use only data at or before t[i]",
        "channels": {k: {"unit": u, "source": s, "description": desc} for k, (u, s, desc) in CHANNELS.items()},
        "activity_names": ACTIVITY_NAMES,
        "subjects": metas,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return metas


def load_recording(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}
