"""Runs a submission on input windows. Executed as the sandboxed agent user.

    python eval_runner.py <submission_dir> <windows.npz> <out.npy>

The windows file holds inputs only (history with target, future covariates).
"""

import importlib.util
import json
import sys
import time
import traceback

import numpy as np


def main():
    sub_dir, win_path, out_path = sys.argv[1:4]
    z = np.load(win_path, allow_pickle=False)
    hist_keys = [k[5:] for k in z.files if k.startswith("hist_")]
    fut_keys = [k[4:] for k in z.files if k.startswith("fut_")]
    n, f = z["fut_t"].shape
    t0 = time.time()
    spec = importlib.util.spec_from_file_location("submission_predict", f"{sub_dir}/predict.py")
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, sub_dir)
    spec.loader.exec_module(mod)
    hist_all = {k: z[f"hist_{k}"] for k in hist_keys}
    fut_all = {k: z[f"fut_{k}"] for k in fut_keys}
    out = np.full((n, f), np.nan)
    for i in range(n):
        h = {k: v[i].copy() for k, v in hist_all.items()}
        fu = {k: v[i].copy() for k, v in fut_all.items()}
        p = np.asarray(mod.predict(h, fu), dtype=float).reshape(-1)
        if p.shape != (f,):
            raise ValueError(f"window {i}: predict returned shape {p.shape}, expected ({f},)")
        out[i] = p
    np.save(out_path, out)
    print(json.dumps({"ok": True, "seconds": round(time.time() - t0, 3), "n": int(n)}))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print(json.dumps({"ok": False, "error": traceback.format_exc()[-3000:]}))
        sys.exit(1)
