"""Equal-budget conventional search: the control an agent must beat to show added value.

Samples configurations from a fixed space, trains each in the worker, spends
exactly one public development call per configuration (the same 30-call
budget a participant gets), and selects the lowest development MAE. Test
data are never consulted for selection.
"""
from __future__ import annotations

import random
import shutil
from pathlib import Path

from . import runner

FEATURE_POOL = ["mass", "stature", "age", "load", "mass_x_load", "load2_per_mass", "total_mass", "load_frac"]


def sample_config(rng: random.Random) -> dict:
    model = rng.choice(["ridge", "ridge", "gbm", "pandolf_recal"])
    if model == "ridge":
        feats = ["mass", "load"] + rng.sample([f for f in FEATURE_POOL if f not in ("mass", "load")],
                                              rng.randint(0, 4))
        return {"model": "ridge", "hyperparameters": {"features": feats, "alpha": rng.choice([0.01, 0.1, 1.0, 10.0, 100.0]),
                                                      "use_sex": rng.random() < 0.5}}
    if model == "gbm":
        return {"model": "gbm", "hyperparameters": {"n_estimators": rng.choice([50, 100, 200, 400]),
                                                    "learning_rate": rng.choice([0.02, 0.05, 0.1]),
                                                    "max_depth": rng.choice([1, 2, 3]),
                                                    "min_leaf": rng.choice([4, 6, 10]),
                                                    "use_sex": rng.random() < 0.5}}
    return {"model": "pandolf_recal", "hyperparameters": {}}


def run_search(prepared_dir, out_dir, track: str, seed: int = 0, budget: int | None = None) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    src = out / "src"
    src.mkdir(exist_ok=True)
    shutil.copyfile(Path(__file__).with_name("baseline_models.py"), src / "train.py")
    dev = runner.DevBudget(out / "dev", max_calls=budget)
    rng = random.Random(seed)
    trials = []
    while dev.used() < dev.max_calls:
        cfg = sample_config(rng)
        cid = f"search-{len(trials):02d}"
        art = out / "candidates" / cid
        res = runner.train_candidate(prepared_dir, src, track, seed, art, cfg)
        if not res["ok"]:  # failed training still costs a development call
            dev.call(prepared_dir, art if art.exists() else src, track, cid)
            trials.append({"candidate_id": cid, "config": cfg, "dev_mae_w": None, "error": res.get("error")})
            continue
        pub = dev.call(prepared_dir, art, track, cid)
        trials.append({"candidate_id": cid, "config": cfg,
                       "dev_mae_w": (pub.get("primary") or {}).get("value") if pub.get("execution_valid") else None})
    valid = [t for t in trials if t["dev_mae_w"] is not None]
    best = min(valid, key=lambda t: t["dev_mae_w"]) if valid else None
    return {"track": track, "seed": seed, "dev_calls_used": dev.used(), "trials": trials, "selected": best,
            "selected_artifact": str(out / "candidates" / best["candidate_id"]) if best else None}
