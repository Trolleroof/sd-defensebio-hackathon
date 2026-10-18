"""Pure scoring for the load-carriage task (shared ``score`` interface).

Order of averaging: error within a person over that person's eligible scored
loads, then people equally. A run with any missing or invalid prediction for
an eligible target is ``execution_valid: false`` with null losses; it is never
scored on the subset that happened to be valid.
"""
from __future__ import annotations

import math
import random

from .schema import validate_prediction

INTERVAL_PENALTY = 20.0  # 2 / alpha for a 90% interval


def interval_score_90(lower: float, upper: float, reference: float) -> float:
    return ((upper - lower)
            + INTERVAL_PENALTY * max(lower - reference, 0.0)
            + INTERVAL_PENALTY * max(reference - upper, 0.0))


def _mean(values):
    values = list(values)
    return sum(values) / len(values) if values else None


def _ordering(rows) -> float | None:
    """Fraction of within-person load pairs (distinct references) ordered correctly; tied predictions count 0.5."""
    pairs = total = 0.0
    for i, a in enumerate(rows):
        for b in rows[i + 1:]:
            if a["reference_w"] == b["reference_w"]:
                continue
            pairs += 1
            ref = a["reference_w"] - b["reference_w"]
            pred = a["mean_w"] - b["mean_w"]
            total += 0.5 if pred == 0 else float((ref > 0) == (pred > 0))
    return total / pairs if pairs else None


def invalid_result(reason: str, n_targets: int, n_failed: int = 0) -> dict:
    return {"execution_valid": False, "failure_reason": reason, "n_targets": n_targets,
            "n_invalid_predictions": n_failed, "primary": {"name": "person_balanced_mae_w", "value": None},
            "skill": None, "secondary": None, "uncertainty": None, "per_load": None, "per_person": None}


def score(reference_records: list[dict], predictions: dict, scoring_config: dict) -> dict:
    score_loads = set(float(x) for x in scoring_config["score_loads"])
    targets = [r for r in reference_records if r["eligible"] and r["load_kg"] in score_loads]
    if not targets:
        raise ValueError("no eligible targets to score")
    if not isinstance(predictions, dict):
        return invalid_result("predictions must map record_id to prediction", len(targets), len(targets))
    failures = []
    rows = []
    for r in targets:
        pred = predictions.get(r["record_id"])
        reason = "missing prediction" if pred is None else validate_prediction(pred)
        if reason:
            failures.append(f"{r['record_id']}: {reason}")
            continue
        lo, mu, hi = float(pred["lower_w"]), float(pred["mean_w"]), float(pred["upper_w"])
        ref = r["reference_w"]
        rows.append({"person": r["source_person_id"], "sex": r["sex"], "load_kg": r["load_kg"],
                     "reference_w": ref, "mean_w": mu, "abs_w": abs(mu - ref), "signed_w": mu - ref,
                     "abs_w_per_kg": abs(mu - ref) / r["body_mass_kg"],
                     "is90": interval_score_90(lo, hi, ref), "covered": float(lo <= ref <= hi),
                     "width": hi - lo})
    if failures:
        result = invalid_result(f"{len(failures)} invalid/missing predictions; first: {failures[0]}",
                                len(targets), len(failures))
        result["failures"] = failures[:20]
        return result

    by_person: dict[str, list[dict]] = {}
    for row in rows:
        by_person.setdefault(row["person"], []).append(row)
    per_person = {}
    for person, prow in sorted(by_person.items()):
        per_person[person] = {
            "sex": prow[0]["sex"],
            "n_loads": len(prow),
            "mae_w": _mean(x["abs_w"] for x in prow),
            "bias_w": _mean(x["signed_w"] for x in prow),
            "mae_w_per_kg": _mean(x["abs_w_per_kg"] for x in prow),
            "is90": _mean(x["is90"] for x in prow),
            "coverage90": _mean(x["covered"] for x in prow),
            "width90_w": _mean(x["width"] for x in prow),
            "ordering": _ordering(prow),
        }
    people = list(per_person.values())
    per_load = {}
    for load in sorted(score_loads):
        lrows = [x for x in rows if x["load_kg"] == load]
        if lrows:  # one row per person per load, so this is already person-balanced
            per_load[f"{load:g}"] = {"n_people": len(lrows), "mae_w": _mean(x["abs_w"] for x in lrows),
                                     "bias_w": _mean(x["signed_w"] for x in lrows),
                                     "coverage90": _mean(x["covered"] for x in lrows)}
    primary = _mean(p["mae_w"] for p in people)
    ordering = [p["ordering"] for p in people if p["ordering"] is not None]
    by_sex = {}
    for sex in ("F", "M"):
        group = [p["mae_w"] for p in people if p["sex"] == sex]
        by_sex[sex] = {"n_people": len(group), "mae_w": _mean(group)}
    baseline = scoring_config.get("frozen_baseline_loss")
    eps = float(scoring_config.get("epsilon_w", 1.0))
    return {
        "execution_valid": True,
        "failure_reason": None,
        "n_targets": len(targets),
        "n_people": len(people),
        "n_invalid_predictions": 0,
        "primary": {"name": "person_balanced_mae_w", "value": primary, "unit": "W"},
        "skill": None if baseline is None else 1.0 - primary / max(float(baseline), eps),
        "secondary": {
            "mae_w_per_kg": _mean(p["mae_w_per_kg"] for p in people),
            "bias_w": _mean(p["bias_w"] for p in people),
            "worst_load_mae_w": max(v["mae_w"] for v in per_load.values()),
            "ordering_concordance": _mean(ordering),
            "ordering_n_people": len(ordering),
            "mae_w_by_sex": by_sex,
        },
        "uncertainty": {
            "is90": _mean(p["is90"] for p in people),
            "coverage90": _mean(p["coverage90"] for p in people),
            "width90_w": _mean(p["width90_w"] for p in people),
        },
        "per_load": per_load,
        "per_person": per_person,
    }


def paired_person_bootstrap(per_person_a: dict, per_person_b: dict, key: str = "mae_w",
                            n_boot: int = 2000, seed: int = 0) -> dict:
    """Bootstrap mean(a - b) over people (not condition rows). Negative favors a."""
    people = sorted(set(per_person_a) & set(per_person_b))
    if set(per_person_a) != set(per_person_b):
        raise ValueError("paired comparison needs the same people")
    diffs = [per_person_a[p][key] - per_person_b[p][key] for p in people]
    rng = random.Random(seed)
    means = sorted(_mean(rng.choice(diffs) for _ in diffs) for _ in range(n_boot))
    lo, hi = means[int(0.025 * n_boot)], means[min(n_boot - 1, int(math.ceil(0.975 * n_boot)) - 1)]
    return {"n_people": len(people), "mean_diff": _mean(diffs), "ci95": [lo, hi],
            "fraction_people_a_better": _mean(float(d < 0) for d in diffs)}
