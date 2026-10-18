"""score(): authoritative numerical outcome for heat-core-forecast-v1.

Aggregation order: absolute error per (cutoff, horizon) target -> mean per
horizon within a person -> equal mean of the three horizons within a person ->
equal mean over people. One bout per person exists in this dataset, so the
bout and person levels coincide. Every requested horizon must have a finite,
ordered prediction; a missing or malformed value makes the whole run invalid
with null losses. Ineligible targets (evaluator-owned mask) are not scored but
still require a valid prediction.
"""
from __future__ import annotations

import math
import random

HOT_SLICE_C = 38.5        # frozen before comparison: reference rectal >= 38.5 °C at target minute
IS_ALPHA = 0.10
SKILL_EPSILON_C = 0.01


def interval_score(lower, upper, ref, alpha=IS_ALPHA):
    pen = 2.0 / alpha
    return (upper - lower) + pen * max(lower - ref, 0.0) + pen * max(ref - upper, 0.0)


def _finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def validate_predictions(requests_by_id: dict, predictions: dict) -> list[str]:
    """Return a list of problems; empty means every requested output is valid."""
    problems = []
    if not isinstance(predictions, dict):
        return ["predictions must be an object keyed by request_id"]
    extra = set(predictions) - set(requests_by_id)
    if extra:
        problems.append(f"{len(extra)} predictions for unknown request ids")
    for rid, req in requests_by_id.items():
        recs = predictions.get(rid)
        if not isinstance(recs, list):
            problems.append(f"{rid}: missing prediction list")
            continue
        hs = [r.get("horizon_min") if isinstance(r, dict) else None for r in recs]
        if hs != list(req["horizons_min"]):
            problems.append(f"{rid}: horizons {hs} != requested {req['horizons_min']}")
            continue
        for r in recs:
            vals = [r.get(k) for k in ("lower_c", "mean_c", "upper_c")]
            if not all(_finite(v) for v in vals):
                problems.append(f"{rid}/h{r['horizon_min']}: non-finite or missing value")
            elif not vals[0] <= vals[1] <= vals[2]:
                problems.append(f"{rid}/h{r['horizon_min']}: requires lower <= mean <= upper")
    return problems


def _mean(xs):
    return sum(xs) / len(xs) if xs else None


def score(reference_records, predictions, scoring_config) -> dict:
    """Common v1 interface.

    reference_records: evaluator targets, each with request_id, person,
        horizon_min, rectal_c, eligible.
    predictions: {request_id: [{horizon_min, mean_c, lower_c, upper_c}, ...]}
    scoring_config: {"requests": {request_id: request}, "baseline_loss_c": float|None,
                     "baseline_id": str|None}
    """
    requests_by_id = scoring_config["requests"]
    problems = validate_predictions(requests_by_id, predictions)
    n_targets = len(reference_records)
    base = {"metric_version": "heat-score-v1", "execution_valid": not problems,
            "requested_targets": n_targets,
            "eligible_targets": sum(1 for r in reference_records if r["eligible"])}
    if problems:
        return {**base, "primary_loss_c": None, "skill_vs_baseline": None,
                "failure_reason": "; ".join(problems[:20]) + (f" (+{len(problems) - 20} more)" if len(problems) > 20 else ""),
                "invalid_output_count": len(problems)}
    pred = {(rid, r["horizon_min"]): r for rid, recs in predictions.items() for r in recs}
    per = {}   # person -> horizon -> list of target dicts
    for t in reference_records:
        if not t["eligible"]:
            continue
        p = pred[(t["request_id"], t["horizon_min"])]
        err = p["mean_c"] - t["rectal_c"]
        per.setdefault(t["person"], {}).setdefault(t["horizon_min"], []).append({
            "err": err, "abs": abs(err), "ref": t["rectal_c"],
            "is": interval_score(p["lower_c"], p["upper_c"], t["rectal_c"]),
            "covered": p["lower_c"] <= t["rectal_c"] <= p["upper_c"],
            "width": p["upper_c"] - p["lower_c"], "under": max(t["rectal_c"] - p["mean_c"], 0.0)})
    horizons = sorted({t["horizon_min"] for t in reference_records})
    people = sorted(per)
    person_rows = {}
    for person in people:
        hrow = {}
        for h in horizons:
            xs = per[person].get(h, [])
            if not xs:
                continue
            hrow[str(h)] = {"n": len(xs), "mae_c": _mean([x["abs"] for x in xs]),
                            "rmse_c": math.sqrt(_mean([x["err"] ** 2 for x in xs])),
                            "bias_c": _mean([x["err"] for x in xs]),
                            "is90": _mean([x["is"] for x in xs]),
                            "coverage90": _mean([x["covered"] for x in xs]),
                            "width90_c": _mean([x["width"] for x in xs])}
        allx = [x for h in horizons for x in per[person].get(h, [])]
        hot = [x for x in allx if x["ref"] >= HOT_SLICE_C]
        person_rows[person] = {
            "horizons": hrow,
            "loss_c": _mean([v["mae_c"] for v in hrow.values()]),
            "horizons_scored": len(hrow),
            "hot_n": len(hot),
            "hot_underprediction_c": _mean([x["under"] for x in hot]),
            "hot_mae_c": _mean([x["abs"] for x in hot]),
        }
    by_h = {}
    for h in horizons:
        rows = [person_rows[p]["horizons"][str(h)] for p in people if str(h) in person_rows[p]["horizons"]]
        by_h[str(h)] = {k: _mean([r[k] for r in rows]) for k in
                        ("mae_c", "rmse_c", "bias_c", "is90", "coverage90", "width90_c")}
        by_h[str(h)].update({"people": len(rows), "targets": sum(r["n"] for r in rows)})
    hot_people = [p for p in people if person_rows[p]["hot_n"]]
    primary = _mean([person_rows[p]["loss_c"] for p in people])
    bl = scoring_config.get("baseline_loss_c")
    return {
        **base,
        "failure_reason": None,
        "primary_loss_c": primary,
        "primary_definition": "mean over people of (mean over horizons 5/10/20 of per-person MAE), °C",
        "skill_vs_baseline": None if bl is None else 1.0 - primary / max(bl, SKILL_EPSILON_C),
        "skill_baseline_id": scoring_config.get("baseline_id"),
        "people": len(people),
        "by_horizon": by_h,
        "uncertainty": {k: _mean([by_h[h][k] for h in by_h]) for k in ("is90", "coverage90", "width90_c")},
        "hottest_period": {"definition": f"targets with reference rectal >= {HOT_SLICE_C} °C",
                           "people_with_samples": len(hot_people),
                           "targets": sum(person_rows[p]["hot_n"] for p in people),
                           "underprediction_c": _mean([person_rows[p]["hot_underprediction_c"] for p in hot_people]),
                           "mae_c": _mean([person_rows[p]["hot_mae_c"] for p in hot_people])},
        "per_person": person_rows,
    }


def paired_bootstrap(per_person_a: dict, per_person_b: dict, n: int = 2000, seed: int = 0) -> dict:
    """Bootstrap the person-balanced loss difference (a - b) by resampling people."""
    people = sorted(set(per_person_a) & set(per_person_b))
    diffs = [per_person_a[p]["loss_c"] - per_person_b[p]["loss_c"] for p in people]
    rng = random.Random(seed)
    stats = sorted(_mean([diffs[rng.randrange(len(diffs))] for _ in diffs]) for _ in range(n))
    return {"people": len(people), "mean_diff_c": _mean(diffs),
            "ci95_c": [stats[int(0.025 * n)], stats[int(0.975 * n) - 1]],
            "fraction_a_better": _mean([d < 0 for d in diffs]), "resamples": n, "seed": seed}
