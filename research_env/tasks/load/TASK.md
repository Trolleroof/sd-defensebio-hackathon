# Load-carriage metabolic cost — retrospective research task v1

Build an executable model predicting gross metabolic power in watts during level walking at 4 km/h, from permitted personal measurements and carried load. Generalize to unseen people at the recorded loads. Design the mathematical structure, fitting procedure, uncertainty and experiments. Improve on the provided baselines under the resource budget and submit reproducible code and fitted parameters. Predictions outside the recorded speed, grade and load range are not part of this task.

## What you have

- `DATA_DICTIONARY.md`, `example-request.json` and `task.json` (tier, variant, budgets, frozen baseline loss).
- `data/train.json`: condition rows for the fit people only, with `gross_w` labels. People are opaque pseudonyms; keep each person's rows together when you validate (no random row splits).
- `data/development_requests.json`: inputs for the feedback people. Their labels are controller-owned; the `evaluate` action scores your `artifact/` on them.
- `baseline/baseline_model.py`: runnable reference baselines (mean-by-load, ridge with mass x load interactions, compact tree, Pandolf 1977 equation). `artifact/` initially holds the frozen ridge baseline. Reproduce with `python baseline/baseline_model.py data/train.json artifact ridge <tier>`.

Reward people are hidden and never enter the workspace.

## Tiers and variants (see `task.json`)

- `primary`: request `calibration` is `null`. Inputs are sex, age, stature, body mass, load, fixed speed and grade. All recorded loads are scored.
- `calibration`: request `calibration` reveals this person's measured unloaded (0 kg) gross watts. The 0 kg condition is then not scored. Calibration results must never be compared to primary results without disclosing the extra measurement.
- `person` variant: fit data contain every recorded load. `heldout15` variant: 15 kg rows are removed from the fit data and only 15 kg is scored on unseen people. This is extrapolation in load, separate from person transfer.

## Submission contract

`artifact/inference.py` implements `predict(artifact_dir, request)` and returns exactly `{"mean_w", "lower_w", "upper_w"}` — finite numbers with `lower_w <= mean_w <= upper_w`, in watts. Load fitted parameters from files in `artifact/` (JSON/NPZ; no fitting at inference). The worker has no network, sees only `artifact/` and one person's requests at a time, and is restarted for each person. Every eligible condition needs a prediction; a missing, extra, nonfinite or unordered output invalidates the whole attempt (loss null, never zero). Person IDs/pseudonyms are not features and are not in requests.

## Scoring

Primary loss: absolute watt error averaged over a person's scored loads, then averaged equally over people. Secondary: RMSE, signed bias, W/kg error, per-load MAE, worst-load MAE, within-person ordering concordance of condition costs (measured costs need not be monotonic), sex subgroup MAE, and 90% interval score `IS90 = (upper-lower) + 20*max(lower-y,0) + 20*max(y-upper,0)` with coverage and width reported separately. Wide intervals cannot win on coverage alone.

Units: targets are gross energy-expenditure *rates*. The source reports kcal/day; watts = kcal_per_day * 4184 / 86400. It is not the calories burned during the 11-minute bout. Net cost (gross minus resting metabolic rate) is a different target and is not scored here.

Limits: CPU only; budgets in `task.json` (fast profile: 20 min, 8 development calls; benchmark: 2 h, 30 calls, failed calls count). One fixed speed (1.111 m/s) and zero grade: this table cannot validate terrain, grade or speed claims. 48 source people; condition rows are repeated measures, not independent people.
