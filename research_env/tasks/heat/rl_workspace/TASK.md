# Task: forecast core temperature during exercise in the heat

Develop an executable model predicting reference core temperature 5, 10 and 20 minutes ahead from the permitted measurement history and known context. Return a point estimate and a 90% prediction interval in degrees Celsius for every eligible horizon. Choose your own representation, fitting procedure, architecture and experiments. Use only inputs available at the cutoff. Improve over the supplied baselines within the resource budget and submit a reproducible final artifact selected using public development evidence.

Tier for this task: **{tier}**. {tier_note}

## Data

Adults cycled at constant load for 60 minutes in a heat chamber (about 36 °C). The reference target is **rectal temperature**, measured independently each minute. This is retrospective laboratory data. It does not establish field performance or heat-injury prevention.

- `data/train_manifest.json`: {fit_n} training people with full input histories and rectal references. You may fit on these.
- `data/example_request.json`: one inference request exactly as your artifact will receive it.
- `DATA_DICTIONARY.md`: fields, units, timing and quality masks.
- `task_manifest.json`: horizons, metric, budget and profile.

{feedback_n} other people are held by the evaluator for development feedback. Each `evaluate` action scores your current `artifact/` on them. A further set of people, never shown, decides the final result.

## Your artifact

`artifact/` must contain `inference.py` defining:

```python
def predict(artifact_dir, request) -> list[dict]
```

It returns one record per requested horizon, in request order:

```json
[{{"horizon_min": 5, "mean_c": 37.9, "lower_c": 37.6, "upper_c": 38.2}}, ...]
```

`artifact/` is copied as-is into a fresh offline CPU worker, one worker per person; requests for one person arrive in increasing cutoff order, and each request carries only data stamped at or before its cutoff. Fit parameters inside your workspace and store them in `artifact/` (JSON or numpy `.npy`/`.npz`). Do not write to stdout from `predict`.

`tools/build_artifact.py MODEL_PY` packages a `model.py`-style module (`train(train_manifest, calibration_manifest, output_dir, seed)` and `predict(artifact_dir, request)`) into `artifact/` by fitting it on `data/train_manifest.json`. The staged `artifact/` already contains the elapsed-time baseline built this way.

Rules:

- Every requested horizon needs finite values with `lower_c <= mean_c <= upper_c`. A missing or malformed value makes the whole evaluation invalid; nothing is dropped.
- Python 3.13 with numpy, scipy and scikit-learn. No network.
- Do not depend on request IDs, person identifiers or request order across people.

## Baselines

`baselines/` holds four executable references: `elapsed_time.py` (population mean curve by minute, ignoring every measurement), `ridge_lag.py`, `gradient_boosting.py` and `thermal_state.py`. Constant-load cycling makes elapsed time a strong shortcut, so compare against it explicitly. The frozen reference for the final result is `elapsed_time`.

## Scoring

Primary loss: mean absolute error per horizon within each person, then the equal mean of the three horizons, then the equal mean over people, in °C. Also reported: per-horizon MAE/RMSE/bias, underprediction when reference ≥ 38.5 °C, and the 90% interval score `IS90 = width + 20·(miss below) + 20·(miss above)`, with coverage and width. Wide intervals cannot win by coverage alone.

## Budget

{wall_min} minutes wall clock, {dev_calls} development evaluations (failed calls count), up to 8 CPU cores, CPU only.

## Final submission

Submit the candidate you evaluated and stand behind, with a short evidence-based reason. Claims must match what development evidence established.
