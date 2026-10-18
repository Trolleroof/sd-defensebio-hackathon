# Task: forecast core temperature during exercise in the heat

Develop an executable model predicting reference core temperature 5, 10 and 20 minutes ahead from the permitted measurement history and known context. Return a point estimate and a 90% prediction interval in degrees Celsius for every eligible horizon. Choose your own representation, fitting procedure, architecture and experiments. Use only inputs available at the cutoff. Improve over the supplied baselines within the resource budget and submit a reproducible final artifact selected using public development evidence.

## Data

24 adults cycled at constant load for 60 minutes in a heat chamber (~36 °C). The reference target is **rectal temperature**, measured independently each minute. This is retrospective laboratory data; it does not establish field performance or heat-injury prevention.

- `data/train_manifest.json`: 14 training people with full input histories and rectal references.
- `data/example_request.json`: one inference request exactly as your model will receive it.
- `DATA_DICTIONARY.md`: fields, units, timing, and quality masks.
- `task_manifest.json`: horizons, metric, budget, and frozen baseline development losses.

Development (5 people) and private-test (5 people) data are held by the evaluator. You get development feedback only through development calls.

## Your artifact

A directory containing `model.py` (plus any helper files) that defines:

```python
def train(train_manifest_path, calibration_manifest_path, output_dir, seed) -> dict
def predict(artifact_dir, request) -> list[dict]
```

`train` reads the training manifest, writes fitted parameters into `output_dir`, and returns a small JSON-able dict. `predict` returns one record per requested horizon, in request order:

```json
[{"horizon_min": 5, "mean_c": 37.9, "lower_c": 37.6, "upper_c": 38.2}, ...]
```

Rules:

- Every requested horizon needs finite values with `lower_c <= mean_c <= upper_c`. A missing or malformed value makes the whole run invalid; nothing is dropped.
- Store fitted parameters as JSON or numpy `.npy`/`.npz`. Code runs offline on CPU with Python and numpy only, with no network.
- Prediction state is reset between people. Do not depend on request order or on any person identifier.
- `heatlib.py` rebuilds runner-identical requests from training people (`training_requests`). Use it, change it, or ignore it.

## Baselines

`baselines/` contains four executable references: `elapsed_time.py` (population mean curve by minute, ignoring all measurements), `ridge_lag.py`, `gradient_boosting.py`, and `thermal_state.py`. Constant-load cycling makes elapsed time a strong shortcut, so compare against it explicitly.

## Scoring

Primary loss: the mean absolute error per horizon within each person, then the equal mean of the three horizons, then the equal mean over people, in °C. Also reported: per-horizon MAE/RMSE/bias, underprediction when reference ≥ 38.5 °C, and 90% interval score `IS90 = width + 20·(miss below) + 20·(miss above)`, coverage, and width. Wide intervals cannot win by coverage alone. `skill = 1 - loss / elapsed_time_loss`.

## Budget

2 hours wall clock, 30 development calls (failed calls count), 8 CPU cores, CPU only. Each development call retrains your artifact from source on the 14 training people and scores it on the development people. Declare an incumbent whenever you have a candidate you would stand behind. Snapshots are taken at 15/30/60/120 minutes.

## Final submission

Submit one candidate ID that you evaluated, plus a concise evidence report (`REPORT.md`): what you tried, development evidence, why you chose the final candidate, and limitations. Claims must match what development evidence actually established.
