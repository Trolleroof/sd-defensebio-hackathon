# Task: load-carriage metabolic power ({track} track)

Build an executable model predicting gross metabolic power in watts during level
walking at 4 km/h, from permitted personal measurements and carried load.
Generalize to unseen people at the recorded loads. Design the mathematical
structure, fitting procedure, uncertainty and experiments. Improve on the provided
baselines under the resource budget and submit reproducible code and fitted
parameters. Predictions outside the recorded speed, grade and load range are not
part of this task.

{track_note}

## What you have

- `data/train.jsonl` — training people only, one row per person-condition. See `DATA_DICTIONARY.md`.
- `examples/request.json`, `examples/prediction.json` — the exact inference contract.
- `baseline/train.py` — a runnable baseline (`mean_by_load`, `ridge`, `gbm`, `pandolf`, `pandolf_recal`{cal_model}).
- Public development feedback: {dev_calls} calls, scored on 10 development people you never see labels for.
  Each call returns aggregate and per-load metrics only. Failed calls count.

## Submission contract

`submission/train.py` defines

```python
def train(train_manifest, calibration_manifest, output_dir, seed) -> dict
```

`train_manifest["records_path"]` points at a copy of `data/train.jsonl`; `calibration_manifest` is `None`.
It must write an artifact directory containing `predict.py` with

```python
def predict(artifact_dir, request) -> {{"mean_w": float, "lower_w": float, "upper_w": float}}
```

`lower_w <= mean_w <= upper_w`, finite, 90% prediction interval. Every eligible request needs a valid
prediction; one invalid or missing prediction makes the whole run invalid (no partial credit).
Training and inference run offline, CPU only, in a fresh sandboxed process with no network and no access
to files outside the copied inputs. Inference starts a fresh process per held-out person.
Store fitted parameters as plain files (JSON/NPY); pin any dependency beyond numpy in `requirements.lock`.

Also submit: seeds, the exact reproduction command, the selected candidate ID and a short evidence report
(what you tried, development results, what you conclude and what this data cannot establish).

## Scoring (numerical, authoritative)

- Primary: absolute watt error averaged over each person's scored loads, then averaged over people equally.
- Secondary: W/kg error, signed bias, per-load error, worst-load error, within-person ordering of measured
  costs (secondary; measured costs are not always monotonic in load), and error by sex.
- Uncertainty: IS90 = (upper - lower) + 20·max(lower - y, 0) + 20·max(y - upper, 0), coverage and width.
  Wide intervals cannot win by coverage alone.

## Resources

{budget}

## Rules

- Inputs are the request fields only. The person grouping key in training rows is for grouped validation,
  never a feature. No other people's labels at inference.
- Target unit: gross power in W, converted from source kcal/day rate values by `W = kcal/day × 4184 / 86400`.
  It is a rate, not the energy of the 11-minute bout. RMR has not been subtracted.
- 48 people and at most 4 conditions each: the condition rows are not independent people.
