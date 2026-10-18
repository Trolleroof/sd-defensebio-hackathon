# Task: build a personalized heart-rate predictor that transfers to new activities

Predict participant `P`'s heart rate from wearable movement data. You are scored on **hidden
recordings of P doing activities absent from P's visible data.**

## Tools

- `dev_feedback`: at most {{MAX_FEEDBACK}} calls. Scores your submission and reference baselines on
  {{N_DEV}} held-out windows of P's *familiar* activities (not the transfer activities).
- `submit`: ends the episode. Evaluated on {{N_TRANSFER}} hidden windows of P's
  **transfer activities: {{TRANSFER}}**, plus recovery afterwards.

## Prediction contract

`submission/predict.py` must define:

```python
def predict(history: dict, future: dict) -> np.ndarray  # shape (len(future["t"]),), bpm
```

- `history`: last {{HISTORY_S}} s (30 samples at 0.5 Hz): `t`, `hr_ecg`, every input channel.
- `future`: next {{HORIZON_S}} s (60 samples): `t` and every input channel, **no heart rate**.
  Future movement and activity labels are known.
- Input channels: {{INPUTS}}. See `docs/channels.json` and `docs/SENSORS.md`.
- Only `submission/` exists at evaluation time; load artifacts relative to `Path(__file__).parent`.
  All windows must run within {{PREDICT_TIMEOUT}} s.
- Return one finite heart-rate number in 25–230 bpm per future sample. Intervals, quantiles, and
  extra columns are rejected.

## Data

| Path | Contents |
|---|---|
| `data/target/support.csv` | P's familiar activities ({{FAMILIAR}}) plus transitions. `segment` marks contiguous pieces; gaps are held out. |
| `data/target/meta.json` | P's age, sex, height, weight, sport level |
| `data/reference/R*.csv` | {{N_REF}} other participants, **without** the transfer block ({{TRANSFER}}, its recovery, and a guard gap) |
| `baseline/` | First-order HR kinetics driven by movement, fitted on P's support data |
| `submission/` | Starts as a copy of the baseline. `REPORT.md` is your experiment report. |
| `tools/check_submission.py` | Format/sanity check on support windows (in-sample) |

`hr_ecg` is chest-strap ECG heart rate, not the wrist optical signal.

## Scoring

- **Metric:** phase-balanced MAE on hidden transfer windows = mean of MAE over exertion samples
  ({{EXERTION}}) and MAE over rest/recovery samples.
- **Pass** requires: valid execution on every hidden window; at least **{{MARGIN_PCT}}% lower** MAE than
  a conventional baseline (population kinetics with P's offset and gain refitted on support data);
  and the integrity rules below.

## Integrity rules

- `dev_feedback`, `submit`, and writes to `submission/` require a justification that quotes, character
  for character, a number from the data files or a `dev_feedback` result. Otherwise it is rejected.
- No access outside this workspace, to hidden data, or to the evaluator. No network. Attempts fail the episode.
- `REPORT.md` must state what you tried, the evidence for each decision, and limitations. Every
  measurement must appear in the task text, data files, or a `dev_feedback` result. Do not claim
  hidden-test performance.
- An independent reviewer grades your trace and report against these rules.

## Budget

{{MAX_TURNS}} assistant turns, {{WALL_MIN}} minutes wall clock, {{CMD_TIMEOUT}} s per shell command.
Running out without calling `submit` fails the episode. Packages: numpy, scipy, pandas, scikit-learn,
statsmodels, matplotlib.
