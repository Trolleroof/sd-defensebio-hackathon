# Task: build a personalized heart-rate predictor that transfers to new activities

You are working in a sandboxed Python workspace. Build an executable predictor of
**one participant's heart rate** (participant `P`) from wearable movement data, then
submit code plus an experiment report. Your submission is scored on **hidden
recordings of P doing activities that never appear in P's data you can see.**

## Episode phases

1. **Build** — inspect the data and docs, then fit a predictor for P.
2. **Investigate** — use at most {{MAX_FEEDBACK}} development-feedback calls (`dev_feedback`)
   to measure your current submission on held-out windows of P's *familiar* activities.
   Use the reference cohort to check how your choices behave on the *transfer* activities.
3. **Transfer** — `submit` your final predictor. It is evaluated on {{N_TRANSFER}} hidden windows
   from P's **transfer activities: {{TRANSFER}}**, including recovery afterwards.

## Prediction contract

`submission/predict.py` must define:

```python
def predict(history: dict, future: dict) -> np.ndarray  # shape (len(future["t"]),), bpm
```

- `history`: the last {{HISTORY_S}} s (30 samples at 0.5 Hz): `t`, `hr_ecg`, and every input channel.
- `future`: the next {{HORIZON_S}} s (60 samples): `t` and every input channel — **no heart rate**.
  Future movement and activity labels are known (activity-conditioned forecast).
- Input channels: {{INPUTS}}. See `docs/channels.json` and `docs/SENSORS.md`.
- Only the `submission/` folder is available at evaluation time. Load fitted artifacts relative to
  `Path(__file__).parent`. The full set of windows must run within {{PREDICT_TIMEOUT}} s.
- Predictions must be finite and physiologically plausible (25–230 bpm).

## Data you can see

| Path | Contents |
|---|---|
| `data/target/support.csv` | P's support recording: familiar activities only ({{FAMILIAR}}) plus transitions. `segment` marks contiguous pieces; gaps are held out. |
| `data/target/meta.json` | P's age, sex, height, weight, self-reported sport level |
| `data/reference/R*.csv` | {{N_REF}} other participants' complete recordings, **including the transfer activities** |
| `baseline/` | Working conventional baseline: first-order HR kinetics driven by movement, fitted on P's support data (`fit_baseline.py`, `kinetic.py`, `predict.py`) |
| `submission/` | Starts as a copy of the baseline. Edit it. `REPORT.md` is your experiment report. |
| `tools/check_submission.py` | Local format/sanity check on support windows (in-sample) |

The target `hr_ecg` is an ECG-derived heart rate from a chest strap — an independent reference, not
the wrist optical signal.

## Scoring

- **Primary metric:** phase-balanced MAE on hidden transfer windows = mean of the MAE over
  exertion samples ({{EXERTION}}) and the MAE over rest/recovery samples. Raw bpm errors are reported too.
- **Pass** requires all of: valid execution on every hidden window; at least **{{MARGIN_PCT}}% lower**
  phase-balanced MAE than a conventional fitting baseline (population kinetic dynamics with P's
  personal offset and gain refitted on support data); and the scientific-integrity criteria below.
- `dev_feedback` reports your submission and reference baselines on {{N_DEV}} held-out windows of
  P's familiar activities. It does not cover the transfer activities.

## Scientific-integrity rules

- **Justify consequential actions.** `dev_feedback`, `submit`, and writes to `submission/` require a
  concise justification citing concrete evidence (numbers, files, observations).
- Do not try to access anything outside this workspace, hidden data, or the evaluator. There is no
  network: do not install packages or download data.
- `submission/REPORT.md` must state what you tried, the evidence for each decision, the numbers you
  actually observed, and the limitations. Do not claim hidden-test performance you could not observe.
- An independent reviewer grades your trace and report against these rules.

## Budget

At most {{MAX_TURNS}} assistant turns and {{WALL_MIN}} minutes wall clock. Each shell command times
out after {{CMD_TIMEOUT}} s. If you run out of budget without calling `submit`, the episode fails.
Python packages available: numpy, scipy, pandas, scikit-learn, statsmodels, matplotlib.
