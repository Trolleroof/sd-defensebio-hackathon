# PhysBench — agent environment for personalized physiology predictors (MVP)

An LLM agent gets a sandboxed Python workspace with public wearable recordings, sensor docs, a
working baseline and fixed budgets. It must inspect the data, build an executable personalized
heart-rate predictor, use limited development feedback, and submit code plus an experiment report.
The submission is scored on hidden recordings; an LLM judge grades its decisions separately.

First task: **heart rate from movement/activity, with activity transfer**, on PPG-DaLiA
(15 participants, ECG-derived heart-rate reference, wrist + chest accelerometers, labelled activities).

## One episode: Build → Investigate → Transfer

| Phase | What the agent has | What it does |
|---|---|---|
| Build | Target participant's support data (familiar activities: sitting, driving, lunch, walking, working), 4 reference participants with the transfer block removed, a fitted conventional baseline | Inspect, fit a predictor |
| Investigate | ≤3 `dev_feedback` calls on held-out familiar-activity windows of the target. Reference files do not contain the transfer activities | Revise with evidence |
| Transfer | `submit` | Hidden scoring on the target's stairs, table soccer and cycling segments plus recovery — activities absent from its support data |

Prediction contract: `predict(history, future) -> (60,)` bpm. History = last 60 s with heart rate and
inputs; future = next 120 s of inputs only (activity-conditioned forecast). Consequential actions
(`dev_feedback`, `submit`, writes under `submission/`) are rejected without a ≥40-character justification.

## Scoring

- **Primary:** phase-balanced MAE (mean of exertion-phase and rest/recovery-phase MAE) on hidden transfer windows; raw bpm errors and per-activity / per-horizon breakdowns are kept.
- **Pass (per episode)** = valid execution **and** ≥ margin improvement over the conventional anchor
  (`kinetic_pop_calibrated`: population HR kinetics with personal offset/gain fitted on support) **and**
  integrity: explicit submit with a report, justifications present, no forbidden-access attempts, judge
  finds no leakage/gaming or fabricated claims, judge report-consistency ≥ 1.
- Criteria are frozen into `configs/frozen/pass_criteria_<version>.yaml` (with hashes and the dev
  evidence) before any held-out run; `physbench run --split heldout` refuses to run without them.
- **Judge:** fixed 7-item rubric (0–2 each) + 2 integrity flags, blind to model identity and hidden accuracy.
- **Pilot report:** Pass@1 (Wilson CI), transfer MAE and improvement (bootstrap CI over cases), judge
  score, runtime, cost, paired model differences, and a predeclared too-easy / too-hard / discriminative diagnosis.

## Participant roles

`configs/task_hr_transfer.yaml`: reference cohort S1, S4, S8, S12 (visible to agents, pseudonymised);
development S2, S3, S7, S10 (calibration and criteria freezing); held-out S5, S9, S11, S13, S14, S15;
S6 excluded (truncated recording). S3 was originally held-out and moved to development after its
baseline numbers were viewed during calibration.

## Run it

On a Linux host as root (RunPod pod used here):

```bash
uv venv --python 3.12 .venv && uv pip install -e .
# download PPG-DaLiA (UCI id 495, CC BY 4.0) and unzip into data/raw/x/  (2.7 GB zip)
physbench prepare                                  # raw pickles -> data/processed/dalia/*.npz (safe unpickler)
physbench setup-sandbox                            # agent venv, permissions, network guard
python tests/smoke_episode.py                      # plumbing + isolation checks, no LLM
echo "OPENROUTER_API_KEY=..." > .env && chmod 600 .env
physbench run --split dev --run-id dev-0 --workers 12 --provisional
physbench freeze --dev-run dev-0 --margin 0.10 --note "why this margin"
physbench run --split heldout --run-id pilot-0 --workers 18      # the reproducible evaluation command
physbench report --run-id pilot-0                  # regenerate runs/pilot-0/summary.md
```

Outputs per episode in `runs/<run-id>/<model>-<subject>-r<k>/`: `trace.jsonl` (every model turn, tool
call, result, tokens and cost), `submission/` (final code + REPORT.md), `judge.json`, `result.json`
(metrics, baselines, criteria). Run-level: `summary.md`, `summary.json`, `results.json`, `baselines.json`,
`run_meta.json` (config + criteria hashes, git commit). Hidden labels and predictions stay in root-only `private/`.

## Make it configurable / extend

- **Channels and targets:** `input_channels`, `target_channel` in the task YAML. The PPG-DaLiA adapter
  already produces `eda_wrist`, `temp_wrist`, `resp_chest_std`; enabling them is a config change. A new
  device needs one adapter that writes the common recording format (`physbench/data/dalia.py` shows it).
- **Difficulty:** support/transfer activity lists, history/horizon, feedback budget, turns, wall clock.
- **Models:** `configs/pilot-0.yaml` (any OpenRouter model with tool calling).

## Isolation (honest version)

Enforced: per-episode Unix users and 700 workspaces; repository, labels and API key root-only;
stripped environment; CPU/memory/file-size/time limits; leftover processes killed before scoring.
Best-effort (the container cannot create network namespaces): dead proxy env vars, curl/wget not
executable, and a Python startup hook that blocks non-loopback sockets. Commands that mention hidden
paths or the network are rejected before they run and fail the episode.

## Training interface (for Nikhil)

`result.json` + `trace.jsonl` provide machine-readable trajectories, dev-feedback results, execution
validity, baselines and judge scores per episode. Choice of checkpoint, data, method and reward mapping
is Nikhil's; no fine-tuning is done here.
