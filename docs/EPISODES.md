# Workspace and episode specification v0

This implements the first benchmark slice from PLAN.md without changing the sensor/product plan. Status: a local trusted-controller prototype with executable predictors, synthetic examples and a model-independent action protocol. Scientific calibration and hardened execution remain open.

## Walkthroughs considered before implementation

1. **Build:** one complete calibration recording is visible. A development request and a final request each expose only history through t=0, forecast timestamps 30/60/90 seconds, and a declared future rest condition. The agent starts with an unimplemented predictor interface.
2. **Investigate:** persistence produces poor development error. The researcher records a hypothesis, requests a listed longer-bout recording for one credit, revises and evaluates again. Reopening the controller preserves spent budget. The toy script uses a known linear decline to make expected scores hand-checkable.
3. **Invalid action:** hidden recordings cannot be acquired; duplicate acquisitions are rejected; invalid predictions consume an evaluation call; submission terminates the episode even when predictions are invalid. Later development feedback is rejected.

## Episode contract

The trusted episode includes schema_version=1, id, origin=synthetic/public/local, budgets (acquisition and evaluations), and sessions. Each session has a unique ID, participant_id, role (support/acquisition/development/test), description, cost and timestamped rows. Each row contains time_s and hr_bpm. The current fixture uses one participant, complete scalar measurements and a cutoff at zero. A canonical CSV adapter supports supplied real recordings with explicit provenance. Missingness, multi-participant aggregation and alternate targets remain subsequent work.

Support and acquired sessions are complete recordings. Development/test requests contain history through cutoff and future timestamps, never future measurements. Initial budgets are one acquisition credit and two development calls. The acquisition menu is fixed to recordings that exist. Invalid development forecasts consume a call; unavailable/duplicate acquisition requests are rejected without charge.

Forecast rows contain exactly time_s, mean, lower and upper. All required times must be present in order; values must be finite and lower <= mean <= upper. Intervals are nominal 90%. Scoring reports MAE, empirical coverage and interval score with alpha=0.1. The toy profile is recovery-only. Invalid runs have valid=false and an error, not a favorable numeric score. Cross-run ranking/failure aggregation is not implemented.

## Workspace layout

`agent/brief.json` contains the task, budgets and acquisition menu. `support.json` contains calibration evidence. `requests.json` contains bounded development and final requests. `predictor.py` is blank. Acquired recordings and public action history are added as operations complete.

The separate `private/` directory contains the full episode, persistent controller state, frozen submission and final score. Final submission returns only a receipt and code hash to the workspace. The demonstration CLI prints private scores for the human developer; it must never be exposed as an agent tool.

Every consequential operation requires evidence, hypothesis, action and expected_result strings before the operation begins. The controller preserves these plus outcomes. This records stated intent and external behavior, not private reasoning. The trace-judge module exports isolated, chronology-bounded action packets and validates external structured verdicts. Reports mark external judging pending until verdicts are supplied; deterministic audit is a separate panel.

## Run the examples

From the repository root, with a fresh output directory:

```sh
python3 -m benchmark demo .benchmark-runs/physiology-example
python3 -m unittest discover -s tests -v
```

The reproducible demo is scripted. A separately saved Luna pilot exercises model-written code and feedback, with controller-mediated file exchanges; it is not a measurement of a hosted model API or automated deployment. It produces 20 bpm development MAE for persistence and zero for the linear revision; the held-out synthetic recording uses the same simple family. These are fixture checks, not research results or proof of generalization.

## Execution boundary and limitations

The `evaluate_code` and `submit_code` paths snapshot and hash predictor source, re-read and verify it, then execute fit(support) and predict(history, forecast_times_s). Only allowed support/acquisitions and cutoff-bounded requests enter that execution. Final scores stay in the trusted control directory; the agent receives a receipt. The older forecast-only evaluate/submit API remains for fixture compatibility and explicitly marks code execution unverified.

The macOS executor uses sandbox-exec to deny network, forks, reads under user/home/temp roots and controller-supplied private paths, and writes outside its temporary execution directory. Successful execution plus file/network denial and timeout probes run in tests. Its Python startup still requires broad reads elsewhere on the host, and process execution is allowed. Memory limits via RLIMIT_DATA/RSS are best effort. This is a prototype boundary, not a hardened multi-tenant executor or full filesystem allowlist. Unsupported hosts fail closed. Use a hardened container/VM backend before running adversarial campaigns; Docker was unavailable during this build.

The controller is single-writer. Atomic state replacement, interrupted-event failure recovery and candidate hash checks are implemented; concurrent writes and adversarial edits by the controller's host user are unsupported. The agent action cap, evaluation credits and acquisitions persist across reopens. Raw actions and episodes have byte limits. A host-user participant can bypass a file-only experiment policy, so the Luna pilot's instruction-limited observation access is a smoke test, not a secure contamination-resistance claim.

## Next research gates

Audit source-level recording identity and activity alignment for multiple usable recordings before building real-data episodes. The adapter does not prove non-overlap of windows renamed with different session IDs. Keep real and synthetic panels separate. Calibrate uncertainty and task difficulty using development data, freeze session/participant splits and model/judge configurations, then compare competitors on untouched cases. Transfer and Sustain need real supported condition changes and repeated-session sequences.

External judge packets and verdict import work, but no calibrated independent LLM judging run or expert audit has been completed. Token/API cost accounting, automatic remote-model configuration and post-training are not implemented. The agent provider interface deliberately leaves inference settings and billing to its caller.
