# Exact protocol: physiology-compact-coding-v1

## Evaluation unit and limits

An agent writes and selects a physiological **predictor**. The fine-tuned weights being evaluated belong to the coding agent; each episode separately produces its own predictor artifact. One model receives three tasks at agent seeds 0, 1 and 2: **nine episodes per model**. The active roster has eight models and therefore 72 episodes. Seeds repeat the agent over the same biological people, rather than sampling independent cohorts.

Each episode has exactly two scheduled training-code phases and their development evaluations, followed by a final candidate selection if any candidate is valid. A native `respond` tool returns code/reason or candidate index/reason. Each phase permits at most two response attempts; invalid/missing/multiple tool calls consume attempts. No valid candidate, an invalid submission or exhausted response attempts is an explicit failed episode. Failed outputs remain in the run pool.

The controller clears `artifact/` before each code experiment, writes and runs `trial0.py` or `trial1.py`, captures training source, snapshots/evaluates the candidate, and submits the chosen snapshot. Models may use published baseline helpers or write their own preprocessing, features, estimator, fitting and intervals. This is a short coding protocol with helper examples, rather than a blank simulator or the older 30-candidate search protocol.

The actual participant context is the compact system message, task brief, helper example, file paths, trial seed and bounded summaries of prior development results, reasons and stderr. Full task/data dictionaries are available in the workspace; they are not automatically included in API context. Training programs can read public files. Command stdout is retained in traces but is not sent back in this compact observation. See `research_env/compact.py` and `protocol.json` for exact messages and tool schemas.

| Limit | Active value |
|---|---|
| Participant output, including reasoning | 1,536 tokens/request |
| Participant canonical request body | 4,096 UTF-8 bytes maximum |
| Participant tool reason | 1–160 characters |
| Training code payload | Up to 12,000 UTF-8 bytes, also constrained by response tokens |
| Judge output, including reasoning | 1,025 tokens/request |
| Judge canonical body | 3,072 UTF-8 bytes maximum |
| Response attempts per phase | 2 |
| Scheduled development evaluations | 2/episode |
| Session controller hard maxima | 7,200 elapsed seconds and 30 development calls |
| Predictor CPU allocation | 8 cores maximum |
| Active predictor memory allocation | 8,216,887,296 bytes, capped by frozen Docker allocation |
| Concurrent API episodes | 4 |
| Concurrent Docker workers | 1 |

The controller's legacy 30-call/two-hour maxima do not give the compact participant extra training phases. Predictor training/replay is offline and CPU-only. GPU model inference runs outside the predictor sandbox; inference time counts toward the episode wall limit. No GPU is exposed to generated predictor programs.

## Tasks and fixed people

| Task | Target | Train / development / test people | Primary loss |
|---|---|---|---|
| Pulse | Trailing 8-second heart rate, updated every 2 seconds from causal wrist PPG/acceleration | 13 / 5 / 5 | Mean window MAE within bout, bouts within person, then equal people; bpm |
| Heat | Independently measured rectal temperature at +5/+10/+20 minutes | 14 / 5 / 5 | Equal-horizon/person MAE; °C |
| Load | Gross metabolic power for 4 km/h, level walking at 0/5/10/15 kg load | 28 / 10 / 10 | Available loads within person, then equal people; W |

Pulse uses GalaxyPPG (23 usable people, 33,819 eligible windows). Its ECG-derived reference quality remains provisional. Heat uses the CORE cycling workbook (24 people, 740 records). Load uses EVOBREATH (48 people, 189 measured conditions); three missing 15-kg measurements are excluded rather than invented. Load references convert gross kcal/day equivalents using `W = kcal/day × 4184 / 86400`, without RMR subtraction. Source URLs/hashes and exact disjoint person lists are in `reference.json`.

Training labels are public. Development requests/signals are public; their labels stay on the trusted controller. Test recordings and labels remain controller-owned. No reference calibration, patient identity features, inference-time ECG/vendor HR, or future observations are introduced. Streaming state resets between people/bouts. Every required prediction must be finite, complete and ordered `lower <= mean <= upper`; a missing/invalid prediction invalidates the attempt. Artifacts implement `inference.py:predict(artifact_dir, request)` and include a dependency lock and fitted parameters. Worker state is at most 16 KiB. Full input/output contracts are in `research_env/tasks/*/TASK.md` and `DATA_DICTIONARY.md`.

## Outcomes and judge

Prediction Elo uses matched agent seeds, initial rating 1,000, K=32, scale=400, tolerance 1e-9 and deterministic match-order seed 0. Lower task loss wins. A valid task beats an invalid one; two invalid tasks skip. Suite matches require all three tasks valid; two complete suites compare equal task votes. An incomplete suite loses to a complete suite, while two incomplete suites skip. Zero valid complete suites means an unranked suite entry. Seeds do not create independent biological-cohort confidence intervals. Exact matches and raw losses remain available.

Uncertainty Elo uses the separately reported 90% interval score: `width + 20 max(lower−reference,0) + 20 max(reference−upper,0)`. It is separate from prediction Elo.

One blind GPT-6.1 Sol **Medium** episode review uses five criteria: evidence, experiment, feedback, reproducibility and selection. Each is 0–4 or null when unobservable, with genuine evidence IDs and reasons. The compact packet contains experiment decision excerpts, development validity/loss/failures and final selection evidence. It omits competitor identity and private test results. It does not contain full executable code. Scores therefore describe observable evidence under this compressed review and are not an independently calibrated scientific-quality measure. This differs from the older per-action 1–5 judge.

## Reproduction and known infrastructure exceptions

`reference.json` binds the actual source hashes, dataset trees, split metadata, ARM64 image ID, policy and measured allocation of the running sweep. Verify these before inference. Same data with a different architecture/image/allocation is a **replica pool** and needs a new suite freeze, paired base-model evaluation and an explicit label. Do not merge its Elo into the original pool.

Raw requests/responses, API usage/timing, source edits, command results, development choices, immutable candidate artifacts, final selections and hash-linked receipts/transcripts are retained. Native reasoning settings and provider/quantization choices differ across the frontier roster and are preserved in its exact configs.

The original shared API budget is $16. A provider reported 72,797 output tokens despite a 1,536-token request; the request cost remains in the ledger. The earlier $15.12 whole-run estimate assumed provider compliance and no longer guarantees completion. `scripts/resume_compact.py` reserves xAI's advertised maximum for accounting and rejects reported over-budget responses. `scripts/parallel_compact.py` shares the same transactional ledger across workers while serializing Docker. Both amendments and their implementation hashes are recorded in `reference.json`. Infrastructure/protocol failures must be retained and distinguished from a valid predictor's numerical loss.

These are retrospective fixed-cohort lab tasks with builder exposure. No RL training, weight update, clinical claim, field/combat validity, or general research-agent improvement follows from a run. Compare a fine-tuned agent against its unchanged base under the same artifact, token, trial, dataset and evaluator contracts.

## Public leaderboard reanalysis: pooled-task-elo-v1

After the run, the public site was revised at the user's request to score every
participant, including partial and failed runs. It pools all pairwise comparisons
within matched task/seed slots (pulse, heat, load; seeds 0, 1, 2), with the same
initial 1,000, K=32, scale=400, tolerance 1e-9, deterministic order seed 0, lower
MAE wins, valid beats failed, and two failures skip. Per-task Elo likewise retains
numeric ratings for every participant. No point losses or judge scores change.

This is a post-run analysis, not a change to the frozen run policy above. The
original complete-suite ratings remain in the public snapshot's `suite_elo`.
The site displays `overall_elo`. Reproduce with `python3 site/scripts/rate_results.py`;
`site/public/results/pooled-task-elo-v1.json` contains the comparison audit.
