# Set 2: heat core-temperature forecasting as an RL task family

Status, 2026-10-17: the task family is implemented and tested on CPU with the `macos-development` worker. Docker and Beam scoring, and any GRPO run, are pending. This is a retrospective laboratory cycling task. It does not establish heat-injury prevention or field performance.

The task itself (importer, quality masks, split, requests, scoring, baselines) is described in `research_env/tasks/heat/README.md`. This page covers the RL contract layer (`docs/RL_ENV_CONTRACT.md` at the rl-core worktree).

## Data

- Source: Figshare 29120024 v1, CC BY 4.0, 84,215 bytes, md5 `01214bfc835215fe1a5d3f5da1e9ad52`, sha256 `499f5466…052d531`. The local evidence copy at `/Users/nikhi/Documents/Army_Physiology_Research_20261017/evidence/heat/core_cycling_2025.xlsx` matches both, and so does its `checksums.json`.
- Prepared into git-ignored `.research-data/heat/` (`raw/` holds the verified copy, `prepared/` the audited records). Every record keeps its sheet, cell and raw text.
- `Exercise Data!AK9` = 1176 bpm (subject 10, minute 7) is masked under `heat-quality-v1` and logged, never replaced. Two other implausible HR cells and one rectal dropout are masked the same way. Post-cutoff values are never filled.
- Split `heat-split-v1` (14/5/5, sex-stratified, frozen after the audit) is unchanged. `rl_pool` refuses a prepared root whose split differs from the committed manifest.

## Task pool (`research_env/tasks/heat/rl_pool.py`, `POOL_VERSION = heat-rl-pool-v1`)

| Outer split | Fit | Feedback | Reward | Profile | Count per seed |
|---|---|---|---|---|---|
| `agent_training` | 8 training people | 3 training people | 3 training people | fast | 8 partitions × 2 tiers |
| `agent_development` | 11 training | 3 training | 5 dataset development people | fast | 1 per tier |
| `agent_final_evaluation` | 11 training | 3 training | 5 private-test people | benchmark | 1 per tier |

- Private-test people appear only in `agent_final_evaluation`. A test asserts this.
- `options = {"tier": "wearable" | "calibrated", "horizons_min": [5, 10, 20]}`, plus `partition` for training tasks. Calibrated tasks reveal the rectal value at minute 1 and are a separate condition.
- `task_id = "heat-" + sha256(family, version, outer split, seed, people, profile, options)[:16]`.
- Profiles: `fast` = 20 min, 8 dev calls, cutoffs every 5th minute (5..55). `benchmark` = 2 h, 30 calls, every minute 5..55. A `fast` score on a benchmark manifest subsamples its cutoffs.

## Workspace (`stage_workspace`)

The workspace contains:

- `TASK.md`, rendered for the task's tier, people counts and budget
- `DATA_DICTIONARY.md`
- `data/train_manifest.json`, holding only the fit people under opaque aliases
- `data/example_request.json`
- `task_manifest.json`
- `heatlib.py` and the four baselines
- `tools/build_artifact.py`
- `artifact/`, pre-built with the elapsed-time baseline
- a `.heat-rl-workspace` marker; `private_manifest` refuses to write under any directory carrying it

The artifact contract is `artifact/inference.py` with `predict(artifact_dir, request)`. It returns the ordered records `{horizon_min, mean_c, lower_c, upper_c}`; a `{"forecasts": [...], "state": ...}` wrapper is also accepted.

## Scoring (`score` → `replay`)

`replay` uses the shared `research_env.core.worker.execute`, with one fresh worker per person (one bout per person here). Requests are streamed in cutoff order and each one is prefix-only. Participant stdout is redirected so it cannot corrupt the protocol.

Every response must be a list of exactly the requested horizons with exactly the four keys, and the values must be finite and ordered. Any failure gives `execution_valid=False` with `primary_loss=None`.

The loss is computed by `scoring.score`: horizon MAE within a person, then the equal mean of the horizons, then the equal mean over people. IS90, coverage, width, bias, RMSE and the hottest-period slice (reference ≥ 38.5 °C) go in `metrics`.

## Frozen reward baseline

`baseline_loss` is **`elapsed_time`**, the mandatory protocol-shortcut reference. It is fitted on `fit_people` and scored on `reward_people` with the same requests, targets and cutoffs as the candidate.

It runs in-process because it is builder code. The tests and the script check that a worker replay of the staged artifact gives the identical loss. `ridge_lag` was not frozen: on the wearable tier it does not reliably beat elapsed time, and it needs leave-one-person-out tuning, so it is slower and its results vary more with the partition.

## Commands

```bash
uv venv --python 3.13 .venv-research && uv pip install --python .venv-research/bin/python -r research_env/requirements.lock
.venv-research/bin/python -m research_env.tasks.heat prepare .research-data/heat/raw .research-data/heat/prepared
.venv-research/bin/python -m unittest tests.test_heat_task -v
.venv-research/bin/python -m unittest discover -s research_env/tasks/heat/tests -t .
.venv-research/bin/python scripts/run_heat_rl_baselines.py .research-data/heat/prepared .heat-runs/rl-baselines
```

No openpyxl is needed; the heat importer reads `.xlsx` with the standard library.

## Baseline replay (executed, seed 0, macos-development, 101 s CPU)

Reward-people primary loss in °C:

| Task | Tier | Elapsed (frozen) | Ridge lag | Grad. boost | Thermal |
|---|---|---|---|---|---|
| agent_development (5 dev people, fast) | wearable | 0.359 | 0.346 | 0.426 | 0.372 |
| agent_final_evaluation (5 test people, benchmark) | wearable | 0.239 | 0.255 | 0.340 | 0.247 |
| agent_development | calibrated | 0.282 | 0.368 | 0.412 | 0.264 |
| agent_final_evaluation | calibrated | 0.461 | 0.241 | 0.308 | 0.211 |

Across the first four training partitions per tier, the frozen elapsed-time loss ranged from 0.18 to 0.36 °C (wearable) and from 0.29 to 0.37 °C (calibrated). With 3 reward people per task, per-task reward is noisy. GRPO groups should compare rollouts on the same task only.

## Integration needs (paths not owned by heat)

- `research_env/core/evaluation.py` and `core/agent.py` are pulse-specific. `ResearchEnvironment` evaluate should call `family.score(frozen_artifact, manifest, backend=…, image=…, wall_s=…, profile=…)` instead of the pulse `evaluate`. Initial evidence should read `workspace/TASK.md` rather than `research_env/tasks/pulse/TASK.md`.
- `rl_pool.py` defines a local `TaskSpec` mirror until `research_env.rl.api` exists. When it does, the module imports `TaskSpec` and `register` from it and registers `FAMILY` at import.
