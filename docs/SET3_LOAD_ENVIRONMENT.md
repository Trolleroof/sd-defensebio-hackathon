# Set 3: load-carriage metabolic-cost environment

Status: what is built and checked as of October 17, 2026. Retrospective laboratory table task. It is not clinically or field validated, and nothing here shows that research-agent training improves anything.

## Data

EVOBREATH 2015 is Mendeley `rhmdzmx6kk` v1 (DOI 10.17632/rhmdzmx6kk.1), licensed CC BY 4.0. The original XLS and protocol PDF came from `/Users/nikhi/Documents/Army_Physiology_Research_20261017/evidence/load/`. Their SHA-256 values match the ones the Mendeley files API publishes. Copies are in the git-ignored `.research-data/load/originals/`. Running `fetch()` (`--fetch`) downloads any missing file from Mendeley and checks it against the same checksums.

The importer is `research_env/tasks/load/data.py`. It runs only on the controller and uses `xlrd==2.0.1`, which is pinned in `research_env/tasks/load/requirements-import.txt`. Workers do not need xlrd.

- **Numeric parsing.** Every numeric cell is parsed according to its stored type. 244 numeric cells are stored as text. A literal `-` is recorded as a missing value, and any other unparseable text stops the import. Each record keeps the original cell coordinate, type and value.
- **Units.** `watts = kcal_per_day * 4184 / 86400`. Speed is 4/3.6 m/s and grade is 0 %.
- **Target.** The target is gross metabolic power. Net power (gross minus RMR) is stored as a separate field and is never scored.
- **Coverage.** All 48 people are kept, giving 189 of 192 possible condition rows. The 15 kg value is `-` for three people (BE0002 N20, BE0058 N38, BE0063 N44), so those three conditions are excluded. These rows are repeated measures within 48 people, not 189 independent cases.
- **Quality flags.** 126 rows belong to people whose costs do not rise strictly with load. Costs are never forced to be monotonic. No row has gross cost at or below 1.5 × that person's RMR.
- **Splits.** Splits are by person and stratified by sex, using salt `evobreath-load-split-v1`: 28 train, 10 development, 10 test. Sex counts are F12/M16, F5/M5 and F4/M6.

## Task

The participant-facing files are `TASK.md`, `DATA_DICTIONARY.md` and `example-request.json`.

- **Request fields.** sex, age, stature, body mass, load, speed, grade and `calibration`.
- **Response.** `{mean_w, lower_w, upper_w}`.
- **Primary tier.** `calibration` is null and all loads are scored.
- **Calibration tier.** The request reveals the person's own 0 kg gross watts, and the 0 kg condition is not scored.
- **`heldout15` variant.** All 15 kg rows are removed from the fit data, and only 15 kg is scored on unseen people. This variant is kept separate from person transfer.
- **Excluded inputs.** RMR, FM, FFM, femur length and bi-iliac breadth never appear in requests or public rows. Person IDs appear only as task-salted pseudonyms.

Scoring (`scoring.py`): the primary metric is absolute watt error averaged within each person, then averaged equally across people. Secondary metrics are RMSE, bias, W/kg error, per-load MAE, worst-load MAE, within-person ordering concordance and MAE by sex. Uncertainty is scored with IS90, with coverage and width reported separately. Any missing, extra, nonfinite, unordered or wrongly keyed output makes the whole attempt invalid, with null loss.

## Baselines

All baselines live in `baseline_model.py`, which needs only numpy and also serves as each artifact's `inference.py`:

- mean by load
- ridge regression on mass, stature, age, sex, load, mass×load and load²
- a compact CART tree (depth 3, at least 8 rows per leaf)
- **Pandolf 1977**, which needs only body mass, load, speed, grade and terrain factor 1.0. It predicts gross watts and nothing is fitted.

Intervals use the 90 % quantile of in-sample absolute residuals.

**LCDA (Looney et al.) is excluded.** Its coefficients could not be checked against the primary source offline, and fabricating parameters is forbidden. Pandolf is known to under-predict at slow speeds, and no correction is applied.

## Executed baseline results

These are trusted builder runs, fitted on the 28 train people. Selection used the development people only. The worker replay (`macos-development`) matched the in-process numbers.

| Config | mean_by_load | ridge | tree | Pandolf | Selected on dev |
| --- | --- | --- | --- | --- | --- |
| primary/person dev (test) | 49.4 (40.5) | 35.2 (31.2) | 54.0 (37.8) | 28.9 (27.1) | Pandolf |
| calibration/person dev (test) | 21.0 (20.1) | 20.2 (19.4) | 29.9 (34.0) | 20.6 (20.0) | ridge |
| primary/heldout15 dev (test) | 46.1 (33.5) | 47.0 (22.6) | 61.3 (34.1) | 43.0 (19.9) | Pandolf |

Still pending:

- equal-budget conventional search
- strict Docker replay
- live agent runs
- live LLM judge runs

## RL family

`rl_pool.py` defines `FAMILY = LoadFamily()` and builds a pool of 24 specs:

- **agent_training:** 6 seeded partitions of the 28 train people into 18 fit / 5 feedback / 5 reward, crossed with 3 configs, run on the fast profile. That gives 18 specs.
- **agent_development:** 23 fit / 5 feedback from train, with the 10 development people as reward. 3 specs, benchmark profile.
- **agent_final_evaluation:** the same split with the 10 test people as reward. 3 specs, benchmark profile.

Test people never appear in training or development specs. The workspace starts from a ridge artifact. The frozen reward baseline is the development-selected fixed baseline per configuration (`REWARD_BASELINES`): Pandolf 1977 for primary and held-out-15 kg, ridge for calibration. `TaskSpec` falls back to a local copy that matches `research_env/rl/api.py` until that file exists.

## Commands

```bash
uv venv --python 3.13 .venv-research
uv pip install --python .venv-research/bin/python -r research_env/requirements.lock -r research_env/tasks/load/requirements-import.txt
.venv-research/bin/python scripts/prepare_load.py .research-data/load/originals .research-data/load/prepared  # add --fetch if absent
.venv-research/bin/python scripts/run_load_baselines.py .research-data/load/prepared .research-runs/load-baselines --worker-backend macos-development
.venv-research/bin/python -m unittest tests.test_load_task -v
```

## Hooks needed in shared code

- `research_env/core/evaluation.py` imports pulse code directly. It needs a task adapter, for example `evaluate(..., task=family)` that delegates to `family.score`.
- `research_env/core/agent.py` hardcodes the path to the pulse `TASK.md`.
- `inference_entry.py` injects `state` into each request and requires `state` in the result, so it cannot be used for this task. The load task ships its own stateless entry.
