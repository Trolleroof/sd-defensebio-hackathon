# Set 3: load-carriage metabolic-cost task: completion note

Status: **pilot environment built; baselines executed; live agent and live LLM judge pending.**
Prepared 2026-10-17. Retrospective public data. Not clinically or field validated, and not evidence of
research-agent training gains.

## 1. Files and ownership

Set 3 owns `research_env/tasks/load/` and `tests/test_load_task.py`.

| Path | Role |
|---|---|
| `sources/evobreath-2015/` | Immutable source XLS + protocol PDF, hashes, license, provenance (`SOURCE.md`) |
| `research_env/tasks/load/importer.py` | `prepare()`: hash check, explicit cell parsing with lineage, unit conversion, quality flags, split assignment |
| `research_env/tasks/load/splits.py`, `config/split_v1.json` | Deterministic sex-stratified 28/10/10 person split, frozen |
| `research_env/tasks/load/schema.py` | Constants, tracks, allow-list request builder, prediction validation |
| `research_env/tasks/load/scoring.py` | `score()`: person-balanced metrics, IS90, ordering, by-sex, person bootstrap |
| `research_env/tasks/load/baseline_models.py` | Self-contained baselines (`train`/`predict`, numpy only, JSON params) |
| `research_env/tasks/load/search.py` | Equal-budget conventional search (30 development calls) |
| `research_env/tasks/load/runner.py` | Controller: permitted-data materialization, opaque IDs, dev budget, replay |
| `research_env/tasks/load/worker.py` | **Interim** sandboxed worker (replace with Set 1's `research_env.core`) |
| `research_env/tasks/load/judging.py` | Blind packet export + verdict validation (**interim** until Set 1's `research_env.judging`) |
| `research_env/tasks/load/workspace.py`, `workspace_template/` | Participant workspace: `TASK.md`, data dictionary, examples, baseline |
| `research_env/tasks/load/config/task_manifest.json` | Budgets, tracks, frozen skill baseline losses, ranking policy |
| `research_env/__init__.py`, `research_env/tasks/__init__.py` | Empty namespace packages (shared; contain no logic) |

Shared-infrastructure note: Set 1's runner and judging packages did not exist when this was built.
`worker.py` reuses the Seatbelt profile from `benchmark/executor.py` and `judging.py` implements only the
common contract (packets, verdict validation, process-score arithmetic). Neither is a competing shared
implementation. Swap both for Set 1's modules when they land; the load-specific parts (`TASK_CONTEXT`
examples, event-to-packet mapping) stay here.

## 2. Reproduction commands

Python 3.11 with numpy (tested with 1.23.5); `xlrd==2.0.1` for `prepare` only (`requirements.lock`).
The worker needs macOS `sandbox-exec` and fails closed without it.

```bash
python3.11 -m research_env.tasks.load prepare --out runs/load/prepared
```

```bash
python3.11 -m research_env.tasks.load baselines --prepared runs/load/prepared --out runs/load/baselines
```

```bash
python3.11 -m research_env.tasks.load search --prepared runs/load/prepared --out runs/load/search_primary --track primary
```

```bash
python3.11 -m research_env.tasks.load workspace --prepared runs/load/prepared --track primary --out runs/load/ws_primary
```

Participant development call (controller side; counts against the 30-call budget, failures included):

```bash
python3.11 -m research_env.tasks.load dev-eval --prepared runs/load/prepared --submission runs/load/ws_primary/submission --run-dir runs/load/run1 --candidate-id c1
```

Private final score (written to the run dir, never returned to the participant), then exact replay:

```bash
python3.11 -m research_env.tasks.load final --prepared runs/load/prepared --submission runs/load/ws_primary/submission --run-dir runs/load/run1 --candidate-id c1
```

```bash
python3.11 -m research_env.tasks.load replay --prepared runs/load/prepared --submission runs/load/ws_primary/submission --run-dir runs/load/run1 --candidate-id c1
```

Judge packets from an experiment event log, and import of verdict attempts:

```bash
python3.11 -m research_env.tasks.load judge-export --events runs/load/run1/events.jsonl --out runs/load/run1/judge_packets.jsonl
```

```bash
python3.11 -m research_env.tasks.load judge-import --packets runs/load/run1/judge_packets.jsonl --verdicts runs/load/run1/verdicts.json
```

Checks:

```bash
python3.11 -m unittest discover -s tests -p "test_load_task.py" -v
```

## 3. Data audit (executed)

- Source hashes match Mendeley's published SHA-256 for both files; `prepare` refuses any mismatch.
- 48 people (21 F, 27 M), 192 person-condition records, **189 eligible**. Three 15 kg targets are `-` in the
  source (`DATA!N20` BE0002, `DATA!N38` BE0058, `DATA!N44` BE0063). They are ineligible records, not
  imputed and not excluded people. The 192 condition rows are **not** 192 independent cases.
- Numeric columns mix numbers and numeric text. Every value keeps its cell reference, original cell type
  and original value. All text cells parsed cleanly except the three `-`.
- Flags (informational, nothing excluded): BE0003 and BE0030 share an FFM of 59.69 kg, the only
  two-decimal FFM values in the table, which suggests a copy error. BE0003 and BE0064 have
  |FM + FFM − mass| > 2 kg. BE0065 and BE0066 have an unloaded-walk/RMR ratio below 2. For 32 of 48 people,
  measured cost does not rise monotonically with load; it is kept as measured. FM, FFM and RMR are
  lab-only fields and never reach primary inference.
- Units: `W = kcal/day × 4184 / 86400` (a rate, not the 11-minute bout's energy); gross, RMR not
  subtracted; speed 4/3.6 m/s; grade 0.
- Split (frozen, `config/split_v1.json`): train 28 (12 F), development 10 (5 F), test 10 (4 F); eligible
  records 111 / 40 / 38.
- Builder exposure: builders viewed all source rows during the audit. The data are public and may be in
  pretraining corpora, so the release is retrospective.

## 4. Executed checks (`tests/test_load_task.py`, 30 tests, all passing; full repo suite 93 tests passing)

| Gate | Check |
|---|---|
| Unit conversion | 86400 kcal/day → 4184 W; 5000 → 242.1296 W; 4 km/h → 1.1111 m/s; Pandolf hand examples |
| Numeric parsing | number / numeric text / `-` cell types; specific cells (`DATA!K2` = 4978 → 241.0642 W) |
| Fail-closed source | Tampered source file → `prepare` raises |
| Person isolation | Splits disjoint, cover all 48, regenerate exactly; every record of a person shares a split |
| Training data | Train people only (28 people, 111 rows); allow-listed fields; no source IDs; no lab fields |
| Primary/calibration separation | Altering hidden 0 kg targets leaves primary requests byte-identical; calibration reveals only the same person's 0 kg value and scores only 5–15 kg |
| Load holdout | `holdout15` training has no 15 kg rows; scored on 15 kg only |
| Metric arithmetic | Hand-computed two-person example: person-balanced MAE 20 (row mean would be 16.7), skill, IS90, coverage, W/kg, bias, per-load, ordering |
| Invalid submissions | Missing, NaN, `lower > mean`, non-numeric, non-object → `execution_valid: false`, null loss and skill |
| Adversarial candidates | Constant and shuffled-across-people candidates are valid but worse than ridge; a ±9e5 W interval reaches 100% coverage but has >1000× the IS90; a predict that raises on 15 kg makes the run invalid |
| Leakage | Reading the private `records.jsonl` from `predict` or from `train` is denied by the sandbox (confirmed `PermissionError`); network is denied |
| State reset | Module-level counter never exceeds 4, so each person gets a fresh process |
| Artifact replay | Retraining from source with the same seed gives identical artifact and test-prediction hashes |
| Budget | Failed dev calls count; the third call on a 2-call budget is refused; public feedback has no per-person rows |
| Judge packets | Decision packet excludes the action's own outcome and later events; identity and private results are stripped; verdict validator rejects bad keys, out-of-range or boolean scores, unknown evidence IDs, missing reasons, uncited scores, scored inapplicable criteria, and hash or event mismatches; at most 2 repairs; unresolved verdicts stay missing; process score `25 × (mean − 1)` is withheld when coverage is incomplete |

## 5. Results

### Baselines (executed, seed 0, CPU)

Each baseline trained in the sandboxed worker in 0.1–0.2 s wall time with about 37 MB peak RSS. The full
3-track sweep, including dev and test scoring, took 39 s on the host. The 2-hour, 8-core, 32 GB budget is
therefore generous for this table task. It is still a provisional design default.

Primary track (person-balanced, W; Δ is the paired person bootstrap of test MAE vs ridge):

| baseline | dev MAE | test MAE | test W/kg | bias | worst load | IS90 | cov90 | Δ vs ridge (95% CI) |
|---|---|---|---|---|---|---|---|---|
| mean_by_load | 32.7 | 63.9 | 0.855 | −20.8 | 65.4 | 261.3 | 0.82 | +31.6 [+10.0, +56.3] |
| ridge (frozen skill reference) | 23.9 | 32.3 | 0.452 | −12.1 | 41.3 | 180.6 | 0.88 | ref |
| gbm (numpy depth-2 boosting) | 27.3 | 36.0 | 0.497 | −25.5 | 48.4 | 215.2 | 0.95 | +3.8 [−9.8, +17.4] |
| pandolf (1977, η = 1.0 treadmill) | 30.8 | 29.3 | 0.407 | −4.7 | 41.7 | 170.2 | 0.93 | −3.0 [−9.3, +3.1] |
| pandolf_recal (linear recalibration) | 26.7 | 30.1 | 0.413 | −8.9 | 41.3 | 165.6 | 0.88 | −2.2 [−7.3, +2.9] |

Calibration track (scored on 5/10/15 kg; uses a measured 0 kg value, so it is **not** comparable to primary
without disclosing that input): `calibration_offset` test MAE 24.2 W versus no-calibration ridge 32.8 W on
the same targets (Δ −8.6 [−19.5, +1.4]).

Holdout-15 track (15 kg on unseen people, never seen in training): ridge 38.3, pandolf_recal 40.7,
pandolf 41.7, gbm 50.9 (trees cannot extrapolate), mean_by_load 60.9 W. Pandolf intervals cover only 62%.

Full tables, including per-load and by-sex results, are regenerated by the `baselines` command
(`baseline_report.md`, `baseline_results.json`).

Readings, with 10 test people: no baseline is distinguishable from ridge except mean_by_load. The bootstrap
CIs for Pandolf and the GBM span zero. All baselines predict cost increasing with load, so ordering
concordance (0.77) is the same for every one; it only separates non-monotone candidates. Female test
error is higher (≈37 W vs 24–29 W for men), but with n = 4 this is a report, not a finding.

### Equal-budget conventional search (executed, primary, seed 0)

30 sampled configurations, one development call each. The selected configuration was ridge with
`alpha = 100` and features mass, load, load²/mass and mass×load. It had the best dev MAE (23.6 W; range of
all 30: 23.6–35.2) but a **test MAE of 44.4 W**, worse than the default ridge (32.3 W). Selecting on 10
development people overfits. An agent has to beat this control under the same budget. Any agent result
should be reported against it and against frozen ridge, with the person bootstrap.

### Live agent and live LLM judge: **pending**

No participant LLM or judge provider is configured or authorized in this session. `judge-export` writes
blind packets with `llm_judge_status: pending`. No verdicts exist and none were simulated.

## 6. Remaining blockers and next decisive gate

1. **Shared runner and judge (Set 1).** Replace `worker.py` and the generic parts of `judging.py` with
   `research_env.core` and `research_env.judging` once frozen. Check that the participant harness writes
   experiment events in the field schema documented in `judging.py`.
2. **Isolation is macOS Seatbelt only.** It is not a hardened multi-tenant sandbox. Memory limits are best
   effort (RLIMIT_DATA/RSS) and peak RSS is an upper bound over child processes. CPU count is set through
   BLAS thread variables, not cgroups.
3. **LCDA equation not implemented.** The Looney et al. LCDA walking equation was not verified against
   the primary source in this audit, so it is excluded rather than reconstructed from memory. Pandolf
   (1977) is implemented with all inputs present (η = 1.0 for treadmill; grade 0).
4. **Small cohort.** With 10 test people, CIs are wide (see above). Repeated agent seeds (≥3) must be
   reported separately from the 10-person biological sample.
5. **Extensions not started:** Slade GRF/EMG data (license unresolved) and the OSF military load-carriage
   project (license and schema unverified). A laboratory-input tier (FM/FFM/RMR) would need its own
   baseline and feature manifest.

**Next decisive gate:** run ≥3 seeds of an unmodified participant agent on the primary track through
Set 1's runner. Compare it to frozen ridge and to the equal-budget search on the private test split using
the person bootstrap. Export packets for the authorized judge. Have two humans audit a stratified packet
subset.
