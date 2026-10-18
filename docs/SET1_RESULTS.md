# Set 1 build evidence — October 17, 2026

The causal pulse task and shared research infrastructure are implemented and exercised on real GalaxyPPG recordings. This is a retrospective prototype with provisional reference labels and open release gates. It is not evidence of research-agent training gains or clinical/field validity.

## Data and provisional reference audit

The original Zenodo GalaxyPPG 1.0.0 archive passed its advertised size/MD5 and a local SHA-256 check. Its license metadata is CC BY 4.0; the full legal text, original record metadata and archive are retained in `.research-data/galaxy/`. Source-member hashes, timestamp ranges, rates, duplicate handling, person exclusions and example ECG peak locations are recorded in `.research-runs/pulse-v2/private/audit.json`.

P01 has no Galaxy Watch PPG/ACC or event file. The other **23 people** provide **33,819 eligible windows**, grouped **13 train / 5 development / 5 private test people**. Development contains 6,646 windows. These are overlapping updates, not 33,819 independent biological samples. All sessions/paired devices of one person stay in the same split.

The final mask was independently replayed from cached raw signals and original event records. All masks reproduced; **1,840 excluded windows** have person, bout, cutoff and reason records in `evidence/window-exclusions.json`. The prepared datasets from the first detector revision are superseded and unused in the reported comparison.

Builder visual inspection covered six ECG excerpts across P02/P04/P07/P18/P11/P22, including movement and inverted ECG morphology. The initial detector counted small edge/noise peaks; adding a relative amplitude requirement improved those inspected cases and preceded every reported comparison. Residual noisy/ambiguous peaks remain, especially in P11. A synthetic 72 bpm waveform verifies beat arithmetic, not real-recording annotation accuracy. Further stratified human annotations and independent inter-device timing checks are required before freezing a benchmark release. `benchmark_ready` remains false.

## Executed fixed baselines

All four CPU baselines use identical permitted channels, masks and person splits. Primary loss averages windows within bout, bouts within person, and people equally.

| Baseline | Development MAE, bpm | Private-test MAE, bpm |
| --- | ---: | ---: |
| Training median | 19.307 | 16.155 |
| Causal spectral | 20.936 | 18.435 |
| Motion-aware spectral tracker | 19.944 | 17.004 |
| Compact learned ridge | **12.934** | **10.014** |

Ridge was selected using development loss, with a selection receipt saved before test replay. Private results did not select the winner. These are conventional predictor results, not LLM-agent results. Fitting took approximately 6–24 seconds per baseline on this local host; timings are builder profiling, not a frozen compute requirement or equal-budget search comparison. Source, parameters, dependency lock, seeds and inference/training hashes are stored with each artifact. Per-person/activity errors, counts, interval scores, coverage and width are available in the JSON evidence.

Intervals are nominal 90% intervals calibrated on training residuals. Ridge's development coverage was only **72.7%** with a mean width of **33.92 bpm**. This demonstrates a calibration limitation; point-error improvement must not be presented as reliable uncertainty. The 10 bpm diagnostic is not a clinical standard.

## Executed shared-runner checks

- **13 focused tests pass:** independent aggregation/interval arithmetic, constant/shuffled/missing/enormous-interval cases, nonfinite outputs, person grouping, state size, causal future invariance, known-rate ECG reference, private-read denial, numerical-library execution, streaming protocol, elapsed timeout, persistent failed-evaluation budgets, checkpoint boundaries and judge integrity/null/schema/repair checks.
- **60 existing tests pass** under the normal interpreter. Running those older executor tests inside the new virtualenv produces six failures because that legacy sandbox denies its `/Users/.../.venv-research` startup path. Legacy executor code was preserved. New tests use the numerical virtualenv.
- PR validation against refreshed `main` additionally passed **63 existing unittest cases** and the standalone **15 reward-hack checks** under the dependency-equipped Miniforge interpreter. All 13 pulse checks were rerun in the numerical virtualenv.
- The complete **command → training → isolated development replay → select → final submit → trajectory export** path ran with the median baseline on all five development people and all 6,646 eligible windows. The local sandbox score exactly matched the trusted builder's 19.30728547416091 bpm. This is one successful conventional wiring run; it does not estimate broad agent validity rate.
- The controller froze the selected artifact and closed feedback on submission. Replay exposed one causal request at a time, in fresh workers per bout, without full recordings or private labels. Hash-linked source/evidence/decision/result records and the frozen artifact are retained. A focused test also rejects altered frozen artifacts before replay.
- Three blind packets were exported from the completed run. LLM judge status is **pending**, not an executed judgment. Mock tests establish validator/provider wiring only. The real HTTPS provider adapter is implemented but no authorized provider was invoked.

## Executed Docker verification

The immutable image is `sha256:7751f6262d2768889bebc8b567c771af0dae369db54ef79b208db155ebae9314`. It was built from the pinned Python base and numerical lock after resolving a truncated SciPy download with a pinned resumable pip version. Package checksum verification remained enabled.

Five explicit container gates passed: numerical ridge fitting, private-file exclusion, disabled network, wall-time termination/cleanup, and a 512 MiB allocation killed under a 64 MiB memory limit (exit 137). The complete one-candidate controller smoke also passed in Docker: training on the public training people, causal development replay, final submission and immutable artifact verification. It scored **19.30728547416091 bpm** on all **6,646 windows / five people**, exactly matching the trusted median baseline. Replay used **99 fresh bout workers** and took **171.68 seconds**. Limits were capped at **8 CPU cores and 8,216,887,296 bytes** of Docker memory, below the provisional 32 GiB ceiling. Offline execution used no network or GPU.

The controller closed on final submission. The selected artifact's complete file hash tree was independently checked against its candidate ID. Evidence is `evidence/container-checks.json`, `evidence/docker-replay.json`, `docker-smoke/private/controller.sqlite`, `docker-smoke/trajectory.jsonl`, and the Docker judge packet/status exports. Three blind packets were exported; semantic LLM judgment remains pending.

Docker memory-limit enforcement was tested, but exact peak Docker memory measurement remains unavailable. Local builder/sandbox RSS measurements are reported with their sampling/cumulative limitations. The successful run is a conventional baseline wiring test, not a live LLM-agent or equal-budget full-search result.

The full 30-call conventional grid is implemented against the same controller budget and worker, but only a one-candidate wiring run was executed. No matched LLM-agent comparison, three-seed pilot, supervised training or RL was run.

## Evidence and commands

See [the reproduction guide](SET1_PULSE_ENVIRONMENT.md) for source acquisition, preparation, all baselines, participant runs, private checkpoint/final replay, judge export/import/provider invocation and trainer-neutral trajectory export. The compact tracked record is [evidence/set1-summary.json](evidence/set1-summary.json). Full local evidence is in `.research-runs/pulse-v2/`:

- `baselines-final/report.json`, `selection.json`, artifact directories and per-baseline development/test JSONs.
- `search-smoke-final/` and `docker-smoke/`: private controller ledger, frozen candidates, source snapshots, `trajectory.jsonl` and `search-results.json`.
- `evidence/focused-tests-final.log`, `ecg-audit.png`, `window-exclusions.json`, judge packet/status exports and container logs.
- Existing-suite proof: `.research-data/galaxy/legacy-tests.log`.

To reproduce window-level exclusions without altering the frozen mask:

```sh
.venv-research/bin/python scripts/audit_pulse_exclusions.py .research-data/galaxy .research-runs/pulse-v2 .research-runs/pulse-v2/evidence/new-window-exclusions.json
```

## Files and ownership

New Set 1 files: `research_env/core/`, `research_env/judging/`, `research_env/tasks/pulse/`, shared CLI/container/dependency lock, `tests/test_research_env.py`, and six pulse scripts (`run_pulse_baselines`, `search_pulse`, `score_pulse_checkpoints`, `check_pulse_container`, `audit_pulse_exclusions`, `report_pulse_build`). Documentation is this note, `SET1_PULSE_ENVIRONMENT.md` and the compact JSON evidence. README links to the new task; `.gitignore` excludes local data/runs/virtualenv. Existing recovery code, unrelated PLAN edits and Sets 2/3 were preserved.

The next scientific gate is reference annotation/alignment review. The next agent-evaluation gate is an authorized unmodified-agent pilot, then a separate LLM-judge pilot with human double-review. The build used local CPU compute and made no paid model calls.
