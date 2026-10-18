# Set 2: core-temperature forecasting (`heat-core-forecast-v1`)

Built from [docs/AGENT_BUILD_INSTRUCTIONS.md](../../../docs/AGENT_BUILD_INSTRUCTIONS.md), Set 2. The task: from noninvasive history, forecast independently measured rectal temperature 5, 10 and 20 minutes ahead, as a point estimate and a 90% interval. This is a retrospective laboratory cycling result. It does not establish heat-injury prevention or field performance.

## 1. Files and ownership

Everything in `research_env/tasks/heat/` is owned by Set 2. The only edit outside it is one `.gitignore` line (`.heat-runs/`).

| File | Purpose |
|---|---|
| `source.py` | Figshare 29120024 v1 identity (CC BY 4.0, md5/sha256), fetch, verify |
| `xlsx.py` | Stdlib `.xlsx` reader that keeps each value's sheet, A1 address and raw text |
| `importer.py` | `prepare()`: layout audit, cell-lineage records, channel roles, dataset manifest |
| `quality.py` | Frozen rules `heat-quality-v1`. They mask values and never replace them. |
| `splits.py`, `split_manifest.json` | Frozen 14/5/5 person split, seeded and sex-stratified |
| `requests.py` | Prefix-only request builder and evaluator-owned targets |
| `scoring.py` | `score()`, IS90, hottest-period slice, person-level paired bootstrap |
| `task_manifest.json` | Horizons, metric, budget, frozen elapsed-time baseline losses |
| `workspace_template/` | Participant `TASK.md`, `DATA_DICTIONARY.md`, `heatlib.py`, four baselines, `tools/heat.py` |
| `worker.py` | **Interim.** Isolated train/predict worker (sandbox-exec, JSON only, one fresh module per person) |
| `session.py` | **Interim.** Budgets, hash-chained experiment events, dev feedback, incumbents, private replay |
| `judging.py` | **Interim.** Decision and final-selection packets, `physiology-research-judge-v1` validator, process score, Anthropic adapter |
| `pilot.py` | **Interim.** Live-agent pilot: `claude -p` in an OS sandbox, file-inbox controller |
| `cli.py`, `__main__.py` | Command line |
| `tests/test_heat.py` | 26 focused checks |

The three **interim** modules exist only because Set 1's `research_env/core` and `research_env/judging` did not exist yet. They follow the proposed v1 interface (`prepare`, `train`, `predict`, `score`). The task-specific modules (importer, requests, scoring, baselines, workspace) are designed to move to Set 1's runner unchanged.

## 2. Reproduction

```bash
python3 -m research_env.tasks.heat prepare data/raw/heat .heat-runs/prepared
python3 -m research_env.tasks.heat baselines .heat-runs/prepared .heat-runs/baselines
python3 -m unittest discover -s research_env/tasks/heat/tests -t . -v
python3 -m research_env.tasks.heat create .heat-runs/prepared .heat-runs/session-01
python3 -m research_env.tasks.heat pilot .heat-runs/prepared .heat-runs/pilot-01 --model sonnet --dev-calls 5 --wall-min 30
python3 -m research_env.tasks.heat replay .heat-runs/pilot-01
python3 -m research_env.tasks.heat judge-export .heat-runs/pilot-01 .heat-runs/pilot-01/judge_packets.json
python3 -m research_env.tasks.heat judge-import packets.json verdicts.json judging.json
```

- `prepare` copies the verified local evidence file when present. Otherwise it downloads from Figshare and verifies the md5 and sha256.
- `baselines --freeze` rewrites the frozen baseline losses in `task_manifest.json`. Do not rerun it after comparisons start.
- Session ops: `dev-eval`, `declare`, `submit`. Participants call the same operations through `tools/heat.py`.

## Data decisions (audited 2026-10-17)

- **Source.** The local copy matches the md5 Figshare publishes (`01214bfc…`); sha256 is `499f5466…`. There are 24 subjects (1–10 and 12–25; 11 is absent everywhere) and 60 minutes each.
- **Time.** In `Core Temp`/`Skin Temp`, row r is minute r−2. `Exercise Data` labels mix intervals (`0-1`) and end minutes (`26`), and the end minute is always r−2. Each interval is stamped at its end. A value stamped t is available at cutoff c when t ≤ c, so the current minute is included.
- **Inputs (wearable tier):**
  - HR, which is sparse: only minutes 6–10, 21–25, 36–40 and 51–55 (plus one at minute 56)
  - `CORE TSK`, the skin temperature reported by the device
  - prescribed work rate
  - measured ambient temperature and RH
  - sex, age, height, mass, BMI
- **Excluded:**
  - vendor CORE core-temperature estimate
  - `YSI TSK`, the laboratory reference skin thermistors
  - VO2 and VO2peak, RPE
  - the ultrasound subcutaneous-fat sheet
  - rectal history
- **Calibrated tier.** Same inputs, plus the measured rectal value at minute 1. This needs a rectal probe before exercise, and the tier is scored separately.
- **Quality masks:**
  - HR `AK9`=1176 (the cell named in the brief), `AI23`=1556 and `AY26`=1152 are all outside 30–230 bpm.
  - Rectal `Y42`=35.0 (subject 24, minute 40) is an isolated dropout between values near 38.4 °C.
  - All four are masked and logged with their cell. Nothing is imputed.
  - Subject 9's record ends at minute 56.
- **Horizons.** These are requested wherever cutoff + h ≤ 60, independent of whether a target exists, so request shape cannot leak masks. Targets with no reference are excluded from scoring by the evaluator, but a valid prediction is still required.
- **Hottest period.** Frozen as reference ≥ 38.5 °C: 463 target minutes across 19 people.
- **Alerts omitted.** Only 9 of 24 people ever reach 39.0 °C, which is too sparse to rank alerts.
- **Split** (`heat-split-v1`, seed 20261017, sex-stratified):
  - train: 1, 2, 3, 5, 7, 8, 12, 17, 18, 19, 20, 22, 23, 25
  - development: 4, 9, 10, 13, 24
  - private test: 6, 14, 15, 16, 21
  - No exclusions were needed. The DaLiA exposure lists do not apply to this dataset.

## 3. Executed checks

`python3 -m unittest discover -s research_env/tasks/heat/tests -t . -v` ran 26 checks, all passing, on 2026-10-17.

- **Provenance:** known source cells (`AK9`, `C3`, `Y42`), cell lineage on every record, the exact masked-cell set with no replacement, and excluded channels absent.
- **Person isolation:**
  - the split is disjoint, complete and reproducible from its seed
  - the train manifest holds only training people
  - aliases are opaque
- **Causality:** perturbing every input and reference after the cutoff leaves the request byte-identical in both tiers. The wearable tier carries no rectal data, and the calibrated tier reveals only minute 1.
- **Horizon masking:** subject 9's end of record, subject 24's masked minute, and requested horizons near minute 60.
- **Runner/participant parity:** `heatlib.request_at` matches the runner's `build_request` exactly.
- **Metric arithmetic:** hand-computed primary loss, skill, IS90 (for example 0.4 + 20·0.3 = 6.4), bias, and the hottest slice, plus the paired bootstrap.
- **Invalid submissions:**
  - missing, reordered, truncated or `lower > mean` outputs make the run invalid with a null loss
  - a NaN at an ineligible target still invalidates the run
  - a shuffled candidate is invalid
- **Enormous intervals:** 100% coverage but a worse IS90, and an unchanged point loss.
- **Leaky candidates:** worker reads of the private targets, the prepared data and the repo are all denied (`Operation not permitted`).
- **Budgets:** failed calls count, the call cap is enforced, the wall-clock cap is enforced, a decision record is required, and feedback closes after submit.
- **Artifact replay:** a clean retrain gives the same artifact hash and the same dev feedback. The event hash chain verifies, and checkpoint replay works.
- **Judge:** decision packets exclude their own outcome and all later events, and contain no private path or session secret. The validator rejects bad scores, fabricated evidence IDs, a scored `final_selection_claims` on a decision packet, missing criteria and a wrong packet hash. Process-score arithmetic (criterion means 3/3/5/3/1 → 50) and partial coverage are checked. A mock provider exercises the repair path.

Evidence paths (all gitignored, not committed): `.heat-runs/prepared/dataset_manifest.json`, `.heat-runs/prepared/quality_log.jsonl`, `.heat-runs/baselines/baseline_report.json`.

## 4. Results

Interpret these cautiously: 5 people per held-out split, one agent seed.

### Baselines (executed, CPU only, 15 s total)

Primary loss in °C; lower is better.

| Tier | Baseline | Dev | Private test |
|---|---|---|---|
| wearable | elapsed-time (frozen reference) | 0.357 | 0.220 |
| wearable | ridge lag | 0.346 | 0.245 |
| wearable | gradient boosting | 0.412 | 0.348 |
| wearable | thermal state-space | 0.365 | 0.226 |
| calibrated | elapsed-time + minute-1 offset | 0.247 | 0.370 |
| calibrated | ridge lag | 0.335 | 0.228 |
| calibrated | gradient boosting | 0.388 | 0.295 |
| calibrated | thermal state-space | 0.249 | 0.193 |

In the wearable tier, no learned baseline clearly beats elapsed-time on both splits. This is the constant-load protocol shortcut the brief warned about. The calibrated thermal model is the best on test, but it uses an extra rectal measurement, so it must not be compared with the wearable results without saying so.

### Live agent pilot (executed once, reduced budget)

- **Setup:** Claude Code `claude -p --model sonnet` (CLI 2.1.302), one seed, budget of 5 dev calls and 30 minutes.
- **What it did:** finished in 93 s, 6 turns, $0.22. It used 1 dev call and never declared an incumbent (that call failed because `--reason` was missing). It submitted a mean-curve + ridge-residual model with `REPORT.md`.
- **Dev:** 0.339 °C, skill +0.05.
- **Private test:** 0.238 °C against elapsed-time's 0.220, skill −0.08. Coverage90 was 0.96 and IS90 1.09.
- **Replay:** clean, matching the recorded dev result.

This shows the loop works. It is not evidence of research ability. Artifacts are in the session scratchpad: `agent_transcript.jsonl`, `private/events.jsonl`, `private/final_score.json`.

### LLM judge: pending

Packets were exported (2 for the pilot) with `llm_judge_status: pending`. No judge provider is configured and no judge calls were made. The Anthropic adapter exists, but it runs only with an explicit config and `ANTHROPIC_API_KEY`.

## 5. Blockers and next gate

- **Set 1 integration.** Port the worker, session and judging modules onto `research_env/core` and `research_env/judging` once those exist. The worker is macOS `sandbox-exec` only, and Python startup still reads system paths, so it is not a hardened multi-tenant sandbox. There is no pinned container image; the runtime is Python 3.14.4 with numpy 2.4.4.
- **Pilot isolation.** Claude Code runs in an `(allow default)` profile with reads denied for the private root, the prepared data, the repo and the evidence directory. Network is open for model inference. The harness, which is outside the sandbox, also reads `~/.claude` user settings.
- **Statistics.** 5 people per held-out split gives wide person-level bootstrap intervals. Interpreting results needs ≥ 3 agent seeds at the full budget (2 h, 30 calls), compared against frozen conventional search with the same budget. Conventional search is not implemented yet.
- **Judge.** Configure an authorized provider and model, run it, then audit a stratified subset with two human reviewers before trusting any process score.
- **Extensions.** The Zenodo earbud/esophageal dataset (19835255) and the Mendeley heat-tolerance dataset are not yet audited.
- **Next decisive gate:** a full-budget, 3-seed Sonnet run in the wearable tier, compared with elapsed-time using the person-level paired bootstrap on private test.
