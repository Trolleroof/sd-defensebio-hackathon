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
| `judging.py` | Heat adapter onto Set 1's shared judge (`research_env/judging`): turns heat events into public evidence records, including candidate source and the session's actual budget, and builds shared packets |
| `claude_judge.py` | Claude Code CLI provider for the shared `judge_packet`: one fresh `claude -p` per packet, no tools, frozen shared system prompt, call and USD budgets |
| `pilot.py` | **Interim.** Live-agent pilot: `claude -p` in an OS sandbox, file-inbox controller |
| `cli.py`, `__main__.py` | Command line |
| `tests/test_heat.py` | 26 focused checks |

**Judging now runs on Set 1's shared judge** (merged from `main`, PR #4). Heat no longer has its own rubric, validator or provider: packets are built with `research_env.judging.judge.packet`, and Set 1's `judge-run` and `judge-import` commands read heat packet files unchanged.

The **interim** worker, session and pilot modules remain. Set 1's `core/evaluation.py` imports pulse data and scoring directly, and its doc says heat and load need task adapters that do not exist yet. Adding that hook changes Set 1's interface, so it needs coordinating with Set 1 rather than editing their code from here. The task-specific modules (importer, requests, scoring, baselines, workspace) are designed to move onto the shared runner unchanged.

## 2. Reproduction

```bash
python3 -m research_env.tasks.heat prepare data/raw/heat .heat-runs/prepared
python3 -m research_env.tasks.heat baselines .heat-runs/prepared .heat-runs/baselines
python3 -m unittest discover -s research_env/tasks/heat/tests -t . -v
python3 -m research_env.tasks.heat create .heat-runs/prepared .heat-runs/session-01
python3 -m research_env.tasks.heat pilot .heat-runs/prepared .heat-runs/pilot-01 --model sonnet --dev-calls 5 --wall-min 30
python3 -m research_env.tasks.heat replay .heat-runs/pilot-01
python3 -m research_env.tasks.heat judge-export .heat-runs/pilot-01 .heat-runs/pilot-01/judge_packets.json
python3 -m research_env judge-import .heat-runs/pilot-01/judge_packets.json .heat-runs/pilot-01/judging.json
python3 -m research_env.tasks.heat judge-run-claude .heat-runs/pilot-01/judge_packets.json .heat-runs/pilot-01/judge-opus --model opus --max-calls 6 --max-usd 3
python3 -m research_env judge-run .heat-runs/pilot-01/judge_packets.json judge-config.json .heat-runs/pilot-01/judge-out --key-env PHYSIOLOGY_JUDGE_KEY
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

`python3 -m unittest discover -s research_env/tasks/heat/tests -t . -v` runs 26 checks. All pass under both Homebrew Python 3.14 (numpy 2.4.4) and Set 1's pinned `.venv-research` (Python 3.13, `research_env/requirements.lock`, numpy 2.2.6). Baseline losses are identical on both. Set 1's `tests/test_research_env.py` (13 tests) also passes after the merge.

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
- **Judge (through Set 1's shared judge):**
  - every evidence record predates its packet's event
  - a decision packet excludes its own outcome, and a later packet sees it only as a `development_result` record, which turns on `feedback_integration`
  - packets contain no private path, session secret or private-test data
  - the shared validator rejects bad scores, unknown evidence IDs, a scored inapplicable criterion, missing criteria and a wrong hash
  - shared `summarize` reports pending, partial, and complete (all 4s → 75)
  - a mock provider exercises the shared `judge_packet` repair path
  - a fake `claude` executable checks the CLI provider: no tools, the shared system prompt, a fresh context per call, the repair path, and the call budget
  - decision packets carry the candidate's source, and the task evidence carries the session's actual budget

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

### LLM judge: executed once on the pilot (Claude Code CLI)

The judge was `claude-opus-5-5`, run through `claude -p` on the user's Claude Code login. It had no tools, used the frozen shared system prompt, and judged each packet in a fresh context. The participant was Sonnet, so judge and participant are different models. Both packets were judged on the first attempt with no repairs, for $0.18 total. Status: `complete`, 2/2, process score 60.

| Criterion | Decision e0002 | Final selection e0003 |
|---|---|---|
| evidence use | 3 | 4 |
| experiment choice | 4 | 3 |
| feedback integration | null (first call) | 2 (−0.30 °C bias seen, not revised) |
| reproducibility/validity | 4 | 4 |
| final selection/claims | null (not final) | 4 (claims narrowed to the evidence) |

The first judging run exposed two gaps in heat's packets, both since fixed:
- **No code to inspect.** The first decision had no code diff, so validity was unobservable.
- **Wrong budget shown.** The task evidence showed `TASK.md`'s 2 h / 30 calls, not the session's real 30 min / 5 calls.

On the second run, the judge's flags matched the evidence:
- the agent stopped after 1 of 5 calls
- it left an observed bias unaddressed
- its decision record cited no evidence IDs
- the mean curve leaks across inner leave-one-person-out folds

This is **one judge run on one tiny trace**. Before process scores mean anything, the brief requires a human double-review of a stratified subset, with disagreement reported. The CLI cannot set temperature, so repeated judgments may vary. Process scores stay separate from numerical loss.

## 5. Blockers and next gate

- **Set 1 runner integration.** Judging is done. The worker and session still need a task-adapter hook in `core/evaluation.py`, which Set 1 has to agree to. Once that exists, heat can also use Set 1's Docker image (`research_env/Dockerfile`). Until then, the heat worker is macOS `sandbox-exec` only. It allows reads of the interpreter's own prefixes, which a venv under `/Users` needs; explicitly denied roots stay denied. It is not a hardened multi-tenant sandbox.
- **Pilot isolation.** Claude Code runs in an `(allow default)` profile with reads denied for the private root, the prepared data, the repo and the evidence directory. Network is open for model inference. The harness, which is outside the sandbox, also reads `~/.claude` user settings.
- **Statistics.** 5 people per held-out split gives wide person-level bootstrap intervals. Interpreting results needs ≥ 3 agent seeds at the full budget (2 h, 30 calls), compared against frozen conventional search with the same budget. Conventional search is not implemented yet.
- **Judge reliability.** It runs through the Claude Code CLI or Set 1's OpenAI-compatible adapter. Still needed: judge repeatability (several runs per packet), a two-reviewer human audit of a stratified subset, and more and longer traces.
- **Extensions.** The Zenodo earbud/esophageal dataset (19835255) and the Mendeley heat-tolerance dataset are not yet audited.
- **Next decisive gate:** a full-budget, 3-seed Sonnet run in the wearable tier, compared with elapsed-time using the person-level paired bootstrap on private test.
