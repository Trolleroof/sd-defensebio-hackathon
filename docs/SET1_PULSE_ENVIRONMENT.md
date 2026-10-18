# Set 1: pulse research environment

This implements Set 1 of `AGENT_BUILD_INSTRUCTIONS.md` in `research_env/`. It leaves the existing recovery benchmark and Sets 2/3 unchanged. The shared interface is `physiology-research-v1`; task version is `galaxy-pulse-causal-v1`. Existing recovery contracts are not substituted for waveform training.

The environment contains a GalaxyPPG importer, causal request construction, four CPU baselines, person/bout-balanced scoring, an offline isolated command worker, a persistent research controller, immutable artifact snapshots, streaming artifact replay, checkpoint receipts, trainer-neutral trajectory export, and a provider-backed physiology judge with packet import/export. The participant retains freedom to choose its estimator and experiments.

## Current measured evidence

See `SET1_RESULTS.md` and `evidence/set1-summary.json`. Results are retrospective builder measurements with provisional ECG references. The dataset is not approved for benchmark release. No live LLM research agent, live LLM judge, research-agent post-training, clinical effect, or field benefit was demonstrated. The source archive and generated run artifacts are locally retained under ignored `.research-data/` and `.research-runs/` directories.

## Setup and source acquisition

Run at the repository root. Python 3.13 is used locally; the container fixes Python 3.13.3 and its base manifest digest. All numerical dependencies are pinned. The build pins pip 26.2.1 for resumable, checksum-verified wheel downloads after a truncated SciPy transfer was observed. The final built image must additionally be selected by immutable image ID/digest; the base digest alone is not the worker image identity.

```sh
uv venv --python 3.13 .venv-research
uv pip install --python .venv-research/bin/python -r research_env/requirements.lock
.venv-research/bin/python -m research_env fetch .research-data/galaxy
.venv-research/bin/python -m research_env prepare .research-data/galaxy .research-runs/new-pulse
.venv-research/bin/python -m unittest discover -s tests -p test_research_env.py -v
```

Fetch verifies Zenodo's archive size and MD5 and retains a SHA-256, record metadata and full CC BY 4.0 legal text. Never overwrite the immutable original archive. A partial download is an error, not synthetic-data permission. `prepare --splits split.json` accepts an explicit complete person split. Otherwise an audited 23-person cohort uses deterministic 13/5/5 grouping; P01 lacks Galaxy Watch/event files. Read `private/audit.json` for per-person exclusions and clock/rate checks. `benchmark_ready` stays false pending further beat-annotation review and timing validation.

## Baselines and trusted builder profiling

```sh
.venv-research/bin/python scripts/run_pulse_baselines.py .research-runs/new-pulse .research-runs/new-baselines
.venv-research/bin/python -m research_env baseline .research-runs/new-pulse/public/train/manifest.json .research-runs/new-ridge --kind ridge --seed 0
.venv-research/bin/python -m research_env replay .research-runs/new-ridge .research-runs/new-pulse/private/development/manifest.json .research-runs/ridge-dev.json --backend trusted-builder
```

The first command fits all four fixed baselines, freezes a development-only choice in `selection.json`, then reports each fixed baseline on private test people. Both local sandbox and strict-container controller smoke runs have been executed (see the results note). Builder profiling executes trusted baseline code directly; its timings are not a strict-budget participant comparison. Do not use `trusted-builder` with participant artifacts. The median, spectral, motion-aware spectral tracker and compact learned ridge models all receive the same split, windows and channels. Intervals are fitted from training residual quantiles. An explicit calibration manifest must contain training people only.

## Offline container worker

Docker Desktop must be running. Build/cache dependencies before starting the offline scored session:

```sh
docker build -t physiology-research:v1 -f research_env/Dockerfile research_env
docker image inspect physiology-research:v1 --format '{{.Id}}'
```

The explicit container gates can be rerun with `scripts/check_pulse_container.py IMAGE_ID REPORT.json`. They check numerical fitting, unavailable private files, disabled network, wall timeout and a forced memory-limit failure.

Use the returned `sha256:...` as `--image` in the commands below; replace the literal `IMAGE_ID` placeholder. The worker uses `--pull=never`, no network/GPU, a read-only root, dropped capabilities, no new privileges, bounded pids, and CPU/memory cgroups. It mounts only a staged public workspace. Keep controller files/private labels outside every mounted directory. CPU/memory limits are capped by host allocation. Local `--backend macos-development` supports numerical execution, blocks private file reads and network, and enforces elapsed timeout; it does not enforce hard memory/core limits and is explicitly ineligible for resource-compliant scoring.

## Participant and conventional-search runs

Use `scripts.search_pulse.stage(public_root, workspace)` to stage public signals, baseline training code, task instructions, and dependency lock. It excludes the trusted evaluator, controller, judging implementation and private labels. The shared controller's `ResearchEnvironment` exposes edit, bounded command, development evaluate, select and submit actions. An external authorized provider may drive `environment.run(provider)`, where `provider(observation)` returns an action. CLI JSON-lines mode is:

```sh
.venv-research/bin/python -m research_env agent RUN/private RUN/workspace DATASET/private/development/manifest.json --image IMAGE_ID
```

The controller receives private labels; the worker never does. Do not give participant code host shell access to this controller invocation. Route observations/actions through an authorized model client. No model client is auto-discovered and no paid inference is started by these commands.

A frozen conventional grid uses identical command, replay, wall-time and evaluation limits:

```sh
.venv-research/bin/python scripts/search_pulse.py .research-runs/new-pulse .research-runs/grid-run --image IMAGE_ID --calls 30
```

The grid declares median, spectral, five motion weights and 23 ridge penalties before seeing feedback. `--calls 1` is a wiring smoke, not equal-budget search evidence. Three agent seeds and matched full-budget comparisons remain pilot work once inference is authorized. Source snapshots, commands, evidence, code-change descriptions, seeds, stdout/stderr, elapsed time, local sampled RSS, outcomes and final artifact hashes are retained. Docker peak RSS reporting remains unavailable; local peak RSS is sampled rather than an exact maximum.

Evaluation copies only the artifact and a minimal entrypoint into a fresh worker per bout. It streams one causal request at a time and waits for its response before exposing the next. Raw full sessions, file/person IDs, activity labels and ECG references never enter replay. State resets between bouts and people. Missing/extra/nonfinite outputs invalidate the whole attempt.

## Private replay and checkpoints

```sh
.venv-research/bin/python -m research_env replay RUN/private/candidates/CANDIDATE_ID DATASET/private/test/manifest.json RUN/private/final-score.json --image IMAGE_ID
.venv-research/bin/python scripts/score_pulse_checkpoints.py RUN/private DATASET/private/test/manifest.json RUN/private/checkpoint-scores.json --image IMAGE_ID
.venv-research/bin/python -m research_env trajectory RUN/private RUN/private/trajectory.jsonl --outer-split agent_development
```

The controller freezes the declared candidate at 15/30/60/120 minutes. Commands poll these boundaries; idle boundaries are recorded before the next action can change the selection. Checkpoint replay is a separate private operation and returns no participant feedback. A missing incumbent is an explicit invalid checkpoint. Final score uses the declared submitted candidate, never a private-score winner. Final outer evaluation tasks cannot be used for training rewards.

## LLM judging

```sh
.venv-research/bin/python -m research_env judge-export RUN/private RUN/private/judge-packets.json
.venv-research/bin/python -m research_env judge-import RUN/private/judge-packets.json RUN/private/judging.json
.venv-research/bin/python -m research_env judge-import RUN/private/judge-packets.json RUN/private/judging.json --verdicts authorized-verdicts.json
```

Without verdicts, status is `pending`. Decision packets exclude the action's outcome and all later evidence; final-selection packets include only permitted public history. Controller-injected identities/private test values are omitted. Embedded source/log/report text is untrusted evidence. Packets include the frozen rubric, anchors and required JSON shape. Verdict import verifies the packet digest, event identity, exact criteria, integer scores/nulls, nonempty reasons and allowed evidence IDs. No criterion gets a default score. Missing observations or unresolved verdicts yield incomplete coverage, not fabricated ratings. Process scores stay separate from numerical model loss.

An actual HTTPS OpenAI-compatible adapter is implemented. To invoke it after obtaining provider authorization, create JSON configuration with `endpoint` (chat-completions URL), exact `model`/version, `temperature`, `max_tokens`, `max_calls`, and `max_total_tokens`; explicitly supply the key variable name:

```sh
.venv-research/bin/python -m research_env judge-run PACKETS.json authorized-judge-config.json NEW_JUDGE_OUTPUT --key-env PHYSIOLOGY_JUDGE_KEY
```

The key is never written to logs. Usage, configuration, rubric/prompt/packet hashes and every attempt/verdict are logged; failed calls count. Initial attempt plus at most two retries use the same evidence and fresh contexts. Mock tests prove wiring only. Human double-review, disagreement and grading reliability remain separate release gates.

## Ownership and remaining gates

Set 1 owns `research_env/core/`, `research_env/judging/`, `research_env/tasks/pulse/`, shared CLI/container/lock, focused checks and pulse scripts/docs. Heat and load builders should reuse the worker, session and judge contracts without rewriting them. Their request/output names and aggregation require task adapters; the current evaluator is pulse-specific and must not silently score another task as pulse.

Next decisive scientific gate: audit a larger stratified set of ECG beat annotations, especially noisy P11 and movement, and independently verify clock alignment before freezing reference masks/splits/baseline thresholds. The six plotted builder-inspected excerpts do not substitute for human-reviewed annotation accuracy. After that, run an authorized unmodified research-agent pilot and judge pilot, with at least three seeds when budget allows. Post-training is outside this brief.
