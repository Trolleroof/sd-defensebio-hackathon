# DefenseBio: sensors, SoliderPhys-Bench and structural campaigns

The shared project contains these connected implementations:

- [Sensor capture](sensor/): live Polar Verity Sense BLE streaming.
- [Operator dashboard](dashboard/README.md): team views and response tracking.
- [SoliderPhys-Bench](PHYSBENCH.md): the existing PPG-DaLiA real-data agent environment, Linux execution, provider loop and separate judge.
- [Structural research campaigns](docs/CAMPAIGNS.md): local synthetic Build → Investigate → Transfer → Sustain tests and Elo summaries.

The `physbench` and `benchmark` packages are complementary runners with different data/execution contracts. The structural fixtures do not replace the real-data pipeline. Both are packaged; `physbench-benchmark` is equivalent to `python -m benchmark`.

# Physiology research-agent benchmark

The full sensor/product and benchmark concept remains in [PLAN.md](PLAN.md). Implementation here focuses on the benchmark: personalized recovery forecasting, a persistent research workspace, bounded experiment tools, executable submissions, and independent trace judging.

## Run

Python 3.10+ and the standard library are sufficient for the controller. Predictor execution currently requires macOS `sandbox-exec`; unsupported hosts fail closed. This prototype blocks declared private roots and network access but permits reads elsewhere for Python startup, so it is not a complete filesystem sandbox. Docker/remote backends are not implemented. No service deployment or API credentials are required for the synthetic examples.

```sh
python3 -m unittest discover -s tests -v
python3 -m benchmark demo .benchmark-runs/demo-01
```

Use a new output directory each run. The demo writes `report.md`, `report.json`, code artifacts, per-run private scores and judge packets. It compares persistence, fitted linear recovery, development-only candidate selection and a scripted research campaign. These are synthetic smoke tests. The first two baselines use fewer development calls than the search/revision runs; all receive the same final calibration evidence and maximum budgets.

## Connect a model

```sh
python3 -m benchmark create .benchmark-runs/session-01
python3 -m benchmark agent .benchmark-runs/session-01/private
```

The second command serves a JSON-lines protocol: one initial public observation, then one JSON action on stdin, followed by its result and the next observation on stdout. A trusted orchestration process may connect its model client to these pipes. Models receive observations and emit actions; they do not receive controller paths or host shell access. An action includes `operation`, `decision`, and the fields required by the observation's protocol. Model inference configuration, token costs and provider credentials belong to the caller; no hosted model is invoked automatically.

For example, edit a candidate with `{"operation":"edit","code":"...","decision":{"evidence":"support.json","hypothesis":"...","action":"...","expected_result":"..."}}`, then evaluate, acquire an available recording, revise/revert, and submit. The Python `AgentSession.run(provider)` API supports a callable observation-to-action provider. Both successful and rejected calls consume the persistent action cap; evaluation and acquisition have additional explicit budgets. A final submission closes feedback access and returns a receipt only.

## Bring recordings

```sh
python3 -m benchmark audit-csv recordings.csv --origin local --dataset pilot --target-provenance 'device-derived HR'
python3 -m benchmark create-csv recordings.csv .benchmark-runs/real-01 --origin local --dataset pilot --target-provenance 'device-derived HR' --participant p01 --episode-id pilot-p01
```

Required CSV columns: `participant_id,session_id,time_s,hr_bpm,role`. Roles are support, acquisition, development and test. Each session ID represents one complete source recording and can have only one role. Relative times reset per session. Episode v0 requires one participant with support, development and test sessions, each including the declared cutoff at zero and later targets. Caller-supplied rest/activity alignment and independent source-session identity must be audited; the adapter does not infer them from heart-rate values. Public data also requires `--source-url`.

See [the data audit](docs/DATA_AUDIT.md) for primary-source coverage and target provenance. One actual PhysioNet recording is retained as an input sample, not a scored episode.

## Judge and inspect

```sh
python3 -m benchmark judge-packets .benchmark-runs/session-01/private packets.json
python3 -m benchmark import-verdicts .benchmark-runs/session-01/private verdicts.json judging.json
```

The verdict file is a list containing one structured judgment per packet. See [JUDGING.md](docs/JUDGING.md). Reports explicitly say pending when external judgments have not been supplied; a deterministic trace audit is not an LLM judgment. Freeze and record the judge model/settings and manually audit verdicts before interpreting them as reliable research-quality measurements.

See [EPISODES.md](docs/EPISODES.md) for contracts, examples and remaining limitations. The current task is recovery-only, with no missing-value target support. Benchmark difficulty, calibrated uncertainty, population generalization and training benefit remain open empirical questions.

## Linked research campaigns

The extended campaign carries one workspace through Build, Investigate, Transfer and Sustain. It releases evidence by round, preserves the shared budget, permits checkpoint submissions and privately measures retention across earlier conditions.

```sh
python3 -m benchmark campaign-demo .benchmark-runs/linked-demo --cases 3
python3 -m benchmark create-campaign .benchmark-runs/agent-campaign --seed 0
python3 -m benchmark agent .benchmark-runs/agent-campaign/private --max-steps 48
```

See [CAMPAIGNS.md](docs/CAMPAIGNS.md) for the LLM walkthrough, staged evidence, structural fixtures, acquisition choices and checkpoint scoring. These conventional comparisons exercise structural/adaptation tradeoffs; no new LLM performance or scientific validation follows from them.

## Elo leaderboard

Campaign reports now include provisional prediction Elo and export a separate uncertainty Elo table, alongside raw metrics. All models start equally; rank follows matched campaign wins. A fine-tuned checkpoint must earn its position on untouched evaluations. See [RATINGS.md](docs/RATINGS.md) for the frozen match/update/failure rules and limits.

```sh
python3 -m benchmark rate .benchmark-runs/linked-demo/report.json .benchmark-runs/linked-demo/ratings.json
```

## Real-statistics campaign pilot

The PPG-DaLiA-grounded synthetic generator and four-stage campaign pilot are described in [the generator and fidelity report](docs/SYNTHETIC_DALIA.md) and [the development results](docs/DALIA_CAMPAIGN_RESULTS.md). These small development results do not establish physiological validity, clinical performance, untouched-participant transfer, or training benefit; the pilot did not update model weights.

## Pulse research environment (Set 1)

The causal GalaxyPPG waveform task and shared CPU research runner live in `research_env/`. See [setup, participant protocol and reproduction commands](docs/SET1_PULSE_ENVIRONMENT.md) and [executed results and remaining gates](docs/SET1_RESULTS.md). This is separate from the recovery prototype and uses provisional ECG references.
