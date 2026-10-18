# Eight-model, three-repeat physiology comparison

Authorized October 18, 2026: the user approved beginning the recommended eight-model economy comparison with a $16 remaining-spend ceiling, High for Kimi/DeepSeek and Medium for the other six. The earlier five-model pool is paused and excluded from this matched pool because its provider settings, run configuration and source freeze differ. Its seven sealed episodes remain intact.

| Participant | Effort | Selected endpoint | Episodes |
|---|---|---|---:|
| Gemini 4 Argon | Medium | google-ai-studio/flex | 9 |
| Claude Sonnet 5.5 | Medium | anthropic | 9 |
| GPT-6 Astra | Medium | openai/flex | 9 |
| Claude Opus 5.5 | Medium | anthropic | 9 |
| Grok 4.7 | Medium | xai | 9 |
| Gemini 3.8 Flash | Medium | google-ai-studio/flex | 9 |
| Kimi K3 | High | relace/fp4 | 9 |
| DeepSeek V4 Flash | High | parasail/fp8 | 9 |

Each participant runs pulse, heat and load at agent seeds 0, 1 and 2, producing 72 episodes. Seeds are passed into prompts, baseline helper examples, Docker experiment metadata, run identity and API seed when the endpoint supports it. Claude does not advertise API seeds; its recorded trial seeds still distinguish prompts and execution metadata. The biological cohorts are fixed across repeats.

Each episode permits two open-ended Python training programs and development evaluations, followed by final candidate selection and one compact blind process review. The reviewer is GPT-6.1 Sol Medium via openai/flex, with five separately reported criteria. Hidden test outcomes are excluded from its packet. Numerical person-balanced losses determine prediction and uncertainty Elo; process means/counts remain separate. The runner retains exact raw requests and responses, timings, usage, action traces, fitted artifacts, source snapshots, and sealed final receipts.

Normal episode requests: 72 × (3 participant + 1 judge) = 288. Nine compatibility probes bring the normal total to 297. One repair per phase/probe permits at most 594 requests. Every request is reserved before transmission in the same persistent $16 ledger. Unknown billing keeps its conservative reservation.

Under the frozen endpoint prices and request caps, the maximum reservation is $15.11879976: participants $12.84816384, episode judges $1.770192, probes $0.50044392. This includes repairs for every request and full allowed output, including reasoning. Actual usage can be lower. Provider max-price restrictions and disabled fallbacks prevent silent movement onto a more expensive endpoint. Price changes may instead make a route unavailable. The runner stops for an accounting audit if a provider reports a charge above its reserved ceiling.

All requests retain the short protocol limits: participant bodies at most 4,096 UTF-8 bytes, output at most 1,536 tokens including reasoning; judge bodies at most 3,072 bytes and output at most 1,025 tokens. Input reservations add 4,096 tokens of overhead. These output limits can truncate reasoning or code; failures stay in the pool and cannot be read as unconstrained capability estimates. Economy endpoints may have more latency or reduced availability. Provider and quantization differences are explicit in each configuration and the pricing snapshot. Passing a compatibility probe establishes tool-response compatibility, not a successful physiology artifact.

Full held-out cohorts, task definitions and offline Docker isolation are preserved. Pulse ECG references remain provisional. Three agent repeats help inspect agent variability; population generalization, independent-cohort replication and process-judge calibration remain open.

Plan/configuration snapshot: `leaderboard/sweeps/compact-eight/`. Suite freeze: `.leaderboard-runs/compact-eight-suite.json`. Output: `.leaderboard-runs/compact-eight/`. The root status file records completion counts and spending at episode boundaries; the live ledger also includes currently reserved requests. The final `LEADERBOARD.md` includes suite and per-task prediction Elo plus process-review means/counts. `leaderboard.json` includes individual losses, uncertainty Elo, cases and review evidence.

```sh
.venv-research/bin/python -m research_env.compact \
  leaderboard/sweeps/compact-eight/plan.json \
  .leaderboard-runs/compact-eight
```

The command reads `OPENROUTER_API_KEY` from the process environment. An existing ledger and unchanged fingerprint are required for safe resume; it never resets prior spend or silently changes the frozen roster.

## Accounting recovery after a provider limit violation

The user requested continuation after the xAI endpoint reported 72,797 completion tokens for a 1,536-token request. That request cost $0.438316 and remains charged to the original $16 ledger. The original $15.11879976 whole-run estimate assumed provider compliance and can no longer guarantee completion. The frozen plan and earlier receipts are retained as originally recorded.

`scripts/resume_compact.py` makes an explicit accounting amendment without changing participant requests or the frozen numerical environment. It reserves Grok's advertised 450,000-token provider ceiling before each Grok call, and rejects a reported completion count above the original requested limit. It reuses the same ledger, output paths, cached probes, phase attempt limits and 12 sealed episodes. `.leaderboard-runs/compact-eight/accounting-amendment.json` records the script hash and scope; `accounting-overrun-original.json` preserves the interrupted request record before its explicit failure status was repaired. No prior charge or failed attempt is discarded. The root status `worst_case` field retains the original plan calculation; consult the accounting amendment and live spend ledger for the revised reservation policy.

## Parallel orchestration

At the user's request, `scripts/parallel_compact.py` now runs up to three episodes concurrently. API requests overlap while a process-wide semaphore serializes Docker execution so the frozen per-worker CPU/memory allocation is preserved. Each episode owns its private state and SQLite connection. All workers reserve costs against the same original spend database using its transactional budget check. The original runner must exit and release `runner.lock` before the parallel coordinator takes ownership. Completed receipts and validated judge reports are skipped. `parallel-amendment.json` binds this orchestration to its implementation hash; final exported results include both accounting and orchestration amendments. Live status lists `active_episodes` and updates every two seconds.

The user subsequently requested a fourth concurrent episode. The coordinator now supports four workers, preserves the previous three-worker amendment by hash, and records its PID and update timestamp in live status. It runs as a detached process with `coordinator.log` and `coordinator.pid` so chat-turn/session exits do not own its lifetime. Docker concurrency remains one and all workers retain the original shared ledger.
