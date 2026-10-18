# PPG-DaLiA-grounded campaign pilot — October 17, 2026

**Outcome:** the new data runs through the persistent four-stage campaign. All three Luna 6 participants submitted valid predictors. Fifteen conventional campaigns and 108 frozen-source real-development transfer evaluations completed. Luna did not outperform simple baselines, and synthetic calibration did not demonstrate a real-data advantage. This is a completed development pilot, not a validated training-environment milestone.

## Fixed development protocol

Profile `dalia-activity-campaign-v1`: synthetic SYN0006/SYN0007/SYN0008; real development S3/S7/S10. Synthetic generator fitting used only the reference cohort. Six real final-test participants were neither loaded nor scored. Each source recording supplies sitting → walking → stairs → cycling stages. All windows have 60s history and 60s horizon at two-second resolution, separated by eight-second gaps. Split selection uses coverage alone, not prediction results. Calibration, development and private windows are nonoverlapping, but come from the same recording and are temporally dependent; they are not independent sessions or participants. The supplied activity remains constant through each forecast. This deliberately narrower profile is not the existing SoliderPhys-Bench 120s future-covariate profile.

Agent inputs are past HR, wrist/chest movement summaries, activity and skin-temperature history. Future HR, movement and temperature measurements are not given to the agent or predictor. Chip temperature is excluded. Predictor interface and score contract are explicit. No model family or initial predictor is supplied to Luna. All competitors share initial data, three acquisition credits, eight development evaluations and 48 action attempts. Conventional policies are fixed, and all acquire matched recordings. Conditions are equally weighted. Sustain tests retention while adding cycling; it does not claim a separate later-day session.

Three separate `gpt-6-luna` contexts each completed one case. An account usage interruption was resumed using persisted state without resetting budgets. No further stochastic repetitions on identical cases were run. Full-agent access boundaries were instructional, not OS-enforced; generated predictors ran in the existing macOS sandbox. Token/cost accounting and exact decoding settings are not available.

## Accuracy and real-development replay

| Competitor | Synthetic MAE (bpm) | Real, synthetic calibration | Real, permitted real calibration |
| --- | ---: | ---: | ---: |
| pooled_ar | 4.701 | 5.968 | 5.288 |
| persistence | 5.038 | 5.448 | 5.448 |
| gpt-6-luna | 5.401 | 7.648 | 6.636 |
| damped_trend | 5.941 | 7.713 | 7.713 |
| contextual_arx | 6.050 | 16.607 | 8.750 |
| contextual_ar | 6.172 | 16.026 | 5.305 |

Transfer replays each frozen source from three synthetic cases on each of three real development cases (nine source/case combinations per competitor), under two calibration modes. Synthetic calibration supplies the same permitted synthetic support/acquisitions; real calibration supplies only designated real calibration windows. No real development or private future labels are included in fitting. Source code is not revised or selected using real outcomes. HR history from each real request remains available in both modes. These nine outcomes reuse three real recordings, not nine independent people. No final-test score or training reward was used.

Conventional competitors: persistence, damped recent trend, pooled scalar AR(1), per-activity AR(1), and per-activity ridge autoregression with movement/skin-temperature covariates. The ARX competitor holds observed covariates fixed through its autonomous rollout; that is a baseline modeling assumption, not a future-observation advantage. This limited competitor set does not establish performance against the strongest possible conventional search.

## Luna cases and research behavior

| Case | Hidden MAE (bpm) | Hidden interval score | Hidden 90% coverage | Final-selection judge |
| --- | ---: | ---: | ---: | ---: |
| 0 | 4.428 | 32.960 | 90.8% | 4/5 |
| 1 | 3.696 | 22.554 | 94.2% | 4/5 |
| 2 | 8.079 | 78.186 | 58.3% | 3/5 |

Case 0 used activity-specific HR baselines and a damped recent trend. A movement-gated modification badly damaged sitting development forecasts, and the agent reverted before submitting. Hidden sitting error nevertheless increased from 2.99 to 3.15 bpm after its initial adaptation; retention is measured, not assumed perfect.

Case 1 used per-activity calibration and added a development-driven +9 bpm stairs adjustment. This improved its development stairs result, but that activity-specific offset has weak physiological justification and is a transfer risk.

Case 2 converged to smoothed terminal-HR forecasts and did not use movement or temperature. It submitted an untested 6.5-bpm interval-width compromise between two tested settings. Its public development MAE was about 3.81 bpm; hidden MAE was 8.08 bpm with only 58.3% coverage. This exposes development-to-private generalization and uncertainty weaknesses.

All 58 Luna action packets are exported with chronology-bounded evidence. Independent blinded LLM judges graded the three final-action packets: selection grades 4/5, 4/5, 3/5; structured hashes, evidence references and scores validated. Full-trace semantic judging is still pending and is not replaced by these final-only grades. Mechanical audits are saved separately. No human or repeated-judge agreement audit was performed. An initial case-2 judge accidentally exposed earlier packets, disclosed it, and was replaced by a fresh judge with a file containing only the final packet; the contaminated judge was excluded.

## Provisional prediction Elo

Every competitor begins at 1000; K=32, scale=400, schedule seed=0. One match compares two competitors on the same full case. These 15 pairwise results per competitor derive from three synthetic recordings, not 15 independent trials.

| Rank | Competitor | Elo | W/D/L |
| --- | --- | ---: | --- |
| 1 | pooled_ar | 1046.8 | 10/0/5 |
| 2 | persistence | 1040.0 | 9/0/6 |
| 3 | gpt-6-luna | 1007.1 | 8/0/7 |
| 4 | contextual_ar | 992.4 | 7/0/8 |
| 5 | damped_trend | 969.6 | 6/0/9 |
| 6 | contextual_arx | 944.1 | 5/0/10 |

Elo is descriptive for this opponent pool, protocol and small development batch; no confidence interval is estimated. It is not comparable numerically with the earlier toy-campaign rating. Real-development raw errors are reported separately and do not become synthetic Elo matches.

## Decision and next gate

The bridge and comparative development runs are implemented and exercised. The results do not yet validate the generator as a useful training distribution: synthetic calibration transfers poorly, simple HR-history methods remain competitive, and no reliable benefit from additional movement or skin-temperature inputs is demonstrated. Do not start a large training run on the assumption that these cases reward the intended research skill.

Next, diagnose activity-conditioned level/transition and low-frequency mismatches in the generator using reference/development recordings. Include activity transitions and bounded recovery forecasts rather than only steady-activity windows. Keep prediction evidence sufficient and maintain an untouched evaluation pool. Re-run fixed baselines first, then repeated model trials. Full-trace judging, judge agreement, reward/trainer integration and actual weight updates remain open.

## Reproduction and artifacts

Create a case with `python3 -m benchmark.dalia_campaign PROCESSED SUBJECT OUTPUT --origin synthetic --case 0`. Real creation accepts only S2/S3/S7/S10, rejecting final subjects before loading. Creation fails explicitly if the source lacks disjoint coverage. Existing structural campaigns remain supported.

Use `PYTHONPATH=. python3 scripts/run_dalia_pilot_baselines.py RUN_ROOT` for fresh baseline directories and `PYTHONPATH=. python3 scripts/run_dalia_pilot_transfer.py RUN_ROOT` to replay available submitted sources. The latter overwrites its development report intentionally; inputs and protocol remain frozen. LLM participant orchestration is saved locally, not a provider-API reproducibility claim.

Local artifacts under `.benchmark-runs/dalia-campaign-pilot-01/`: `protocol.json`, source case manifests, `comparison.json`, `ratings.json`, `real-transfer-report.json`, `final-selection-judgments.json`, and each participant’s code, decisions, checkpoints and judge packets. Bulk artifacts remain ignored by Git.

**Verification:** 60 tests pass. New tests check source-window separation, causal public views, missing-coverage rejection and real final-subject exclusion. All final submissions executed validly. No model weights or hardware firmware changed.
