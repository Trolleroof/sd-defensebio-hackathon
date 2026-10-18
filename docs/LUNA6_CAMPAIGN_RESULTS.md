# Luna 6 full-campaign pilot — October 17, 2026

**Result:** `gpt-6-luna` completed Build → Investigate → Transfer → Sustain, submitted valid executable predictor code, and achieved **0.477766 bpm** condition-balanced MAE on case 0’s hidden synthetic recordings. It placed **second of six** with **1042.5 provisional prediction Elo** (four wins, one loss). No fine-tuning occurred.

## What Luna did

Luna began with a baseline-anchored exponential. Its first `fit` assigned the learned decay locally while prediction used a default value; the short fixture nevertheless scored zero development error. After a long-bout failure, it corrected state persistence and introduced a fitted stretched exponential. It then selected among retained curves by recorded duration, load, and session day. It acquired the matching long, loaded, and later-session diagnostic recordings, using all three credits. It used seven of eight development evaluations, including an unnecessary repeated final evaluation, and completed 18 actions within the 48-action limit.

## Hidden outcomes and retention

These private scores were inspected only after submission. They never went to the participant or action judges. Each checkpoint forecasts all four hidden conditions; future-condition columns measure forward generalization, not agent-visible feedback.

| Checkpoint | Short MAE | Long MAE | Loaded MAE | Later-session MAE |
| --- | ---: | ---: | ---: | ---: |
| Build | 0.0000 | 14.7634 | 21.8801 | 32.5040 |
| Investigate | 0.0000 | 0.7291 | 7.5660 | 17.8490 |
| Transfer | 0.0000 | 0.7291 | 0.8648 | 10.3581 |
| Sustain | 0.0000 | 0.7291 | 0.8648 | 0.3173 |

Earlier-condition errors did not regress at subsequent checkpoints. Final nominal 90% intervals covered all 24 scored future samples; proper interval score was **12.4661**. This tiny synthetic sample does not establish uncertainty calibration. The final public-development MAE was 0.4383 bpm; it is distinct from hidden MAE above.

## Matched-case Elo

Only case 0 is included for every competitor. Existing conventional results were reused for the same seed and task. All six start at 1000; `campaign-elo-v1` uses K=32, scale=400, seed=0 and one complete shuffled round robin. Each competitor has five pairwise outcomes from one underlying case, not five independent trials. Lower hidden MAE wins.

| Rank | Competitor | Hidden MAE (bpm) | Prediction Elo | W/D/L |
| --- | --- | ---: | ---: | --- |
| 1 | contextual_structure_search | 0.0000 | 1070.8 | 5/0/0 |
| 2 | gpt-6-luna | 0.4778 | 1042.5 | 4/0/1 |
| 3 | pooled_structure_search | 9.9551 | 1016.8 | 3/0/2 |
| 4 | recent_only_search | 14.9336 | 986.6 | 2/0/3 |
| 5 | contextual_redundant_acquisition | 17.2869 | 954.4 | 1/0/4 |
| 6 | persistence | 57.1791 | 928.9 | 0/0/5 |

Uncertainty Elo has the same ordering and values on this case. Ratings depend on the opponent pool and update order, have no estimated confidence interval, and cannot be compared directly with the earlier three-case table. The contextual search baseline includes the fixture model family; this is a strong reference, not a frontier-model comparison.

## Independent LLM trace judging

All **18 action packets** received semantic LLM grades from **18 fresh independent Codex judge subagents** using the inherited default model (the tool does not expose its exact service model identifier). Each judge received only one blinded packet, the frozen 1–5 rubric, and evidence available before its action. Judges did not receive action outcomes, later trace packets, private predictive scores, or participant identity. Judges were not allowed to coach Luna. Structured verdict hashes, indices, panel shapes, and evidence references validated successfully; the separate mechanical trace audit passed. No human agreement or repeated-judge calibration audit has been performed.

| Judge panel | Mean across 18 actions (1–5) |
| --- | ---: |
| evidence_use | 4.89 |
| experiment_choice | 4.44 |
| feedback_integration | 4.22 |
| action_consistency | 4.50 |
| final_selection | 3.06 |

The frozen rubric assigns **3 when a criterion is unobservable**. In particular, final selection is unobservable on the first 17 actions; its all-action mean should not be mistaken for the final-submission grade. The actual submission received: evidence_use 5/5, experiment_choice 4/5, feedback_integration 5/5, action_consistency 5/5, final_selection 4/5.

Judge strengths: accurate evidence references, informative acquisitions, feedback-driven revisions, and retention of successful condition-specific curves. Judge concerns: the initial fit-state bug and unsupported interval behavior; changing persistence and representation together without an ablation; repeating an identical successful evaluation without justification; and insufficiently explicit generalization/calibration caveats and candidate comparisons in the final selection. These are semantic judgments, separate from executable prediction scores and Elo.

## Integration defect and evaluation limits

The initial attempt (`luna6-full-01`) was aborted before any checkpoint or submission because the public brief omitted the required callable and forecast-output schema. Its actions and two preliminary judge verdicts remain locally preserved and are excluded from the official results. A fresh Luna subagent started `luna6-full-02` with a public interface supplement, no competitor solutions, and full original budgets. The supplement states `predict(history, forecast_times_s)`, optional `fit(support)`, and the required forecast fields. At the time of this run the workaround did not repair the default brief. A subsequent change now publishes the versioned contract automatically in default observations and public briefs; the original run evidence remains unchanged.

Participant access to only the protocol and participant files was enforced by instruction, not a hardened whole-agent filesystem boundary. Submitted predictors ran in the existing macOS sandbox. The complete case is one synthetic participant in one generator family; matched-context hidden recordings do not demonstrate transfer to untouched families, real sessions, or physiological validity. No trainer integration, weight update, training gain, or stable model ranking was established. Token/cost accounting and explicit decoding settings were not exposed by the subagent tool.

## Verification and next gate

`python3 -m unittest discover -s tests -q` passed **50 tests** in this checkout. The earlier supplied 53-test count was not reproduced. All 18 judge verdicts validated, submission execution was verified, and the mechanical chronology audit passed.

The public execution contract is now included in default observations (53 tests pass after three regression tests were added). Next: repeat fixed-model campaigns across development cases and reserve untouched structural families/real sessions before tuning. The full LLM campaign integration gate is now demonstrated once; meaningful difficulty and validated training readiness remain open.

## Local evidence

- [Official run metadata](../.benchmark-runs/luna6-full-02/run-metadata.json)
- [Matched comparison and raw scores](../.benchmark-runs/luna6-full-02/comparison.json)
- [Ratings and every pairwise update](../.benchmark-runs/luna6-full-02/ratings.json)
- [Validated LLM judgments and mechanical audit](../.benchmark-runs/luna6-full-02/judging.json)
- [Judge summary](../.benchmark-runs/luna6-full-02/judge-summary.json)
- [Chronology-bounded judge packets](../.benchmark-runs/luna6-full-02/judge-packets.json)
- [Submitted predictor](../.benchmark-runs/luna6-full-02/private/submission.py)

Run artifacts are ignored by Git and available in this local workspace; this report does not make them a published reproducible release.
