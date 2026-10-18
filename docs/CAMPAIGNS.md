# Linked research campaigns

This extends the seven benchmark components in EPISODES.md: data audit, episode specification, evaluator, numerical competitors, workspace, trace judge and pilot comparisons. PLAN.md retains the sensor/product concept. This implementation adds the previously specified Build → Investigate → Transfer → Sustain progression on top of that infrastructure.

## Participant walkthrough

You begin with one complete short-bout calibration, a blank predictor interface and a short-bout development request. You choose a model structure, implement fitting/forecasting, and test with the shared budget. A checkpoint locks your candidate for private evaluation and opens the next research round without ending your workspace.

| Round | Newly available problem | Available evidence choices | Research decision |
| --- | --- | --- | --- |
| Build | Short activity followed by recovery | Initial calibration | Invent an executable predictor and uncertainty outputs |
| Investigate | Longer activity reveals a slower recovery component | Longer-bout recording or redundant short-bout recording | Decide whether parameter fitting is enough or model structure needs to change |
| Transfer | A carried-load condition has different response dynamics | Condition-matched recording or redundant short-bout recording | Adapt to new workload without destroying earlier performance |
| Sustain | A later session has changed response dynamics | Later-session recording or redundant short-bout recording | Retain useful candidates/evidence and adapt while preserving prior conditions |

Every acquired recording already exists. Descriptions expose actual recorded context—activity duration, carried load and session day—but never generator equations or parameters. The rest condition and pre-activity baseline provenance are declared. The task forecasts recorded observables, not diagnoses or latent full-body physiology.

The supplied cases are **explicitly synthetic structural fixtures**. They combine fast/slow exponential response components with context-dependent changes. Their purpose is to prove that representation, experiment selection and retention can matter. They are not validated simulations of human physiology. The conventional grid-search competitor includes the fixture's model family and can solve it; this is a useful reference and no claim of an unsaturated benchmark.

## Persistent contract

A campaign declares all four stages and each session's `release_stage` and `condition`. Complete initial/support or acquired recordings are visible only when released and permitted. Development requests accumulate across rounds, so evaluation reports both current and earlier-condition error. Final-test histories, labels and future-stage menus remain private throughout the participant interaction.

An episode shares 3 acquisition credits and 8 development calls across all rounds. Previously acquired evidence stays available. The participant's code, candidates, action cap and trace persist; reversion and checkpointing never refund resources. The acquisition-policy ablation uses the same menu and budget but chooses redundant calibration. There is no forced minimum tool-call count or mandated equation.

The existing edit/evaluate/acquire/revert operations remain. `checkpoint` freezes executable source and privately evaluates it using only calibration available at that time. It advances one round and returns only a receipt and the next round's name. Only Sustain permits final `submit`. Forecast-only legacy APIs are rejected for campaigns. A participant may advance with a poor model; private validity/error panels record its failure rather than selecting a winner for it.

Each private checkpoint evaluates the candidate on independent test recordings for all conditions. Earlier conditions measure retention; future ones measure forward generalization. These scores do not return to the participant or select actions. The final evaluator uses all four conditions and equal condition weights. Each condition's session errors are averaged, then conditions are averaged. The first fixtures contain one person and one test session per condition; population aggregation is not established.

Private reports include a checkpoint-by-condition matrix and prior-condition error changes relative to the previous checkpoint. Positive changes mean regression. Invalid checkpoints remain explicitly invalid, with no fabricated error value or retention delta.

## Trace judging

Every campaign action captures its public evidence snapshot before execution. Judge packets use those snapshots, preventing later edits to public files from leaking future-round data into earlier decisions. Current code artifacts are content-hash verified; the action's own result and private checkpoint scores remain excluded. External judge verdicts and expert agreement remain separate empirical work.

## Run

```sh
python3 -m benchmark campaign-demo .benchmark-runs/linked-demo --cases 3
python3 -m benchmark create-campaign .benchmark-runs/agent-campaign --seed 0
python3 -m benchmark agent .benchmark-runs/agent-campaign/private --max-steps 48
```

`campaign-demo` runs persistence, pooled structure search, recent-only structure search, contextual structure search and a redundant-acquisition ablation. These are fixed conventional policies, not LLM runs. All use the same initial evidence, menus, credits, development-call schedule and hidden panels. At each new stage they evaluate before acquisition and after acquisition. The source stays fixed while permitted within-episode fitting changes parameters; contextual search also chooses between one/two-timescale representations through its fitting grid.

The CLI emits a report path and aggregates. Reports, private scores, candidate snapshots, judge packets and per-condition development feedback remain in the chosen output directory. Use a new directory per run. macOS executor limitations documented in EPISODES.md still apply.

## Next empirical gates

The existing real-data audit remains valid as an availability audit, not proof that these task families are supported by real recordings. Build real campaigns only after auditing whole source-session identities, workload/context labels and repeated-session coverage. Keep real and synthetic panels separate.

Run multiple fixed-model agents on development campaigns to diagnose structural choices, acquisition efficiency, retention and selection. Hold out meaningful generator families/real conditions for any generalization claim; these three fixtures vary parameters within the same family. Calibrate feedback granularity, uncertainty and budgets on development cases before freezing a scored release. Richer fixtures and passing tests alone do not establish benchmark difficulty or post-training value.
