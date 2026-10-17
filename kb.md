# DefenseBio Hackathon — Unified Knowledge Base

**Last consolidated:** October 17, 2026 (America/Los_Angeles)  
**Canonical detailed plan:** [PLAN.md](PLAN.md), current on main. This KB combines the earlier Slack/meeting notes with the current project direction and confirmed event context. Use the plan for full product, benchmark, experiment, and scoring specifications.  
**Source standard:** Distinguish organizer-confirmed facts, team proposals, and unverified assumptions. Slack search/history can be incomplete; follow the source links before acting on changing logistics.

## 1. Current direction

### Product

Help a medic or team operator follow several people when direct assessment is delayed. The operator needs to know **who needs a check, what changed, and what response has already happened**.

The product loop is:

1. Capture supported wearable measurements and manual events.
2. Show the person’s current evidence, data quality/freshness, movement/activity, and pulse trend.
3. Compare the observed response with a clearly labeled experimental personal exertion/recovery predictor.
4. Put an explainable reason in an attention queue for operator review.
5. Let the operator request a check, acknowledge it, record a response, and see reassessment.

The primary user is an operator/medic, not an autonomous medical device. A deviation from a predictor means “check the person or the measurement”; it is not a diagnosis. Any physical output in a demo must be harmless and human-approved.

### Event fit

- **Lead track:** Human Performance & Prolonged Field Care.
- **Secondary fit:** Biological Data for Defense, through synchronized wearable data and executable modeling.
- **Official prompt match:** “Operational Phenotyping with Minimal Sensors” asks for useful operational-health signals while minimizing device burden (wearing, charging, and reliability). Signal processing and biohardware optimization are suggested demo types.
- **Demo principle:** show a tangible sensor → visible evidence → operator action → reassessment loop. A model or dashboard alone is not proof that a complete product loop works.
- Event direction, track prompts, and demo guidance are in the [organizer build announcement](https://2026sddefense-dcs5018.slack.com/archives/C0C3LG6PKRB/p1790278909686629).

### Research

Test whether an AI agent can build, transfer, and update a small executable model of an individual’s exertion and recovery using the same allowed inputs as the wearable product. This is a proposed benchmark, not an implemented, calibrated, or validated result.

The decisive early comparison is an agent versus a directly fitted personalized model, with the same permitted data and fitting resources, on an untouched session. If the agent does not improve prediction or useful adaptation, report that and revisit the task before investing in a large training run. Fine-tuning method, training recipe, and reward mapping remain decisions for Nikhil.

## 2. Project history and decision trail

Earlier brainstorming explored three broad directions:

1. **Fine-tune an open U.S. model to outperform Chinese models in an RL environment.** This was judged the weakest initial project fit without a task-grounded product and evidence.
2. **Protein-binding design/RL.** Potentially relevant to countermeasures, vaccines, or delivery, but dependent on substantial protein-design expertise and validation.
3. **Distributed phenotyping with minimal sensors.** Compare wearable channels and models to learn what can be inferred from limited, practical signals. This fit the event prompt best.

A later brainstorm combined phenotyping with a soldier-status dashboard and imagined casualty detection, geolocation, extraction prioritization, and targeted protein/stimulant delivery with a human confirmation button. Those are **historical concepts, not the current MVP**. The current scope narrows to an operator-reviewed exertion/recovery monitoring loop and a model-building benchmark. Therapeutic delivery, injury diagnosis, autonomous casualty triage, and location tracking are not demonstrated or supported by the current sensors. Do not imply the wearable can safely trigger drug delivery.

Earlier notes suggested heat-casualty warning during training and preferred EDA/GSR plus skin temperature. These remain possible future research questions, not validated signals or the selected first build. “CO2” in the inventory is an ambient air measurement; it does not directly measure blood oxygen conversion. One PPG stream does not establish complete physiology.

## 3. Hardware: inventory versus actual access

The organizer posted a rough onsite inventory on October 16 with an attached workbook, DefenseBio_Hackathon_Hardware_Final_v2 (1).xlsx (Slack file F0C9SS9HCM9). The workbook is evidence of what was listed, **not** that our team received, reserved, or tested any item. The organizer said onsite hardware is first come, first served in event logistics messages; team access remains to be checked. Teams’ own hardware requires registration approval. One shared four-legged robot with an arm/claw was announced, but the product does not depend on it.

| Workbook item | Proposed role | Open verification |
| --- | --- | --- |
| SparkFun Pulse Oximeter & Heart Rate Sensor, Qwiic **SEN-15219** (row 7) | Pulse/PPG signal for an initial response-prediction task | Physical access, raw/processed output, placement, motion quality. Workbook notes reset/MFIO jumpers beyond Qwiic. |
| Adafruit **LSM6DSOX** STEMMA QT (row 13) | Movement/orientation context | Access, mounting, sampling rate, timestamps, useful activity features. |
| **Polar H10**, Polar Verity Sense, **AD8232** ECG front end (rows 8, 16–17) | Prefer H10 as an independent development reference if allocated | Access, output, synchronization, and signal quality. Verity Sense is optical, not an ECG reference. |
| Arduino Nano ESP32 **ABX00083** or Adafruit ESP32-S3 Feather; buttons, LEDs, power banks, straps (rows 2–3, 37, 44, 47–49) | Capture and manual distress/acknowledgment loop | Exact board received, wiring, power, and connection. Nano has no Qwiic port; workbook lists adapters. |
| SparkFun Qwiic **TMP117** and Grove GSR (rows 10, 12) | Optional observed channels | Access, contact, quality, response time, and a defined validation target before adding scope. |

**Current hardware hypothesis:** SEN-15219 pulse + LSM6DSOX movement + a manual distress button, if working units are available. Prefer H10 as an independent reference if allocated. Keep temperature and GSR optional. Confirm hardware before locking the first prediction target. The workbook’s application ideas (including contactless heartbeat, fever, and tissue oxygenation proxies) are suggestions, not demonstrated performance or clinical capabilities.

Older brainstorming mentioned ECG, PPG, EDA/GSR, skin temperature, CO2, IMU, force-sensitive resistors, 60 GHz radar, thermal camera, VOC/particulate and spectral sensors, night vision webcam, and broad “sensor kit” availability. Treat that as an early/rough list. The October 16 workbook and actual equipment desk confirmation are the current inventory references. Do not assume every channel is available per team.

An EEG availability question was asked October 16; no reply was present in its thread when checked. EEG availability is unknown. [Question thread](https://2026sddefense-dcs5018.slack.com/archives/C0C3TDMLM4Z/p1792199415550999).

## 4. Data, product, and safety boundaries

- Distinguish measured values from inferred states, alerts, and clinical conclusions. A predictor is experimental; do not claim clinical efficacy, injury prediction, treatment guidance, or operational readiness without appropriate validation.
- Missing, stale, disconnected, and poor-quality data must be visible and distinct from normal. Show the measurements and reason behind each alert.
- An IMU alone does not prove injury or consciousness. Skin temperature does not establish core temperature. PPG-derived pulse is not an independent ground-truth reference.
- Keep LIVE, REPLAY, and SYNTHETIC data clearly labeled. Acknowledgment or a harmless indicator does not demonstrate therapeutic effect.
- Preserve participant/session IDs in pseudonymous form, timestamps, units, device and processing versions, validity masks, activity markers, and target provenance. Collect only voluntary, comfortable recordings with appropriate consent. Do not prescribe exertion from model output.
- If independent reference data are unavailable, label the target as a sensor-derived estimate and limit claims accordingly. Never claim the underlying physiology is known during sensor dropout without an independent usable reference.
- Location is optional. Label whether it comes from a live phone, replay, or a manually placed demo marker; GPS-denied capability is not established.
- Product and dashboard execute reviewed predictors. Agent-generated code does not automatically replace a deployed model.

## 5. Benchmark summary

The full environment specification is in [PLAN.md](PLAN.md), sections 2.1–2.14. In brief:

- **Build:** fit a personalized predictor from limited calibration recordings; evaluate on another session from that participant.
- **Transfer:** adapt an existing predictor to a changed activity/workload with limited new evidence; evaluate on separate unseen sessions.
- **Update:** use later recordings while preserving earlier-session performance.
- Build comes first. Transfer needs genuine activity/workload coverage; Update needs repeated sessions. Until those data exist, latter environments are proposals.
- Inputs are frozen at each prediction time. Future targets are hidden. Activity-conditioned tasks and forecasts based only on past values are separate tasks.
- Keep participants and sessions grouped across splits. Separate training, development, and final-test participants. Personalization on explicitly supplied support data is allowed; hidden-session access is not.
- Fit preprocessing only on permitted data; keep complete sessions together; do not split overlapping windows across support and hidden sets; hide target files, future labels, generator seeds, and private manifests.
- Report raw bpm error and execution/resource information. For exertion/recovery episodes, compute MAE for each phase and average phase errors equally; average sessions within participants and participants equally. Keep synthetic and real results separate.
- Fixed numerical baselines, a direct individual fit, and selected agents should use the same episodes and budgets. Preserve first and final submissions and all feedback traces. Test feedback/revision separately from extra compute.
- Synthetic participants can add practice episodes, not independent human evidence. A small dataset supports only a narrow simulator prototype; report a synthetic gain as synthetic until real-session transfer is shown.
- Dataset size, pilot duration, and provisional episode settings must be calibrated using development data. Do not claim an unsaturated benchmark from design alone.
- Candidate reward mappings and LLM trace audits are development ideas, not selected scoring rules. Numerical hidden-test performance remains primary; any quality judge does not override it.
- Related work and novelty still need review. The plan identifies [TimeSeriesGym](https://github.com/moment-timeseries-foundation/TimeSeriesGym) and prior personalized physiological modeling as comparison points. Novelty is a hypothesis, not established.

## 6. Event facts and logistics

- The organizers describe a 24-hour build. The Slack build prompt requests preparation beforehand and says teams can bring hardware subject to registration approval.
- Event project/proposal form: the September welcome message specified one per team by **October 15, 11:59 p.m. Pacific**. That deadline has passed. This message does not establish a final code deadline or prove our team submitted. [Welcome announcement](https://2026sddefense-dcs5018.slack.com/archives/C0C3LG6PKRB/p1790225319014299).
- Slack-based schedule/logistics messages summarized in the earlier KB stated registration at 9:30 a.m. Saturday Oct 17, briefings at 11:00 a.m., build clock noon Saturday through noon Sunday Oct 18, demos at noon Sunday, and final judging at 2:00 p.m. These are event-day details and should not be treated as upcoming. Photo ID and onsite registration approval were stated as entry requirements. Check the [event page](https://luma.com/kfgh2fvu) or organizers for corrections.
- A historical proposal described a three-minute presentation. The current plan explicitly treats that only as a rehearsal format: no official per-team demo duration or scoring rubric was verified in the reviewed Slack searches.
- Judges, current attendance, proposal submission status, equipment allocation, final deliverable instructions, and EEG availability are not established by this KB.

## 7. Current status and next decisive checks

**Verified from reviewed sources:** official track/build prompt; Oct 16 inventory announcement and attached workbook; the EEG question and lack of thread reply at review time; proposal form deadline wording; current PLAN.md on main with the product and benchmark plan.

**Not verified:** receipt or working state of any sensor/board; H10 reference access; data permissions/consent; synchronized pilot capture; participant coverage; predictor performance; implementation/deployment; project-form submission; any separate final deliverable; EEG availability.

1. Confirm and test the actual SEN-15219, LSM6DSOX, board, and H10 availability.
2. Decide whether the first recording includes H10 alongside PPG and movement. That fixes target provenance and what can be scored.
3. Capture one consented, synchronized, comfortable pilot with activity markers; inspect signal quality and timestamps before expanding scope.
4. Build the smallest end-to-end sensor → evidence → operator check/action → reassessment loop.
5. Compare a direct personalized fitter and an agent with the same evidence on an untouched session before expanding training or simulator work.

## 8. Source links

- [Current detailed project and benchmark plan](PLAN.md)
- [Slack: tracks and build guidance, Sep 24](https://2026sddefense-dcs5018.slack.com/archives/C0C3LG6PKRB/p1790278909686629)
- [Slack: hardware announcement and attached workbook, Oct 16](https://2026sddefense-dcs5018.slack.com/archives/C0C3LG6PKRB/p1792194966030419)
- [Slack: EEG availability question, Oct 16](https://2026sddefense-dcs5018.slack.com/archives/C0C3TDMLM4Z/p1792199415550999)
- [Slack: welcome and project-form deadline, Sep 23](https://2026sddefense-dcs5018.slack.com/archives/C0C3LG6PKRB/p1790225319014299)
- [Event page and logistics](https://luma.com/kfgh2fvu)
