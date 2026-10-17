# Wearable physiology product + model-building benchmark

**Start here:** confirm which sensors the team can physically use. That choice fixes the first prediction task.

**Environment reading path:** [first episode](#210-environment-v0-the-first-complete-episode) → [five challenges](#211-five-challenges-with-a-reason-for-each) → [difficulty and saturation](#212-keep-the-benchmark-informative-as-models-improve) → [sensor configuration](#213-make-the-sensor-easy-to-plug-in-and-the-benchmark-easy-to-tune).

**Product:** help a medic or team operator follow people remotely, see who needs a check, and track an authorized response.

**Research:** teach AI agents to write small, executable models of an individual's exertion and recovery using the same inputs our wearable provides.

**Status — October 17, 2026:** environment v0 is specified in sections 2.10–2.14; it has not been implemented or empirically calibrated. Plan updated after reading the connected Slack announcements, general questions, relevant threads, and the organizer hardware workbook. Listed parts are verified against the workbook; physical allocation, data collection, predictor performance, deployment, and training results remain unverified. Fine-tuning decisions belong to Nikhil.

## 1. Hardware device, dashboard, and product

### 1.1 The person and problem

Our first user is a medic or team operator responsible for several people when direct assessment is delayed. They need a clear answer to: **who needs a check, what changed, and what has already been done?**

The wearable collects supported measurements. The dashboard shows current evidence and an individual's expected response to activity. When something deserves attention, the operator reviews the evidence, requests a check, records an action, and follows the outcome.

**Why build this:** the biological model earns its place by helping the operator interpret a changing person over time. The product must demonstrate a complete action and reassessment loop.

### 1.2 Choose hardware from what is actually available

The organizer’s [October 16 hardware announcement](https://2026sddefense-dcs5018.slack.com/archives/C0C3LG6PKRB/p1792194966030419) includes an Excel inventory. The exact parts below are verified against its “Order List” sheet. **Being listed does not confirm the team has received a working unit.** The “Qty” column often contains usage notes rather than counts; do not assume every item is available per team.

| Listed equipment | Proposed use | What must be verified |
| --- | --- | --- |
| SparkFun Pulse Oximeter & Heart Rate Sensor, Qwiic **SEN-15219** — row 7 | Pulse measurement for the first prediction task | Available outputs and raw-data access; sensor placement; movement quality. Workbook notes two extra reset/MFIO jumpers beyond Qwiic. |
| Adafruit **LSM6DSOX** STEMMA QT — row 13 | Movement and orientation features | Mounting, sample rate, timestamps, supported activity distinctions |
| **Polar H10**, Polar Verity Sense, **AD8232** ECG front end — rows 8, 16–17 | Prioritize an independent chest-strap reference; retain alternatives | Access, synchronization, signal quality, supported output. Verity Sense is optical; it is not interchangeable with an ECG reference. |
| Arduino Nano ESP32 **ABX00083** or Adafruit ESP32-S3 Feather; buttons, LEDs, power banks, straps — rows 2–3, 37, 44, 47–49 | Wearable capture and distress/acknowledgment loop | Exact board received, wiring, power, connection. Nano has no Qwiic port; workbook lists adapters. |
| SparkFun Qwiic **TMP117**; Grove GSR — rows 10, 12 | Optional additional observed channels | Contact, quality, response time, and a defined validation target before adding scope |

Start with SEN-15219 pulse, LSM6DSOX movement, and a manual distress button if those components work. Use the H10 as a development reference if allocated. Keep temperature and GSR optional. Radar, thermal cameras, and environmental sensors stay outside the first build unless a core requirement needs them. A single quadruped with an arm/claw is announced for the event; shared robot availability is not a dependency for this product.

**Hardware descriptions are not validation:** the workbook lists application ideas such as contactless heartbeat, fever, and tissue oxygenation proxies. Treat these as suggested uses, not demonstrated accuracy or clinical capabilities. The workbook attached to the [hardware announcement](https://2026sddefense-dcs5018.slack.com/archives/C0C3LG6PKRB/p1792194966030419) provides exact rows and wiring notes (event Slack access required).

**Decision rule:** if movement sensing fails, use recorded activity start/stop markers and narrow the task to recovery after a known activity. If pulse measurement fails, fix acquisition before claiming a physiology-model demonstration.

### 1.3 The device-to-dashboard pipeline

```text
Wearable measurements + button events
                  ↓
Timestamped acquisition and connection checks
                  ↓
Gateway on a phone or laptop
                  ↓
Stored recording + processed pulse/movement features
                  ↓
Personalized predictor + current measured response
                  ↓
Operator dashboard → reviewed action → acknowledgment → reassessment
```

1. **Collect:** retain participant ID, session ID, device ID, timestamps, units, and available raw signals.
2. **Synchronize:** align devices, record offsets, and check timing drift. Record activity markers explicitly.
3. **Process:** generate pulse estimates, movement features, and signal-quality flags. Preserve gaps and rejected samples.
4. **Predict:** run an approved predictor and show its expected response beside measurements. Display uncertainty only if the method has been evaluated.
5. **Act and follow:** let the operator request a check, acknowledge an alert, record a response, and see the next update.

The LLM builds predictors in a separate controlled environment. The wearable and dashboard execute a reviewed predictor; newly generated code does not automatically replace the deployed model.

### 1.4 What the dashboard must show

| View | Minimum content | Why it exists |
| --- | --- | --- |
| Team overview | Person, last update, pulse trend, movement, distress status, pending checks | Find people requiring attention |
| Individual view | Measured and predicted response, recent activity, quality, model version | Explain what changed |
| Attention queue | Trigger, supporting measurements, freshness, acknowledgment | Make the reason for each check visible |
| Action panel | Request check, acknowledge, record action, reassess; optional human-approved mock actuator | Complete the operational loop |
| Recording/model panel | Session saved, available inputs, predictor version, validation status, data origin | Show how hardware feeds model development |

A deviation from predicted recovery means **check this person or the measurement**, not a diagnosis. Missing data, disconnected devices, sensor failure, and physiological deviations must have visibly different states. An IMU alone does not establish injury or consciousness. A skin-temperature reading does not establish core temperature.

Location is optional and must identify its source. A phone coordinate, replayed coordinate, or manually placed demo marker must be labeled accurately.

### 1.5 The demo: one complete loop

1. Show a real wearable feed reaching the dashboard and saving a synchronized session.
2. Show activity and recovery with the current predictor overlaid. Use a labeled replay if a live trajectory takes too long.
3. Trigger a manual distress event or labeled replay scenario. Show its evidence in the attention queue.
4. Have the operator request a check and acknowledge the response. If an actuator exists, demonstrate a harmless external indicator after explicit approval.
5. Show continued monitoring and the saved event history. Open one benchmark episode showing how an agent improved a predictor.

Display **LIVE / REPLAY / SYNTHETIC** on the relevant views. Demonstrating an acknowledgment or actuator does not establish a therapeutic effect. Physical drug delivery remains future work.

### 1.6 Presentation: explain why each part exists

**Official prompt anchor:** the organizer’s [build guidelines](https://2026sddefense-dcs5018.slack.com/archives/C0C3LG6PKRB/p1790278909686629) explicitly include **Operational Phenotyping with Minimal Sensors**. It asks for measurements that support personnel decisions while minimizing what a warfighter must wear, charge, and maintain. Signal processing and biohardware optimization are explicitly listed demo types.

**Why our design follows that prompt:** use a small wearable; compare each person with their own modeled response; show the evidence that prompts an operator check; measure predictive improvement and device burden. This is the strongest verified event-specific justification for the product. Injury triage is a longer-term application; an exertion/recovery pilot establishes only its tested capability.

**Suggested opening:** “We’re tackling the operational-phenotyping prompt: help a team leader see who needs a check using a small wearable. Our dashboard follows each person’s exertion and recovery, while AI agents build personalized predictive models from those same recordings. We test those programs on sessions they have never seen.”

**Suggested three-minute presentation** — a rehearsal format, not a verified event time limit:

| Time | Show | Point |
| --- | --- | --- |
| 0:00–0:25 | Operator problem and physical wearable | A concrete person needs this |
| 0:25–1:20 | Feed → evidence → operator action → reassessment | The product loop works |
| 1:20–2:05 | Recording → virtual participant → agent-written model | Explain the research mechanism |
| 2:05–2:40 | Hidden-session prediction plot and baseline comparison | Measurements determine predictive success |
| 2:40–3:00 | Verified progress, limitations, next experiment | Make the evidence easy to assess |

**Track positioning:** lead with Human Performance & Prolonged Field Care. Biological Data for Defense is a secondary fit through synchronized wearable data and executable modeling. Do not stretch the MVP into distributed biomanufacturing or pathogen countermeasures. The live [event page](https://luma.com/kfgh2fvu) lists these tracks and the 24-hour build format; the detailed track descriptions and operational-phenotyping prompt are now verified in [Slack](https://2026sddefense-dcs5018.slack.com/archives/C0C3LG6PKRB/p1790278909686629).

**Judge-specific emphasis:** the five names below come from your pasted October 13 organizer announcement. The Slack channels and searches reviewed did not independently reproduce that five-person roster or reveal an official scoring rubric or per-team demo length. Current attendance remains unverified. Presentation advice is our inference from public backgrounds, not a statement of private preferences.

| Judge from pasted announcement | Public evidence | Presentation implication |
| --- | --- | --- |
| Arye Lipman — Biosphere | Biosphere describes its focus on practical biological infrastructure and manufacturing bottlenecks. [Company](https://www.biosphere.io/about) | Show the complete measurement-to-model pipeline, its bottleneck, and how another device or participant can enter it. Explain the biological measurement being modeled. |
| Jake Adler — Pilgrim | His own site describes hardware, chemistry, wearables, and battlefield-ready biotechnology. [Personal site](https://www.jakeradler.com/) | Put the physical device and medic workflow first. Be ready to explain mounting, power, connection loss, and what was demonstrated. |
| Moufeed Kaddoura — 1517 Fund | His public profile lists 1517. The fund describes support for early builders and hardware with a data component. [Profile](https://www.linkedin.com/in/moufeed), [Fund](https://www.1517fund.com/) | Explain who adopts the product first, what the team built, and why deployment creates useful data. Treat this as a fund-level inference. |
| Edward You — former FBI biosecurity agent | A current symposium bio describes work across biotechnology, AI, and national security. [Speaker bio](https://biosecuritysymposium.org/fusion_element/toggle-edward-you/) | Explain defensive value, protection of identifiable recordings, human authority, and the difference between measured performance and speculative capability. |
| Joseph Alzagatiti — American Bioenhancement Corp., per pasted announcement | ABE publicly emphasizes American biological capability; public material reviewed did not establish his personal judging priorities. [Company](https://www.americanbioenhancement.com/) | Explain how individual response modeling could support performance and recovery monitoring. State which biological conclusions the sensors can support. |

**Why the U.S.–China experiment belongs:** it tests whether a specialized U.S. open-weight model can build better predictors on this defined task than selected Chinese frontier models. Lead with usefulness and evidence. A scoped result does not establish national superiority across biology.

### 1.7 Product build order and proof

Planning estimates assume two people and working tools; they are not promises. The organizer’s [schedule reply](https://2026sddefense-dcs5018.slack.com/archives/C0C3TDMLM4Z/p1791957304789599) confirms building from noon Saturday October 17 to noon Sunday October 18, Sunday demos start at 12:02 p.m. sharp, and judging/final selection is at 2 p.m., local Pacific time. A separate thread confirms overnight access. A three-minute presentation remains our rehearsal format; no per-team limit was found.

**Participation checks from Slack:** the [welcome announcement](https://2026sddefense-dcs5018.slack.com/archives/C0C3LG6PKRB/p1790225319014299) specified one project/proposal form per team by October 15 at 11:59 p.m. (written “PST” in the post). This is a past administrative deadline, not evidence of a final-code submission deadline. Verify existing team submission status with the organizers if unknown. Participation is in person, and outside hardware requires approval at registration. These do not prevent preparing the plan.

| Build window | Deliverable | Proof before moving on |
| --- | --- | --- |
| Hours 0–2 | Confirm hardware and smallest supported task | A saved usable signal with timestamps |
| Hours 2–6 | Gateway, recording, dashboard, manual distress loop | A real event reaches the correct person and is acknowledged |
| Hours 6–12 | Repeat recordings and numerical baseline; benchmark work can proceed alongside product work | Separate sessions and a baseline prediction on an unseen session |
| Hours 12–20 | Prototype virtual participants, agent episode, evaluation | Executable submission and saved evaluation artifacts |
| Hours 20–24 | Freeze demo, rehearse, capture backup | Working product loop, labeled evidence, reproducible results |

Collect only voluntary, comfortable activity recordings; record the activity rather than prescribing exertion from the model. Broad participant validation extends beyond the hackathon. If only one person or a few sessions are available, present a feasibility pilot. Fine-tuning is a separately owned workstream whose timing depends on Nikhil's decision.

## 2. Benchmark, environments, and evaluation

### 2.1 The task in one sentence

**“Given limited recordings from this wearable, write a small program that predicts this person's exertion and recovery response during an unseen session.”**

The LLM inspects data, chooses a model, fits it with numerical tools, tests it on development recordings, and revises its code. Its final answer is an executable predictor and a short evidence-based report.

Start with processed pulse/heart-rate and activity features from the real acquisition pipeline. Add raw-waveform tasks only after acquisition and reference validation work. These are different task difficulties and should have separate results.

**What the predictor knows:** freeze the available inputs at each prediction time. For an activity-conditioned task it may see the activity schedule or allowed activity features, but never the hidden future pulse values. A forecast based only on past measurements is a separate task. Report the prediction horizon and any warm-up period.

### 2.2 Build virtual participants from our sensor recordings

```text
Real sensor recordings from development participants
                         ↓
Fit a family of activity-to-response models
                         ↓
Sample virtual participants within supported ranges
                         ↓
Apply a separate sensor/measurement model
                         ↓
Produce the same fields our wearable pipeline supplies
                         ↓
Agent builds a predictor → evaluator runs hidden sessions
                         ↓
Final transfer test on untouched real recordings
```

1. **Collect repeated sessions:** capture comfortable activity and recovery, with activity timing, stable placement, and independent reference measurements where available.
2. **Fit the generator:** estimate shared dynamics and individual variation from development data. Begin with simple response curves or dynamical models and compare them on held-out development recordings.
3. **Generate participants:** vary supported response characteristics and session conditions. Keep physiological variation separate from sensor noise and missing data. Avoid parameter combinations the recordings cannot justify.
4. **Check realism:** compare generated and real curves, recovery behavior, motion relationships, gaps, and variation. Use held-out development data; a believable chart alone is insufficient.
5. **Freeze evaluation:** keep test participants and sessions out of generator fitting, agent-training examples, and tuning. Reserve different model families and activity patterns for synthetic generalization tests where feasible.

**What a “body” means here:** a virtual participant modeling the observable responses supported by our sensors. Limited pulse and movement recordings do not uniquely reveal internal physiology. Additional generated participants create practice cases; they do not increase the number of independently measured humans.

A tiny dataset supports a narrow simulator prototype. It cannot establish a reliable population distribution. Keep public-data, locally recorded, and synthetic results identifiable.

### 2.3 The three environments

| Environment | Agent receives | Agent must do | Hidden evaluation |
| --- | --- | --- | --- |
| Build | Limited recordings from a new participant | Construct a useful personalized predictor | Another session from that participant |
| Transfer | An existing predictor and a small amount of data from a changed activity/workload | Adapt using the allowed new evidence | Remaining unseen sessions in the new setting |
| Update | An existing predictor and another session | Improve while retaining earlier performance | Held-out later sessions plus retained earlier test sessions |

Build comes first. Transfer requires genuine activity/workload coverage. Update requires repeated sessions. Publish the latter two as planned until those prerequisites exist.

**Example episode:** “Here are two recordings of participant A and the permitted activity inputs. Fit a predictor, inspect development error, and revise it. Submit code for a third session.” All numbers and session counts in examples are illustrative until collection establishes what is available.

### 2.4 Tools, submissions, and the oracle

| Item | Required design |
| --- | --- |
| Tools | Same Python environment, data access, permitted libraries, fitting runtime, development evaluations, and submission contract for all LLM agents |
| Submission | Executable predictor, fitted artifact, input/output specification, model version, short report, and saved experiment trace |
| Execution | Isolated runner with time/memory limits and no network access; hidden labels stay outside the agent-visible environment |
| Synthetic oracle | Hidden generator produces expected trajectories for unseen inputs; parameter recovery is scored only when identifiable |
| Real-world target | Hidden recorded response, preferably with an independent reference; compare predictions with measurements rather than demanding identical equations |

An agent can learn to imitate our simulator without generalizing to people. **Synthetic performance and real-recording performance are separate reported results.** If a real target is a processed PPG estimate, say that explicitly; it is not an independent physiological reference.

### 2.5 Score predictive performance and scientific behavior separately

| Evaluation | What we measure | What it establishes |
| --- | --- | --- |
| Prediction — primary | Mean absolute error in bpm, error by exertion/recovery phase, improvement over fixed baselines | Accuracy on the defined hidden task |
| Generalization | Per-participant and per-session results, transfer error, retained performance after updates | Whether improvement persists across the tested changes |
| Reliability | Submission failures, invalid outputs, runtime, resource use; calibration/coverage if intervals are submitted | Whether predictors execute and uncertainty is meaningful |
| Agent behavior — secondary | Trace-grounded review of experiments, revisions, evidence, leakage, and unsupported claims | Quality of the modeling process |
| Product workflow — separate | Correct-person events, time to display/acknowledgment, disconnect handling, approval/action log | Whether the operator loop functions |

Report uncertainty across participants where sample size supports it. Do not treat adjacent timestamps as independent people. Predefine treatment of missing data and failed submissions; report failures rather than silently dropping them. Freeze the primary metric and aggregation before final evaluation.

**LLM judge:** use a fixed rubric with code, logs, and measured outcomes. Hide model identities and randomize ordering for pairwise reviews. Audit a sample with human review. Report this score separately from measured prediction accuracy and deterministic workflow checks. Documented judge biases are one reason to preserve that separation. [Research](https://arxiv.org/abs/2306.05685)

### 2.6 Competitors and a fair comparison

| Competitor | Why include it |
| --- | --- |
| Simple numerical baselines | Persistence where appropriate, a fixed population model, and a directly fitted personalized response model show how much the agent adds |
| U.S. base checkpoint | Measures performance before specialization |
| Fine-tuned U.S. checkpoint — Nikhil's decision | Tests the specialization hypothesis once training exists |
| Selected Chinese frontier models | Tests the stated international comparison on the same task |
| Fixed modeling workflow without an LLM | Tests whether ordinary automated fitting solves the task with less complexity |

Freeze exact model IDs, versions, prompts, decoding settings, allowed inputs, seeds/repeats, fitting limits, and number of development calls. Give each agent the same tools and controlled execution resources. Report tokens, API costs, and latency separately; hosted model inference compute cannot be assumed identical. Choose comparator versions before examining final test results.

### 2.7 Fine-tuning — reserved for Nikhil

**Owner: Nikhil. Decision: open.** The benchmark supplies episodes, saved trajectories, development feedback, and a frozen evaluation interface.

| Decision | Status |
| --- | --- |
| U.S. open-weight checkpoint and license | Nikhil to choose |
| Training data and teacher/example-generation method | Nikhil to choose |
| Supervised training, RL, or sequence | Nikhil to choose |
| Compute, budget, training method, hyperparameters | Nikhil to choose |
| Training success criteria and final checkpoint selection | Nikhil to choose before final evaluation |

Training and compute selection remain pending Nikhil's decisions. The product and base-model benchmark can progress while these decisions remain open.

### 2.8 Research grounding and presentation claims

Personalized physiological heart-rate modeling already exists. Our proposed contribution is the environment and evaluation for agents that **construct and adapt executable predictors from limited wearable evidence**. Novelty still needs a focused comparison with existing system-identification and scientific-agent benchmarks. [Related physiological modeling research](https://www.nature.com/articles/s41746-023-00926-4)

AssayLoop, the Virtual Embryo Challenge, and SynthPert remain research leads from the discussion, rather than verified evidence in this document for this task or a U.S.–China outcome.

**The demonstration is ready when:**

1. A real wearable recording reaches the dashboard and can be saved/replayed.
2. An operator completes a distress/check/acknowledgment/reassessment loop.
3. A simple baseline and at least one agent-written predictor execute on a hidden session.
4. A virtual-participant episode uses the real pipeline's input schema, with generator limitations stated.
5. Results clearly identify real versus synthetic data, measured performance versus judged behavior, and completed versus planned training.

### 2.9 Source status and remaining checks

| Source | Status on October 17, 2026 | How it is used |
| --- | --- | --- |
| [Luma event page](https://luma.com/kfgh2fvu) | Live public HTML inspected in the earlier pass | Event format, tracks, sponsors |
| [Slack build guidelines](https://2026sddefense-dcs5018.slack.com/archives/C0C3LG6PKRB/p1790278909686629) | Full message and thread read; no replies | Official operational-phenotyping prompt, track fit, demo types, preparation guidance |
| [Hardware announcement](https://2026sddefense-dcs5018.slack.com/archives/C0C3LG6PKRB/p1792194966030419) and attached workbook (event Slack access required) | Message/thread and all workbook rows read; earlier hardware Q&A also read | Exact listed parts; physical allocation remains unverified |
| [Schedule thread](https://2026sddefense-dcs5018.slack.com/archives/C0C3TDMLM4Z/p1791927436967009) | Parent and organizer reply read | Build, demo, and judging schedule |
| [October 13 Luma announcement](https://luma.com/blast/uRhDv8llCV) | Judge roster remains sourced to the earlier user paste; not independently found in reviewed Slack | Provisional judge-specific preparation |

**Review coverage:** read the full returned histories of announcements and general-questions, with no remaining history pages. Read hardware, schedule, overnight-access, and build-guideline threads. Searched accessible public messages for judges/judging, hardware, presentation, rubric, criteria, demo, and minutes. No official rubric or per-team presentation limit was found. This is a search result, not proof none exists elsewhere.

**Open event checks:** current five-person judge roster; presentation length/scoring weights; actual equipment allocation and functioning; team proposal status and any separate final deliverable instructions. Judge-specific emphasis remains inference from public work. Fine-tuning remains Nikhil's decision.

**Hardware gate:** check the equipment desk for the SEN-15219, LSM6DSOX, and an available H10 reference; record what the team actually receives.


### 2.10 Environment v0: the first complete episode

**Decision:** start with Build, focused on predicting recovery after a recorded activity ends. Keep the task narrow enough to verify and the interface broad enough to reuse for Transfer and Update.

**Example agent assignment:** “Use this person's calibration recordings to build a recovery predictor. You may inspect the files, run fitting code, and request development feedback. Submit an executable predictor for an unseen session.”

1. **Inspect:** read the channel manifest and allowed calibration recordings. Identify units, missing data, activity markers, and target provenance.
2. **Build:** write a predictor and fit its parameters using the allowed recordings. Simple equations, state-space models, and permitted statistical models are all eligible.
3. **Evaluate:** request feedback on a separate development session. Receive execution status, error in bpm, and error by time segment. Development feedback may include a residual plot.
4. **Revise:** use the remaining fixed budget to change the model or fitting process. Preserve every candidate, execution log, and feedback result.
5. **Submit:** freeze the selected code and fitted artifact. The evaluator runs it against hidden sessions with the same allowed input contract.

**Recovery task input:** the runner supplies an observed history ending at a declared cutoff and the recorded activity-stop marker. The submitted predictor generates a future trajectory without seeing future pulse values. Initially allow a recovery-only schedule where the subsequent rest condition is known; validate that it was actually recorded. Later activity-conditioned tasks may provide future activity inputs, but must be labeled separately.

**Provisional pilot settings:** 60 seconds of observed history, a 120-second prediction horizon, three development-feedback calls, and ten minutes of agent time per episode. These are engineering starting points, not evidence that recordings support them. Shorten or extend them using development recordings before freezing a scored version. Record token use, fitting runtime, execution runtime, and wall-clock limits separately.

### 2.11 Five challenges, with a reason for each

These are capability checks inside the three environments, not five independent products. Launch challenge 1 first; add the others when recordings and pilot results support them.

| Challenge | What changes / what the agent must do | Why it belongs | Evidence required |
| --- | --- | --- | --- |
| 1. Personalize — Build | Fit an individual's response from a small set of calibration recordings; predict another session | Tests whether personal data improves on the population predictor | Improvement over population and directly fitted individual baselines on hidden sessions |
| 2. Improve from feedback — Build | Revise a submitted predictor after limited development evaluations | Tests whether the agent learns from an experiment | Hidden performance of its first and selected final submissions, evaluated afterward without exposing test feedback |
| 3. Transfer activity — Transfer | Adapt an existing predictor with a small amount of evidence from a changed activity pattern or workload | Tests whether it can handle new operational conditions | Separate unseen sessions in the new condition; report adaptation-data duration |
| 4. Update across sessions — Update | Use a later recording while preserving useful earlier behavior | Tests continued personalization without forgetting | New-session improvement and earlier-session retention reported separately |
| 5. Handle incomplete evidence — modifier on Build/Transfer/Update | Remain usable with validated gaps or degraded channels; provide uncertainty if supported | Wearable deployment includes incomplete observations | Error and coverage on independently valid targets; uncertainty scored by a proper rule if enabled |

Challenge 5 supports the modeling task; sensor-fault detection is not the main research contribution. Optional inputs, interval estimates, and abstention must have explicit scoring rules. An agent cannot improve its headline score by skipping hard windows. In the primary point-prediction track it must return every required prediction; report execution failures separately. Add an abstention track only with a fixed coverage requirement or a fixed penalty.

**Feedback ablation:** compare revision with real development feedback against the same agent given the same total time and calls but no numeric development feedback. Preserve the original base-model and numerical-baseline comparisons. This checks whether interaction helps beyond extra computation.

### 2.12 Keep the benchmark informative as models improve

**“Unsaturated” means competitors still have measurable room to improve on useful, solvable tasks. It cannot be guaranteed by a design document or by generating unlimited examples.** We must run strong agents and conventional fitting methods to find the actual ceiling.

**Difficulty comes from five controlled sources:**

| Control | Easy starting case | Harder supported case | Guardrail |
| --- | --- | --- | --- |
| Personalization data | Several complete sessions | Fewer or shorter calibration recordings | Ensure calibration still contains enough information for the prediction task |
| Response structure | One response timescale | Different exertion/recovery timescales, slow drift, or multiple response components | Include only dynamics grounded in development data or clearly labeled synthetic stress tests |
| Generalization | Familiar activity pattern | New activity sequence or held-out dynamics family | Changing random seeds alone does not test structural generalization |
| Adaptation budget | More new-session evidence and feedback calls | Less adaptation evidence or fewer evaluations | Use equal budgets for all competitors within each comparison |
| Observation quality | Clean aligned channels | Gaps and artifacts measured in our acquisition pipeline | Keep underlying physiological change distinct from sensor error |

**Do not add arbitrary noise to make everyone fail.** If the target cannot be predicted from the permitted inputs, characterize the information limit, improve measurements, or narrow the claim. A hard task needs a plausible route to better performance.

**Pilot diagnosis:** run a population model, direct individual curve fitting, an automatic model-selection workflow, the U.S. base model, and selected frontier agents on the same development episodes. Inspect individual cases as well as averages.

| What the pilot shows | Decision |
| --- | --- |
| Everyone scores well and similarly | Keep the slice as a basic check; add supported transfer, reduced-data, or multi-session challenges |
| A fixed fitter wins throughout | Report that result. Test whether the agent adds value in model selection/adaptation; retire an uninformative agent task if it does not |
| Everyone fails, including strong numerical methods | Diagnose missing inputs, labels, synchronization, or impossible extrapolation before increasing difficulty |
| Rankings change across repeats | Increase repeated episodes or participants; report uncertainty before ranking models |
| Improvements appear only on synthetic cases | Claim simulator improvement and keep real-data transfer unresolved |

**Renew cases without moving the target:** publish a versioned task distribution and frozen scoring protocol. Keep the current test batch fixed within a leaderboard version. Add genuinely new recordings and validated dynamics in a later version; rerun anchor baselines and compare models within the same version. Record hashes and seeds so a result can be reproduced. Fresh cases help limit contamination; they do not prove difficulty or scientific novelty. [LiveBench research](https://arxiv.org/abs/2406.19314) motivates refreshed cases and objective scoring, without guaranteeing immunity to contamination.

**Prevent shortcuts:** keep complete sessions together, fit normalizers and preprocessing only on allowed data, and ensure online features never use samples beyond their cutoff. Do not put hidden parameters, future targets, or generator seeds into agent-visible files or metadata. Separate the runner's hidden labels from the candidate's inputs. Audit near-duplicate recordings and do not split overlapping windows between support and hidden sessions.

### 2.13 Make the sensor easy to plug in and the benchmark easy to tune

Use five small components with explicit responsibilities:

| Component | Responsibility | What changes after the sensor works |
| --- | --- | --- |
| Sensor adapter | Convert device outputs into a common timestamped recording format | Device parsing, units, channel availability, delays, quality flags |
| Recording store | Keep immutable raw files and processed versions with provenance | Add real sessions; retain participant/session grouping |
| Episode builder | Select support, development, and hidden recordings for a challenge | Coverage requirements, durations, horizons, input channels |
| Participant generator | Fit supported response families and measurement variation from permitted recordings | Parameters and measured artifact distributions; no test participants used for generator fitting |
| Agent runner and evaluator | Expose tools, enforce budgets, execute submissions, calculate metrics, save traces | Keep the interface stable; version scoring changes explicitly |

**Common recording contract:** relative time in seconds; pseudonymous participant and session IDs; available pulse and movement channels with units; validity masks; activity markers; device/processing versions; and target provenance. Required fields and optional fields are explicit. A missing channel remains missing. Retain device sample timestamps rather than assuming arrival order gives synchronization.

**Reference separation:** independent ECG/chest-strap labels may score a prediction while being unavailable as live model inputs. If no independent reference is collected, label the target as the sensor-derived pulse estimate and scope claims accordingly. Do not score “true physiology during PPG dropout” unless an independent usable reference exists.

**One configuration controls each episode family.** An illustrative specification:

```yaml
benchmark_version: pilot-0
challenge: build_recovery
sensor_profile: pending_actual_device
input_channels: [pulse_bpm, movement, quality, activity_markers]
target_source: pending_reference_check
history_seconds: 60
forecast_seconds: 120
support_sessions: 2
development_sessions: 1
hidden_sessions: 1
max_feedback_calls: 3
agent_wall_seconds: 600
primary_metric: phase_balanced_mae_bpm
```

Counts and durations above are provisional. A virtual episode can supply them immediately; a real episode is enabled only when distinct recordings exist. Public configurations describe the contract. Final-test record IDs, target files, and seeds live in the evaluator's private manifest.

**After hardware is ready:**

1. **Confirm the data contract — about 30–60 minutes after a usable recording exists:** inspect actual outputs and timestamps; decide processed-input availability and reference target.
2. **Collect a pilot — about 1–2 hours if capture works:** obtain distinct voluntary activity/recovery recordings with markers. A one-person collection is a plumbing and feasibility pilot.
3. **Calibrate difficulty — about 1–2 hours of engineering plus agent runtime:** fit numerical baselines, set supported horizons and data requirements, and diagnose easy versus impossible slices using development data.
4. **Freeze pilot-1 — about 30–60 minutes:** save the config, processing versions, metrics, budgets, and participant/session splits before scoring competitors.
5. **Run the comparison:** calculate actual runtime as number of episodes × repeated runs × agent time limit, plus baseline and evaluation costs. Use identical episode batches across competitors.

These are planning estimates, contingent on capture quality. Broader participant collection and biological validation require additional time.

### 2.14 Training feedback, final scoring, and tuning boundaries

**Two levels of splitting:**

| Level | How it works |
| --- | --- |
| Across participants | Agent-training participants fit the generator and supply training episodes. Separate development participants guide benchmark/model choices. Final-test participants never inform those choices. |
| Inside an episode | The agent sees support recordings and limited development feedback for the target person. A different session remains hidden. Calibration on a test person's explicitly supplied support data is permitted personalization; it does not make their hidden session available. |

Synthetic training episodes can be resampled freely from the training distribution. Benchmark development uses separate seeds and cases, plus withheld development participants. Final scoring uses sealed test cases with restricted submissions and no per-window feedback. Full traces and labels can be released after a test batch is retired if participant permissions allow; use a new batch for future comparisons.

**Primary error:** for a challenge with exertion and recovery, calculate MAE separately in each phase and average the two phase errors equally. For recovery-only tasks, report recovery MAE. Average sessions within each participant, then participants equally. Never let a long easy rest recording dominate the result. Keep synthetic and real panels separate, and report each challenge separately before proposing an overall score.

**Improvement over baseline:** report `1 - agent_error / baseline_error` with the baseline fixed before testing. Positive means improvement, zero matches, and negative means worse. Always report raw bpm error too. Near-zero baseline error makes the ratio unstable; predeclare a minimum denominator or omit the ratio on that slice. Baselines are not an oracle ceiling; an agent may beat them.

**Feedback/reward interface for future training:** return a machine-readable result containing execution validity, phase errors, baseline errors, resource use, and version IDs. An illustrative bounded training signal is `-min(development_MAE_bpm / 10, 1)` for valid submissions and `-2` for failed execution. This keeps failed execution worse than a successful prediction, but saturates training feedback above 10 bpm; tune the scale on development data before freezing it. Raw final-test MAE remains uncapped. This is a candidate interface example, not a selected fine-tuning recipe. Keep final-test outcomes out of training. Nikhil selects the eventual learning method and reward mapping.

**Separate audit:** an LLM judge reviews experiments, revisions, and justified claims using the saved trace. Its score does not override hidden numerical error. Uncertainty, abstention, and model complexity become scored objectives only in separately specified tracks; they are not adjustable penalties introduced after results are known.

**Close prior work:** [TimeSeriesGym](https://arxiv.org/abs/2505.13291) already evaluates agents on time-series engineering and multiple artifacts with numerical and LLM-based evaluation. We should inspect its [runner and grading interfaces](https://github.com/moment-timeseries-foundation-model/TimeSeriesGym) before choosing to reuse infrastructure. Our proposed specialization is wearable-grounded personal response modeling across Build, Transfer, and Update, with real-session validation. This is a positioning hypothesis, not established novelty.

**First decisive experiment:** on an untouched session, compare a directly fitted personalized response model with an agent that has the same recordings and fitting resources. If the agent cannot improve the predictor or adapt it more usefully across the tested cases, report that and revisit the task before committing to a large training run.

**Next action — under two minutes:** confirm whether the first recording will include an H10 reference alongside PPG and movement. That sets the benchmark target and the claims we can evaluate.
