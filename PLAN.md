# Wearable physiology product + model-building benchmark

**Start here:** confirm which sensors the team can physically use. That choice fixes the first prediction task.

**Product:** help a medic or team operator follow people remotely, see who needs a check, and track an authorized response.

**Research:** teach AI agents to write small, executable models of an individual's exertion and recovery using the same inputs our wearable provides.

**Status — October 17, 2026:** plan updated after reading the connected Slack announcements, general questions, relevant threads, and the organizer hardware workbook. Listed parts are verified against the workbook; physical allocation, data collection, predictor performance, deployment, and training results remain unverified. Fine-tuning decisions belong to Nikhil.

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

Planning estimates assume two people and working tools; they are not promises. The organizer’s [schedule reply](https://2026sddefense-dcs5018.slack.com/archives/C0C3TDMLM4Z/p1791957304789599) confirms building from noon Saturday October 17 to noon Sunday October 18, Sunday demos from noon, and judging/final selection at 2 p.m., local Pacific time. A separate thread confirms overnight access. A three-minute presentation remains our rehearsal format; no per-team limit was found.

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

**Next action — under two minutes:** check the equipment desk for the SEN-15219, LSM6DSOX, and an available H10 reference; record what the team actually receives.
