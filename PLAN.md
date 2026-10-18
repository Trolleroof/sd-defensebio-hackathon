# Wearable physiology product + model-building benchmark

**Start here:** confirm which sensors the team can physically use. That choice fixes the first prediction task.

**Two phases:** Phase 1 is the hardware, dashboard, and product assistant. Phase 2 is the physiology research workspace and post-training environment. Read [workspace](#22-the-agents-research-workspace) → [tasks](#23-four-task-families-in-a-persistent-workspace) → [rl-environments](#25-rl-environments-and-post-training) → [scoring](#27-scoring-the-decisions-final-actions-and-predictor).

**Product:** help a medic or team operator follow people remotely, see who needs a check, and track an authorized response.

**Research:** teach AI agents to write small, executable models of an individual's exertion and recovery using the same inputs our wearable provides.

**Status — October 17, 2026:** Phase 2 below reflects the research-workspace, RL-environment, and reasoning-judge discussion; it has not been implemented or empirically calibrated. Plan updated after reading the connected Slack announcements, general questions, relevant threads, and the organizer hardware workbook. Listed parts are verified against the workbook; physical allocation, data collection, predictor performance, deployment, and training results remain unverified. Fine-tuning decisions belong to Nikhil.

## 1. Phase 1 — hardware device, dashboard, and product assistant

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

**Selected team hardware for the two-person demo:** the available inventory is Arduino Uno R3 ×1, Raspberry Pi 4B ×1, Arduino Nano ESP32 ×3, nRF52840 Starter Kit ×1, Polar Verity Sense ×2, LSM6DSOX ×3, and battery packs ×2. Use two Nano ESP32 boards as wearable nodes, each paired over BLE with one Verity Sense and connected over I²C to one LSM6DSOX; strap the node and IMU to the participant. Use the third Nano ESP32 as a USB-powered ESP-WIFI-MESH root at command, connected to the laptop by USB serial. Power the wearables from the battery packs and keep the third LSM6DSOX as a spare. The team-specified Nano ESP32 configuration has an ESP32-S3, 512 KB internal SRAM, 8 MB PSRAM, and 16 MB flash. The Uno, Pi, and nRF52840 kit are not required for v0. Confirm board revisions, wiring, and power before assembly.

Omit the ArduCam UC-844 Rev. B camera and GSR in v0. No skin-temperature sensor or node-mounted GNSS receiver is confirmed; do not describe GSR as temperature, sweat rate, or hydration. Use the Verity Sense for heart rate, and label its optical reading as a sensor measurement rather than an independent ECG reference. Radar, thermal cameras, environmental sensors, and the shared quadruped are outside this build.

**Hardware descriptions are not validation:** the workbook lists application ideas such as contactless heartbeat, fever, and tissue oxygenation proxies. Treat these as suggested uses, not demonstrated accuracy or clinical capabilities. The workbook attached to the [hardware announcement](https://2026sddefense-dcs5018.slack.com/archives/C0C3LG6PKRB/p1792194966030419) provides exact rows and wiring notes (event Slack access required).

**Decision rule:** if movement sensing fails, use recorded activity start/stop markers and narrow the task to recovery after a known activity. If pulse measurement fails, fix acquisition before claiming a physiology-model demonstration.

### 1.3 The device-to-dashboard pipeline

```text
Polar Verity Sense ──BLE──┐
LSM6DSOX ───────────I²C───┴─ Nano ESP32 wearable nodes
                                  ↓ ESP-WIFI-MESH
                         Nano ESP32 root at command
                                  ↓ USB serial (JSON event per line)
                         Local Python service on laptop
                         ├─ append-only JSONL session log
                         └─ WebSocket → local browser dashboard
                                  ↓
                         operator review → priority → route → response
```

The first transport milestone is the [Arduino IMU mesh prototype](firmware/imu_mesh/README.md): the same sketch builds as the command root or a wearable, samples the LSM6DSOX, relays versioned packets over an ESP-WIFI-MESH chain, and emits JSON Lines from the root's USB serial port. It reserves `hr_bpm` for the Verity Sense stage. The full device path remains: each wearable is a BLE central for its Verity Sense and reads its IMU; the command Nano is the mesh root; the laptop connects to that root over USB serial and is not a mesh relay. Full telemetry adds movement class, position estimate with source/quality, and sensor/link freshness. The dashboard keeps disconnected nodes visible with last-update time and a distinct stale/offline state.

1. **Collect:** retain participant ID, session ID, device ID, timestamps, units, and available raw signals. The gateway-to-laptop serial stream uses one versioned JSON event per line; append telemetry and operator actions to a local JSONL session log.
2. **Synchronize:** align devices, record offsets, and check timing drift. Record activity markers explicitly.
3. **Process:** generate pulse estimates, movement features, and signal-quality flags. Preserve gaps and rejected samples.
4. **Predict:** run an approved predictor and show its expected response beside measurements. Display uncertainty only if the method has been evaluated.
5. **Act and follow:** let the operator request a check, acknowledge an alert, record a response, and see the next update.

The LLM builds predictors in a separate controlled environment. The wearable and dashboard execute a reviewed predictor; newly generated code does not automatically replace the deployed model.

The laptop and local dashboard are the v0 command center. Palantir Foundry is a possible later integration, not a dependency for the demo.

### 1.4 What the dashboard must show

| View | Minimum content | Why it exists |
| --- | --- | --- |
| Team overview | Person, last update, heart-rate trend, movement, map position and source, alert/critical status, pending checks | Find people requiring attention |
| Individual view | Measured and predicted response, movement-matched heart-rate baseline, recent activity, signal quality/freshness, position confidence, model version | Explain what changed |
| Attention queue | Automatic abnormality reason, supporting measurements, freshness, operator-set priority, acknowledgment | Make the reason for each check visible |
| Action panel | Request check, acknowledge, record action, reassess; optional human-approved mock actuator | Complete the operational loop |
| Route panel | Critical people in operator priority order, proposed route, added distance, configurable extra-distance limit | Help the operator review an extraction sequence |
| Recording/model panel | Session saved, available inputs, predictor version, validation status, data origin | Show how hardware feeds model development |

A deviation from predicted recovery means **check this person or the measurement**, not a diagnosis. Missing data, disconnected devices, sensor failure, and physiological deviations must have visibly different states. An IMU alone does not establish injury or consciousness. A skin-temperature reading does not establish core temperature.

Use laptop OS location as the command-point origin when available; allow manual origin entry if the laptop has no location source. Initialize both wearable nodes there. Show node positions as **estimated** when derived from the arm/chest IMUs, and provide operator map-position correction when that estimate is unusable. Identify estimated and corrected positions separately. IMU-only dead reckoning will drift and is not reliable geolocation. Use an online OpenStreetMap-based map and routing service; if it is unavailable, keep people and priorities visible but do not claim a computed route.

Build an initial heart-rate baseline and update a separate baseline for each node and movement class (stationary, walking, running) as valid data accumulates. Surface a demo abnormality when heart rate remains at least 20 bpm above the matching baseline for 10 seconds; show the value, baseline, movement class, duration, and signal freshness. Do not adapt a baseline while that node has an active abnormality. The operator, not the sensor rule, marks someone critical and sets priority. The route visits operator-marked critical people in priority order, then considers lower-priority stops as insertions. Include a lower-priority stop only if total route distance is within the operator-configured extra-distance limit versus the critical-only route. Default the limit to 10% and make it editable in the dashboard.

### 1.5 The demo: one complete loop

1. Show both live wearable feeds reaching the laptop through the mesh root and saving a synchronized session.
2. Show heart-rate and movement trends, movement-matched baselines, and estimated or corrected map positions. Label replayed or synthetic data when used.
3. Trigger a sustained heart-rate abnormality and show its evidence and freshness in the attention queue; have the operator mark a person critical and set priority.
4. Show the proposed route, adjust the extra-distance limit in the dashboard, and demonstrate an optional lower-priority stop entering or leaving the route. Have the operator acknowledge/check the person and record the response.
5. Show continued monitoring and the saved event history. Open one benchmark episode showing how an agent improved a predictor.

Display **LIVE / REPLAY / SYNTHETIC** on the relevant views. Demonstrating an acknowledgment or actuator does not establish a therapeutic effect. Physical drug delivery remains future work.

### 1.6 Presentation: explain why each part exists

**Official prompt anchor:** the organizer’s [build guidelines](https://2026sddefense-dcs5018.slack.com/archives/C0C3LG6PKRB/p1790278909686629) explicitly include **Operational Phenotyping with Minimal Sensors**. It asks for measurements that support personnel decisions while minimizing what a warfighter must wear, charge, and maintain. Signal processing and biohardware optimization are explicitly listed demo types.

**Why our design follows that prompt:** use a small wearable; compare each person with their own modeled response; show the evidence that prompts an operator check; measure predictive improvement and device burden. This is the strongest verified event-specific justification for the product. The demo can route to operator-marked priorities; clinical injury triage remains a longer-term application, and an exertion/recovery pilot establishes only its tested capability.

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
| Hours 2–6 | Mesh gateway, recording, dashboard, HR abnormality and route controls | Both node feeds reach the laptop; an alert can be reviewed and route updates when the detour limit changes |
| Hours 6–12 | Repeat recordings and numerical baseline; benchmark work can proceed alongside product work | Separate sessions and a baseline prediction on an unseen session |
| Hours 12–20 | Prototype virtual participants, agent episode, evaluation | Executable submission and saved evaluation artifacts |
| Hours 20–24 | Freeze demo, rehearse, capture backup | Working product loop, labeled evidence, reproducible results |

Collect only voluntary, comfortable activity recordings; record the activity rather than prescribing exertion from the model. Broad participant validation extends beyond the hackathon. If only one person or a few sessions are available, present a feasibility pilot. Fine-tuning is a separately owned workstream whose timing depends on Nikhil's decision.

### 1.8 V0 integration checks and limits

Before calling the wearable-to-command loop complete, demonstrate both Verity Sense feeds, IMU movement classes, and telemetry reaching the laptop through a multi-hop path. Confirm baseline accumulation and a sustained-abnormality alert; operator-set critical status and priority; acknowledgment and response logging; and clear stale/offline status when a node disconnects. Verify laptop-location/manual-origin fallback, corrected-position labeling, route recalculation when the extra-distance limit changes, and graceful behavior when the online map or routing service is unavailable. Label all live, replayed, and synthetic data; save the session log for review.

This two-node demo does not establish medical accuracy, injury detection, reliable GPS-denied localization, or operational mesh range. IMU position estimates on an arm or chest can drift substantially; preserve estimated/corrected labels and leave priority and response decisions with the operator.

## 2. Phase 2 — physiology research workspace, benchmark, and post-training environments

### 2.1 Purpose and research grounding

**The LLM acts as a physiology researcher: it plans experiments, writes executable predictors, uses feedback, and improves models across changing scenarios.** Phase 1 supplies the physical device, dashboard, and operator workflow. Phase 2 tests and eventually trains the research capability behind personalized modeling.

**Example assignment:** “Build a small executable model of this participant's response to activity. You have limited calibration recordings and a fixed experiment budget. Choose what to inspect, test competing approaches, and submit a model that generalizes to later scenarios.”

Start with observed exertion and recovery, using the channels actually available. Heart rate and activity are an initial task profile, not a permanent restriction. Add measured sweat-related signals, temperature, or other channels as acquisition and suitable reference targets become available. An additional input channel does not automatically create a validated prediction target or establish a hidden physiological state.

| Research precedent | What it supports | What we borrow |
| --- | --- | --- |
| [RSI-Bench](https://rsi-benchmark.com/blog/announcing-rsi-bench) | Open-ended research tasks with starting environments, compute budgets, and verifiers for improvement over baselines | A research workspace with iterative experiments, resource limits, and measurable generalization |
| [APEX-Agents paper](https://arxiv.org/html/2601.14242) | Expert-created workspaces, tools, deliverables, and criterion-based grading | Physiology work products and acceptance criteria grounded in realistic research work |
| [HORIZON](https://arxiv.org/html/2604.11978) | Planning, memory, and accumulated-error failures in the models and domains studied | Test necessary dependent decisions and diagnose failures across the trajectory; duration alone is not difficulty |
| [AIDE](https://arxiv.org/abs/2502.13138) | Iterative model development as search over code | A strong frozen-model search competitor; post-training must earn its advantage |
| [Virtual Embryo agent rules](https://virtualembryo.ai/challenge/rules), [Virtual Cell Challenge](https://arcinstitute.org/news/virtual-cell-challenge-2026), [AssayLoop](https://arxiv.org/abs/2609.11877) | Locked agent runs, context generalization, and feedback-conditioned experimental selection | Preserve run evidence, hold out meaningful conditions, and measure experiment selection |

[TimeSeriesGym](https://arxiv.org/abs/2505.13291) is another close prior-work lead for time-series engineering agents; inspect its runner and grading interfaces before choosing infrastructure or claiming novelty.

These sources motivate the design; none establishes that our benchmark is difficult, novel, or improved by post-training. The published APEX-Agents paper's main judge sees final outputs and workspace changes without the full trajectory. Our decision-trace judge is an additional design choice, not a claim that APEX uses this exact protocol. Timeline bookkeeping is a validity check, not the central research challenge.

### 2.2 The agent's research workspace

Give the agent a sandboxed research workstation, rather than a physiology question answered in prose.

```text
Research brief + sensor documentation + allowed recordings
                            ↓
Python/code workspace + plots + experiment notebook
                            ↓
Inspect → hypothesize → choose experiment → fit → evaluate
                            ↓
Revise or revert → select model → submit executable artifact
                            ↓
Private runner: hidden measurements + trace/action judge
```

| Workspace component | Agent-visible content or capability |
| --- | --- |
| Research brief | Target, permitted inputs, prediction cutoff/horizon, task requirements, and resource budget |
| Evidence | Calibration recordings, activity schedules, recording notes, units, signal quality, and provenance |
| Modeling tools | Python, fixed permitted libraries, plots, numerical fitting, starter predictor, and conventional baselines |
| Experiment tools | Development evaluation, requests for available additional recordings, candidate comparison and reversion |
| Work products | Predictor code, fitted artifact, experiment history, decision records, and final evidence-based report |

The agent chooses its approach. Simple response curves, dynamical/state-space models, and permitted statistical methods are eligible. Multiple solutions may be valid. Open-ended solutions do not require a subjective or irreproducible evaluator.

Use an isolated execution environment with time/memory limits and no network access during scored runs. Keep hidden labels, evaluator files, generator parameters, and test seeds outside agent access. Save code versions, tool results, resource usage, and artifacts so decisions can be reconstructed.

**Prediction contract:** supply history only up to a declared cutoff. A recovery-only task may supply a known subsequent rest condition. Activity-conditioned forecasting may supply a declared future activity schedule. Neither may expose future target measurements. These are separate task profiles with explicit horizons and warm-up periods.

### 2.3 Four task families in a persistent workspace

| Task | Assignment | Decisions and hidden evaluation |
| --- | --- | --- |
| Build | Construct a personalized predictor from limited recordings | Choose representation and fitting method; evaluate on another recording from the target participant |
| Investigate | Diagnose a predictor's failure and spend a limited budget on evidence | Compare hypotheses, choose a diagnostic or available recording, revise the model; measure final hidden performance and acquisition efficiency |
| Transfer | Extend the model to another supported workload/activity | Decide between refitting parameters and changing model structure; evaluate unseen recordings in the new and original conditions |
| Sustain | Improve across a sequence of sessions | Preserve experiments and useful candidates, select checkpoints, and retain earlier capabilities while adapting; evaluate retained and later conditions |

**Investigate example:** a starter model fits short bouts but underpredicts late recovery after longer bouts. The agent may inspect residuals, compare model structures, or request an additional available recording. It must state which result would distinguish its explanations. The case does not prescribe a particular correct equation.

**Persistent campaign:** initial calibration → Build → new evidence exposes a weakness → Investigate → another supported scenario → Transfer → further sessions → Sustain → final submission. Measurement/evaluation budgets remain spent. A change that improves one condition can harm another, making selection and regression checks consequential.

The long horizon comes from necessary dependent decisions, not forced tool-call counts. Each round must have evidence and a solvable prediction target. An example can use different exertion/recovery timescales or session drift only when supported by recordings, or as an explicitly synthetic structural test.

**Replay limitation:** a public or local dataset can reveal only recordings that exist. A request tool must offer an explicit available menu and charge its declared cost. It cannot fabricate the response to an unrecorded activity. A simulator may support a broader menu, but remains a synthetic panel. Comfortable voluntary recordings support data collection; the agent does not prescribe live exertion to participants.

### 2.4 Observable physiology, virtual participants, and the reference answer

A “body” means a virtual participant describing responses observable with our sensors. Pulse and movement do not uniquely reveal all internal physiology, and a good forecast does not establish the true mechanism.

1. Collect repeated, timestamped activity/recovery recordings with reference measurements where available, or begin with compatible public wearable datasets.
2. Fit multiple candidate response families using the training pool. Use a separate development pool for realism checks and task design. Keep final-test participants out of both.
3. Generate supported physiological variation, separately modeling measurement noise, gaps, and artifacts. Preserve the real pipeline's input fields.
4. Compare synthetic and real behavior on development recordings, including recovery, activity relationships, and variation. A believable plot alone is insufficient validation.
5. Freeze synthetic generalization tests and validate transfer on untouched real recordings. Synthetic episodes do not add independently measured people.

| Evaluation panel | Reference answer | Claim boundary |
| --- | --- | --- |
| Synthetic | Hidden generator's response to the episode inputs | Predictive skill inside that simulator; parameter recovery only if identifiable |
| Public real recordings | Hidden measured response from a compatible public dataset | Performance on that dataset and device profile |
| Our device recordings | Hidden measured response, preferably independently referenced | Performance on the tested device, people, and sessions |

Reference ECG/chest-strap measurements may be evaluator-only targets. A processed PPG estimate is a sensor-derived target, not an independent heart-rate reference. Do not claim true physiological recovery through a PPG dropout without a valid reference. Report each panel separately.

**Public-data candidates:** [PhysioNet structured stress/exercise recordings](https://physionet.org/content/wearable-device-dataset/1.0.1/) include cycling, cooldown, PPG, and motion, but heart-rate estimates derive from PPG; [PPG-DaLiA](https://archive.ics.uci.edu/dataset/495/ppg+dalia) provides wrist PPG/motion with chest ECG reference but is primarily daily-activity data. Audit actual recording coverage, quality, participant matching, and terms before enabling a task. Public data can develop the environment while device acquisition proceeds; it does not validate our hardware.

### 2.5 RL environments and post-training

**The same workspace supports evaluation and training episodes.** The benchmark should produce trajectories suitable for post-training; the learning method remains Nikhil's decision.

| Training environment | Capability learned across episodes |
| --- | --- |
| Model development | Inspect → build → evaluate → revise → select, including executable model generation |
| Experimental investigation | Maintain competing hypotheses → choose informative observations → interpret results → update |
| Continual adaptation | Handle successive scenarios → repair weak conditions → preserve earlier performance |

| Environment operation | Contract |
| --- | --- |
| Reset | Load participant case, allowed evidence, starter model, task profile, and budget |
| Observation | Accessible files, tool outcomes, experimental history, and remaining resources |
| Action | Inspect, edit, fit, evaluate, request an available recording, compare, revert, or submit |
| Transition | Execute the action; update workspace, immutable trace, and resource ledger |
| Reward and termination | Return declared training-reward components; end on submission, exhausted budget, or a defined invalid run |

**Two learning processes:** inside an episode, the agent changes predictor code and fitted parameters. Across training episodes, post-training changes the LLM's ability to plan experiments and use feedback. During final evaluation, LLM weights stay fixed while allowed within-episode personalization remains available.

Expose a machine-readable reward interface containing execution validity, hidden training-pool predictive error, transfer/retention, experiment cost, judge criteria, and benchmark version. Agent-visible development feedback and private trainer reward are distinct. Private reward targets belong to training episodes, never the final-test panel. No hidden-test result becomes another action-selection signal during a scored evaluation.

Use predictive outcomes to ground rewards and judge criteria to assess research decisions. Reward improvement that survives separate recordings, rather than verbosity, number of hypotheses, or merely running more experiments. Do not select arbitrary weights, RL algorithm, teacher, or optimizer here; Nikhil owns those decisions and their calibration. Broad training-data sufficiency remains unresolved until recordings and pilot episodes exist.

### 2.6 Required decision justifications and experiment history

Every consequential research action must have a concise decision record **before its result is returned**: model changes, fitting experiments, budgeted recording requests, evaluation calls, reversions, and final model selection. Routine file inspection may use a short purpose statement; do not force a long essay per tool call.

| Decision-record field | Required content |
| --- | --- |
| Evidence | Available observation, plot, residual, or earlier result with a source reference |
| Hypothesis | Proposed explanation and relevant uncertainty |
| Action | Exact experiment, code change, or selection |
| Expected diagnostic result | What result would support or contradict the hypothesis |
| Resource use | Declared measurement/evaluation cost and remaining budget |

**Example:** “Late-recovery residuals remain negative in development bout B. A second recovery timescale may help. Compare one- and two-timescale models using identical fitting data and evaluate on another development bout. Improvement there would support the additional component; otherwise revert. Cost: one evaluation call.”

After the result, append the observation, interpretation, and resulting decision. Keep earlier entries immutable. The final report identifies the selected artifact, why alternatives were rejected, remaining failures, and limitations.

This is the agent's observable stated rationale, not privileged access to its private internal reasoning. Judge consistency with available evidence and actual actions. A plausible retrospective explanation does not replace a missing pre-action record.

### 2.7 Scoring the decisions, final actions, and predictor

**LLM-as-a-judge is a required evaluation layer for the decision trace, final actions, and their justifications.** It receives the relevant observations, recorded rationale, actual tool call/code change, and result, and grades short expert-authored criteria with evidence references.

| Judge dimension | Example criterion |
| --- | --- |
| Evidence use | The decision cites evidence available at that time and accurately describes it |
| Experiment choice | The experiment can distinguish the stated explanations, given the available data |
| Feedback use | Interpretation and next action respond appropriately to contradictory results |
| Action consistency | Actual code changes/tool calls match the stated plan; claimed completion has evidence |
| Final selection | The chosen artifact and reasons agree with the experiment history; limitations are supported |

Judge early decisions using only evidence available then; use subsequent results to judge interpretation, not to reward hindsight. Blind model identities, keep rubrics and judge configuration fixed within a version, and audit a sample against independent expert review. Save verdicts, rationales, evidence references, and unresolved disagreements. The judge's reliability on our task must be measured, not borrowed from another benchmark.

Run the submitted program against hidden targets as well. Publish distinct score panels rather than letting persuasive prose erase incorrect predictions:

| Score panel | Measurements |
| --- | --- |
| Predictive performance | Target-specific error, improvement over fixed baselines, transfer, and retention |
| Research quality | LLM-judged decision records, actual final actions, evidence use, and justification |
| Execution/reliability | Invalid submissions, required-output coverage, runtime, resource use, reproducibility |

For heart rate, report raw MAE in bpm. Where exertion and recovery are both scored, average their errors equally; for a recovery-only task, use recovery MAE. Average recordings within participant and then participants equally. Report uncertainty at the participant/case level; adjacent samples are not independent participants. Additional targets require units, a suitable metric, and aggregation declared before scoring; do not mix bpm and sweat-related error without a justified normalization.

A baseline-relative measure may use `1 - agent_error / baseline_error`, alongside raw error. Handle near-zero denominators by a predeclared rule. Failures and missing required predictions cannot disappear from averages. Uncertainty or abstention require separately specified proper scoring/coverage rules. No newly chosen penalty may change rankings after results are observed.

The judge evaluates what the agent did and why; executable evaluation establishes whether its predictor works. Falsified data, future-label access, or an artifact that cannot run are integrity/execution failures, not errors a favorable explanation can compensate for. Phase 1's operator action/acknowledgment checks remain product validation, not the centerpiece of this research score.

### 2.8 Meaningful difficulty, search, and saturation

**Unsaturated means useful, solvable tasks still separate strong competitors. It cannot be guaranteed by a document or unlimited synthetic examples.** The aim is to make structural decisions and transfer matter, then test whether post-training helps.

| Difficulty control | Useful progression | Guardrail |
| --- | --- | --- |
| Evidence budget | More calibration → fewer recordings or costly additional observations | Required dynamics must still be inferable from permitted inputs |
| Model structure | Cases solved by parameter fitting → cases benefiting from a different representation | Ground real cases in development evidence; label synthetic structural tests |
| Dependency and horizon | One build → investigation → transfer → sustained updates | Necessary decisions and downstream consequences, not forced tool-call counts |
| Generalization | Familiar condition → unseen people, sessions, workload combinations, dynamics families | New seeds or renamed participants alone are insufficient |
| Feedback/resources | More experiments → fixed limited evaluations and compute | Equal budgets within comparisons; final-test scores never become interactive feedback |

Evaluate earlier and later conditions after revisions: local gains can cause regressions elsewhere. Preserve candidates and measure which model the agent selects. Report performance-versus-resource curves as well as final outcomes.

**Pilot diagnosis:** if simple fitting wins throughout, investigate whether agents add value in structure selection, experiment selection, or adaptation; retire uninformative tasks. If all methods fail, diagnose missing evidence, synchronization, labels, and impossible extrapolation. If all do well, retain a basic check and add supported challenges. Do not add arbitrary noise to make everyone fail. Broad model testing and conventional baselines must establish actual task difficulty.

**Research question:** does the post-trained U.S. model build better predictors on unseen cases with the same research budget, and does it use additional budget more effectively? A frozen frontier model with effective search may win. Report that outcome; do not handicap inference-time search to manufacture a need for training.

### 2.9 Competitors, splits, and evaluation controls

| Competitor | Purpose |
| --- | --- |
| Numerical baselines | Persistence where appropriate, fixed population predictor, direct personalized fitting |
| Automatic model selection | Test whether conventional fitting/search solves the task |
| Frozen U.S. checkpoint with strong prompting | Establish specialization baseline |
| Frozen LLM with code-search/experiment management | Compare post-training with inference-time search |
| Specialized U.S. checkpoint and selected Chinese frontier agents | Scoped comparison after training exists, with exact model versions recorded |

Freeze prompts, decoding settings, tools, input contracts, compute limits, evaluation calls, model-selection rules, and case batches. Repeat runs and report variability. Report tokens, API cost, inference latency, and fitting resources separately; identical hosted inference compute cannot be assumed.

| Split level | Rule |
| --- | --- |
| Across participants | Agent-training pool supports generator fitting and training; separate development participants support design choices; final-test people remain untouched |
| Within an episode | Explicitly supplied support recordings allow personalization; development feedback is bounded; another recording stays hidden |
| Across campaigns | Hold out meaningful scenario structures/combinations and synthetic model families where feasible, not just seeds |

Keep complete sessions together, avoid overlapping windows across splits, and fit preprocessing/normalizers using permitted data only. Online features never use samples beyond the cutoff. Exclude hidden parameters, labels, and seeds from visible metadata. Public datasets may have appeared in model training; their held-out split is not proof of pretraining-contamination resistance. Fresh local recordings provide a separate transfer panel.

Version tasks, metrics, processing, artifact interfaces, and case manifests. Freeze each leaderboard's test batch. Renew cases in a new version with anchor baselines rather than silently changing difficulty. [Biological ML leakage guidance](https://www.nature.com/articles/s41592-024-02362-y) motivates dependency-aware splitting; [RE-Bench](https://arxiv.org/abs/2411.15114) motivates resource-dependent comparisons, without establishing present-day results for this task.

### 2.10 Sensor extensibility and the first research campaign

Keep the workspace independent of any one sensor using a small adapter and task profile:

| Component | Responsibility |
| --- | --- |
| Sensor adapter and recording store | Preserve immutable raw data; normalize timestamps, units, quality, device versions, and provenance |
| Episode builder | Select permitted support/development/hidden recordings and available acquisition menus |
| Participant generator | Produce explicitly synthetic practice cases grounded in allowed training recordings |
| Agent runner and evaluator | Enforce budgets, execute predictors, preserve decisions, compute errors and judge criteria |
| Task profile | Declare available channels, target/reference, cutoff/horizon, conditions, metrics, and budget |

The recording contract includes pseudonymous participant/session IDs, relative time, available channels with units, validity masks, activity markers, and device/processing versions. Missing channels remain missing. Sweat-related inputs must identify the actual measurement, such as conductance or a measured chemical concentration, rather than labeling every related signal “sweat.” EDA/GSR alone does not establish sweat rate or hydration.

**First environment:** Build → Investigate → Transfer in one persistent workspace, with a working conventional baseline, an explicit menu of existing recordings, bounded development feedback, required decision justifications, and a final executable submission. Add Sustain when repeated recordings support it. This is a research design, not evidence that any dataset already satisfies its coverage requirements.

Confirm actual sensor outputs and reference availability, audit compatible public data, establish a numerical baseline on untouched recordings, and calibrate difficulty using development cases before freezing a scored version. Keep live, replay, public-data, and synthetic sources visibly labeled. No full-body, causal-mechanism, medical-readiness, or treatment-efficacy claim follows from a recovery-prediction pilot.

### 2.11 Fine-tuning — reserved for Nikhil

**Owner: Nikhil. Decision: open.** Environment construction, base-model comparisons, trajectory capture, and reward-interface design can progress before selecting a learning recipe.

| Decision | Status |
| --- | --- |
| U.S. open-weight checkpoint, license, and comparator versions | Nikhil to choose |
| Training data, teacher/example-generation method, and data permissions | Nikhil to choose |
| Supervised training, RL, or sequence | Nikhil to choose |
| Reward combination, compute budget, optimizer, and hyperparameters | Nikhil to choose |
| Training success criteria and checkpoint selection | Define before final evaluation |

Training/evaluation model weights and the episode's physiological predictor are different artifacts. Do not claim a fine-tune, improved benchmark score, sufficient training corpus, or U.S.–China result until measured. A potential initial demonstration is one executable model-building campaign with inspectable decisions and hidden-recording results; full population validation extends beyond the hackathon.

### 2.12 Event context, source status, and remaining checks

The organizer's [Slack build guidelines](https://2026sddefense-dcs5018.slack.com/archives/C0C3LG6PKRB/p1790278909686629) explicitly name Operational Phenotyping with Minimal Sensors and Human Performance & Prolonged Field Care. The full message was re-read during the research discussion. Phase 1 demonstrates measurement and operator follow-through; Phase 2 develops the modeling research capability. Judge-specific presentation advice in Phase 1 remains inference from public backgrounds, not a verified rubric.

| Source | Review status | Use and remaining limitation |
| --- | --- | --- |
| [Luma event page](https://luma.com/kfgh2fvu) | Public HTML inspected in earlier pass; later browser fetch failed | Event format and tracks; do not treat failed refresh as new verification |
| [Hardware announcement](https://2026sddefense-dcs5018.slack.com/archives/C0C3LG6PKRB/p1792194966030419) and workbook | Announcement/thread and workbook rows reviewed earlier | Candidate hardware; receipt, functioning, and outputs need direct checks |
| [Schedule thread](https://2026sddefense-dcs5018.slack.com/archives/C0C3TDMLM4Z/p1791927436967009) | Organizer reply read earlier | Event schedule, not a verified per-team presentation limit |
| [October 13 announcement](https://luma.com/blast/uRhDv8llCV) | Roster sourced to user paste, not independently reproduced | Judge attendance remains provisional |
| RSI/APEX/HORIZON/AIDE and biology sources linked above | Primary descriptions/papers inspected in the discussion | Task-design precedent; actual difficulty, novelty, and judge reliability remain unmeasured |

Earlier Slack review covered returned announcement/general-question histories and relevant hardware, schedule, overnight-access, and build-guideline threads. No official rubric or per-team presentation limit was found; this does not prove none exists. Remaining event checks include current judge roster, scoring weights, functioning equipment, proposal status, and final-deliverable instructions.

**First decisive research test:** compare a conventional personalized fitter and a coding agent on the same evidence and untouched recording; extend the comparison to investigation and transfer when coverage exists. Establish a repeatable failure and a valid evaluator before committing to a large post-training run.
