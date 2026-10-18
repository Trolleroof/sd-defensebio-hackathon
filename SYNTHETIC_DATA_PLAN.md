# Synthetic data plan — blood-side physiology + our sensor data

**Status — October 17, 2026:** proposal. Nothing below is implemented or validated. Companion to [PLAN.md](PLAN.md) §2.4 (virtual participants) and §2.5 (RL environments). All generated data stays labeled **SYNTHETIC**.

**Scope note:** no `blood/` or `proxy/` folder exists in this repo or under the home directory. This plan treats "blood / proxy for blood" as public datasets that measure a blood variable (blood pressure, lactate, glucose, SpO2, blood volume) or a lab stand-in for one (lower-body negative pressure for blood loss, ventilatory threshold for lactate threshold). Point us at the folder if it lives somewhere else.

## 1. What our own data actually contains

From `sensor/test1–4_*.csv` and `dashboard/sessions/*`:

| Stream | Device | What we have | Physiological read |
| --- | --- | --- | --- |
| HR (`*_hr.csv`) | Polar Verity Sense (optical) | Integer bpm, ~1 Hz. Test 4 ≈ 6 min, 61–75 bpm. Test 1 ≈ 1 min, 71–91 bpm | Seated or light activity only. **No exertion or recovery block yet** |
| PPI (`*_ppi.csv`) | Verity Sense | ~1.2 rows/s with `error_ms`, `blocker`, `skin_contact`. Median 850–880 ms (≈ 68–70 bpm) | Enough beats for short-window RMSSD. The first rows of each file show a 380–400 ms interval with `blocker=1`, a startup artifact the generator must reproduce |
| ACC (`*_acc.csv`) | Verity Sense | ~54 Hz, 3-axis mG | Posture and gross movement |
| Raw PPG (`*_ppg.csv`) | Verity Sense | **All four files are empty (header only)** | No waveform morphology yet. Pulse amplitude and shape features are unavailable |
| IMU | LSM6DSOX | 52 Hz acc + gyro + die temperature | Rotation and activity. Die temperature is board/air, not body |
| Dashboard sessions | Server | 1 Hz `samples.csv` and `events.jsonl` (alerts, operator actions) | The synthetic teammates there come from a hand-tuned first-order model (`dashboard/server.py` `SimSource`), not fitted to anything |

**Bottom line:** our data is one person, about 8 minutes, at rest, with no reference strap. Its main value is the **observation model**: how this exact device quantizes, drops, flags, and corrupts physiology. It cannot yet ground response dynamics (gain, time constants, recovery). Public data must supply those.

## 2. Blood-side datasets and what they teach us

| Blood variable (or proxy) | Dataset | What's in it | What it gives the generator | Mismatch with our device |
| --- | --- | --- | --- | --- |
| **Blood volume loss**, via LBNP (the standard human proxy for hemorrhage) | US Army ISR LBNP studies (218 subjects, 500 Hz Finometer PPG), used for the Compensatory Reserve work ([MDPI 2023](https://www.mdpi.com/2306-5354/10/5/612)) | Continuous PPG/BP to presyncope | Shape of compensation: HR rises late and variably, PPG amplitude and pulse pressure shrink early, HRV falls, then decompensation comes abruptly | **Public access not confirmed.** It looks like a request-only dataset. Until it's obtained, use literature-reported trajectories as priors, labeled as such ([review](https://link.springer.com/article/10.1007/s10974-022-09635-z)) |
| **Blood pressure** | [PulseDB](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9944565/) (MIMIC-III + VitalDB): 5,361 subjects, 14,570 h of ECG/PPG/ABP in 10 s segments | Finger PPG + arterial line | PPG morphology ↔ BP relationships; beat-level variability | ICU and surgical patients, finger site, at rest. Useful for morphology priors, not exercise dynamics |
| **Blood pressure + SpO2 during activity** | [PhysioNet Pulse Transit Time PPG](https://physionet.org/content/pulse-transit-time-ppg/1.1.0/) | Multi-site PPG, IMU, ECG, BP, SpO2, sensor attachment pressure | Motion-artifact ↔ IMU coupling; contact-pressure effects on PPG | Small cohort; check which activities were recorded before relying on it |
| **Lactate**, via ventilatory threshold | [Malaga treadmill maximal tests](https://physionet.org/content/treadmill-exercise-cardioresp/): 992 tests, breath-by-breath HR/VO2/VCO2/VE + speed | Graded exercise to maximum | **The best prior for HR-response dynamics:** HR gain vs workload, HRmax, the threshold where VCO2/VE break away (a lactate-threshold proxy), inter-individual spread | Lab treadmill with chest HR. Distribution of people is athletes aged 10–63 |
| **Lactate (direct)** | [Graded incremental test files](https://zenodo.org/records/10841412): 835 cycle/row/run tests | Incremental tests | Possibly direct lactate–HR pairs | **Verify the fields:** search results don't confirm lactate columns |
| **Glucose** | [BIG IDEAs Lab glycemic + wearables](https://physionet.org/content/big-ideas-glycemic-wearable/): 16 people, 10 days, Dexcom G6 every 5 min + Empatica E4 (BVP 64 Hz, ACC, EDA, TEMP) | Free-living, standardized meals | Slow metabolic state alongside wrist PPG/ACC; realistic multi-day drift and gaps | The glucose link to HR is weak and confounded. It's a long-horizon context channel, not a first target |
| **Exercise stress (no blood)** | [Wearable stress/exercise, Zenodo](https://zenodo.org/records/13993658) (E4); [PPG-DaLiA](https://archive.ics.uci.edu/ml/datasets/PPG-DaLiA) (already loaded in `physbench/data/dalia.py`) | Wrist PPG + ACC, ECG reference (DaLiA) | Free-living HR-vs-activity transfer; wrist motion artifacts | DaLiA is mostly daily activities, with only brief high exertion |

### Physiological picture across these sources

- **HR is the most available signal and the least specific.** Exertion, heat, stress, and blood loss all raise it. Blood loss makes this worse: HR often stays flat until a large volume is lost, and some people slow down (vasovagal response). Any synthetic "blood" scenario that only bumps HR teaches the wrong thing.
- **Early blood-volume signals sit in the pulse waveform**, not the rate: shrinking PPG amplitude, pulse-shape change, falling HRV. We currently capture none of that because the PPG files are empty. Fixing raw PPG capture is a prerequisite for any blood-proxy claim on our device.
- **Lactate leaves a slope change in the HR–workload curve and recovery time**, not a level. The ventilatory threshold in Malaga gives a per-person breakpoint. That breakpoint is the kind of structural change our Transfer and Investigate tasks need: a model fitted below threshold underpredicts above it.
- **Every relationship is individual.** The spread across Malaga's 992 tests, and across PulseDB's calibration-based vs calibration-free gap, is the argument for personalized models. That spread is what synthetic participants should sample.

## 3. What makes synthetic data valuable here (and what doesn't)

Valuable, in order:

1. **Benchmark episodes where a structural decision matters.** These include threshold behavior, slow cardiac drift, and recovery that depends on prior load. Real recordings from one person can't supply enough of these; a generator with hidden structure can. This is PLAN.md §2.3 Investigate and Transfer.
2. **Realistic device failure modes, so alert logic gets tested.** Startup PPI artifacts, contact loss, integer quantization, BLE gaps, PPG-mode HR silence. Only our own recordings can supply this, and it's what keeps the dashboard honest.
3. **Labeled operator-demo scenarios** beyond today's three scripted events. One example is an "unexplained HR rise with falling HRV at low movement" case, shown with its evidence and labeled SYNTHETIC.

Not valuable: more seeds of the same curve, synthetic "blood loss detection accuracy," or any number presented as evidence about real people.

## 4. Generator design: four layers, each fit from the right source

```text
Scenario layer      activity schedule, posture changes, events (distress, dropout, synthetic decompensation)
        ↓
Latent physiology   per-participant parameters → exertion drive, HR kinetics, threshold, drift, HRV,
                    optional latent states: volume deficit, lactate-like load, thermal load
        ↓
Observation model   device-specific: Polar HR quantization/cadence, PPI error/blocker/contact,
                    IMU noise/orientation, gaps, startup artifacts, PPG-mode HR silence
        ↓
Output              same files as the real pipeline (sensor/*_hr|ppi|acc.csv, samples.csv, events.jsonl)
                    + a separate hidden truth/ folder (latent states, params, seed) that agents never see
```

**Layer 1 — scenario (hand-specified, not fit).** Activity plans built from the real activity menu (rest, walk, run, stand-up), plus the dashboard's existing events. Generated schedules can only include activities the observation model has actually seen.

**Layer 2 — latent physiology (fit from public data).**
- Base model: HR = rest + gain·drive, with separate on and off time constants (one fast and one slow recovery component) and slow cardiac drift. Fit per test on Malaga (speed → HR), then take the population distribution and covariance of the parameters as the participant prior. Cross-check on DaLiA wrist activity.
- Threshold structure: a per-participant breakpoint from the Malaga ventilatory threshold. Above it, gain steepens and recovery lengthens with accumulated load. This is an explicitly modeled lactate-like state, not measured lactate.
- HRV: RMSSD falling with HR and load, fit from DaLiA/ECG and our PPI.
- **Volume-deficit state (optional, literature-parameterized):** compensatory HR rise delayed and variable, pulse-amplitude reduction, HRV suppression, an abrupt decompensation point. It includes a "paradoxical bradycardia" subtype. Its parameters come from published LBNP trajectories until the ISR data is obtained. It's tagged `prior_source: literature` in the hidden truth.
- Families: sample from **several** model structures (first-order, two-compartment, threshold) so the generator isn't one equation an agent can just recover. Hold out a family for synthetic generalization tests (PLAN.md §2.8).

**Layer 3 — observation model (fit from our recordings).** This is where our internal data is irreplaceable:
- HR: integer rounding, ~1 Hz cadence with the jitter seen in `test4_hr.csv`, gap distribution.
- PPI: `error_ms` distribution, rate of startup artifacts with `blocker=1`, beat drop and duplication rate, skin-contact loss.
- ACC/IMU: noise floor and gravity orientation by posture from `*_acc.csv` and LSM6DSOX logs. Die-temperature drift as board-thermal, decoupled from physiology.
- Device modes: no HR while raw PPG streams (`sensor/polar_stream.py` notes this).
- Fit the noise process separately from physiology, as PLAN.md §2.4 step 3 requires.

**Layer 4 — output contract.** Same columns and units as the real pipeline, so `physbench/episodes.py`, the baselines, and `dashboard/server.py` consume synthetic and real data through one path. The `source` column is always `SYNTHETIC`. Hidden truth lives outside anything the sandbox mounts.

## 5. Internal data we need to collect (the cheapest, highest-value step)

Our current recordings can't ground dynamics. With consent and comfort-limited effort, ideally with an H10 chest strap as the reference:

1. **Exertion/recovery session:** 3 min seated → 3 min walk → 2 min brisk walk, stairs, or jog → 6 min seated recovery. Verity HR+PPI+ACC and LSM6DSOX. Repeat on a second day. This is the minimum for fitting Layer 2 per person and for a real hidden-session test.
2. **Raw PPG session:** the same protocol in `--ppg` mode. Today's PPG files are empty, so first confirm capture works. This gives pulse amplitude, our only waveform-level route toward volume-status signals.
3. **Active stand test (a safe in-house central-hypovolemia proxy):** 5 min lying down → stand still 3–5 min. Standing pools roughly 0.5–1 L of blood in the legs and produces a mild, self-limiting version of the LBNP response (HR rise, HRV drop, PPG amplitude change). It's the most defensible way to anchor the "volume-deficit" layer with our own device. Stop immediately if the participant gets lightheaded. No prescribed exertion or deeper stressor.
4. **Artifact session:** deliberate loose strap, sensor removal and reattach, arm motion at rest. This labels failure modes for Layer 3.

## 6. Validation gates (before synthetic data is used for anything scored)

| Gate | Test | Pass condition |
| --- | --- | --- |
| Marginals | HR, PPI, RMSSD, movement distributions, synthetic vs held-out real (development pool only) | Within the spread across real participants, not just one person |
| Dynamics | On/off time constants and recovery half-time on synthetic vs real exertion blocks | Same distribution; recovery depends on prior load the way the data shows |
| Artifacts | Gap, artifact, and contact-loss rates vs our recordings | Matches; dashboard alerts fire at similar rates on replay vs synthetic |
| Discriminator | Small classifier trying to tell real from synthetic windows | Near chance; inspect what it uses if not |
| TSTR | Train baselines on synthetic → test on untouched real (DaLiA dev participants, then our sessions) | Does as well as training on the matching amount of real data. If not, synthetic data is a practice set only |
| Identifiability | Can a direct fitter recover generator parameters from episode-length data? | Only claim parameter recovery for parameters that pass |

Keep synthetic, public-real, and our-device results in separate panels (PLAN.md §2.4 table).

## 7. Build order

| Step | Work | Output | Depends on |
| --- | --- | --- | --- |
| 1 | Collect sessions 1 and 4 (§5); fix raw-PPG capture | Real exertion + artifact data | Hardware, consent |
| 2 | Observation model from `sensor/*` + new sessions | `physbench/synth/observe.py` + fitted noise params | Step 1 (start on existing files now) |
| 3 | Download Malaga; fit HR kinetics + threshold per test; save the prior | `physbench/synth/priors/malaga.json` | None (can start now) |
| 4 | Latent generator with 2–3 model families → real output contract | `physbench/synth/generate.py`, `truth/` separation | Steps 2–3 |
| 5 | Validation gates on the DaLiA dev pool + our sessions | Gate report | Step 4 |
| 6 | Replace `SimSource` in the dashboard with generated scenarios (still labeled SYNTHETIC) | Richer demo team | Step 4 |
| 7 | Synthetic Investigate/Transfer episodes in `physbench/episodes.py` | New task profiles | Step 5 |
| 8 | Stand-test session (§5.3) + literature priors → optional volume-deficit layer | Labeled synthetic decompensation scenarios | Step 1, raw PPG working |
| Later | Request ISR LBNP data; check Zenodo lactate fields; consider BIG IDEAs for multi-day drift | Better-grounded blood-side priors | Data access |

For the hackathon window, steps 2, 3, 4, and 6 are reachable. Steps 5 onward and the blood-volume layer are post-event unless data collection goes quickly.

## 8. Claim boundaries

- A synthetic decompensation scenario shows what the **dashboard and agents do** with such a pattern. It does not show that the wearable detects blood loss.
- Lactate-like and volume-deficit states are **modeled latent variables**, not measurements. Volume-deficit priors come from literature until a real LBNP or stand-test dataset grounds them.
- Synthetic participants add practice episodes, not people (PLAN.md §2.4 step 5). Any gain on synthetic episodes is reported as synthetic until it transfers to untouched real sessions.
- Optical HR and PPI from the Verity Sense are sensor-derived targets, not an independent reference.
