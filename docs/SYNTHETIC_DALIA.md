# PPG-DaLiA-grounded synthetic feature recordings

The new `benchmark.dalia_synthetic` path fits a statistical generator to real PPG-DaLiA reference recordings. The former exponential fixtures remain deterministic software checks; they are not the statistical source for this dataset. No old Luna score or Elo has been relabeled as a result on this dataset.

## Implemented boundary

The dataset preserves **SoliderPhys-Bench's processed feature contract**, rather than the original high-rate waveform pickle format:

- One `.npz` and metadata `.json` per synthetic recording; a dataset manifest records synthetic provenance, channel units, source attribution, seeds, and generator/model hashes.
- Equal-length one-dimensional arrays: `t`, `activity`, `hr_ecg`, `acc_wrist_enmo`, `acc_wrist_std`, `acc_chest_enmo`, `acc_chest_std`, `eda_wrist`, `temp_wrist`, `resp_chest_std`.
- A two-second grid beginning at t=8, matching the source adapter's end-of-window convention. The HR field preserves the interface name, but **contains generated numbers, not measured ECG**. Raw PPG/ECG waveforms are not generated.
- Demographic keys exist for compatibility but are explicitly unavailable; demographic values are not invented.
- The existing activity-conditioned task receives 60 seconds of history (30 samples) and forecasts 120 seconds (60 samples). Future covariates are permitted by that task; future HR is excluded from predictor inputs.

This is the `physbench/` data contract. The separate `benchmark/` four-stage action/predictor protocol has not been unified with SoliderPhys-Bench by this change. The generated dataset is consumable by the real-data runner; these files alone do not create a validated four-stage campaign.

## What comes from real data

Fitting reads only explicitly selected reference participants **S1, S4, S8, S12**. The fitting API rejects the development and final-test participants. Each source `.npz` is hashed. The original SoliderPhys-Bench adapter is reused for preprocessing, including its restricted pickle loader; no alternative conversion is invented.

The generator learns empirical channel quantiles, activity-specific cross-channel autoregressive dynamics and innovation covariance, between-reference-recording variation, activity durations, protocol orders/transition structure, and missing fractions. Each output samples a fitted reference dynamics model, resamples durations for a recorded activity order, and evolves new correlated random innovations. It does not copy raw trajectories. There are only four fitted source participants: additional synthetic realizations do not add independently measured people, prove anonymity, or establish unseen-family generalization.

The implementation uses empirical quantile transforms with an activity-conditioned VAR(1) in Gaussian coordinates. Gaussian innovations, activity-mean anchoring, ridge regularization, spectral stabilization, and fitted-range caps are declared modeling assumptions. Means, covariance, and time scales are estimated from data; physiological equations are not inferred uniquely. Durations and protocol order are statistical templates, not evidence for unrecorded workloads, carried loads, environmental changes, or repeated-session physiology.

Keep the fitted generator JSON, parameters, and seeds in trusted infrastructure. A benchmark participant should receive only the recordings/contexts its episode permits. Existing agent execution/isolation limits still apply.

## Executed dataset and fidelity audit

The official UCI archive was downloaded. Only the four reference and four designated development participants were extracted and converted; S6 and the six real final-test participants were not unpickled or converted. Source: [PPG-DaLiA, Reiss, Indlekofer and Schmidt (2019), UCI, CC BY 4.0](https://doi.org/10.24432/C53890).

**Run:** `.benchmark-runs/dalia-realstats-02/`, generator `dalia-feature-copula-var-v1`, seeds 1000–1011. It contains **12 synthetic recordings, 52,641 rows**, with durations about 124–166 minutes. Fitting used 17,166 rows from four reference people. A separate descriptive audit used 18,455 rows from four development people. Real final-test recordings remain uninspected. Generator calibration considered fit-cohort fidelity; development fidelity is diagnostic and never a source of fit statistics.

| Statistic | Real fit cohort | Synthetic | Interpretation |
| --- | ---: | ---: | --- |
| Mean HR (bpm) | 77.39 | 78.68 | Similar aggregate mean |
| HR standard deviation (bpm) | 15.50 | 13.75 | Understates variability |
| HR lag-one correlation | 0.9906 | 0.9842 | Similar but less persistent |
| Mean wrist ENMO (g) | 0.075 | 0.073 | Close aggregate movement level |
| Wrist ENMO standard deviation (g) | 0.087 | 0.088 | Close aggregate movement spread |
| Sitting mean HR (bpm) | 53.9 | 62.3 | Material activity-specific mismatch |
| Stairs mean HR (bpm) | 99.0 | 95.4 | Residual activity mismatch |
| Cycling mean HR (bpm) | 104.6 | 102.0 | Residual activity mismatch |
| Mean skin temperature (°C) | 32.19 | 31.30 | Residual drift/distribution mismatch |

Mean absolute difference over the full 8×8 channel correlation matrices is **0.0825** against fit recordings and **0.0885** against development recordings. Real development mean HR is **86.54 bpm**, versus synthetic **78.68 bpm**: fitting four references does not reproduce the wider population automatically. Full reports also include quantiles, missingness, activity occupancy, duration distributions, transition counts, per-activity HR variation, and channel correlations. No fidelity pass threshold or physiological validation is claimed.

All 12 recordings passed the **existing SoliderPhys-Bench** loader, support/development/transfer splits, window shapes, exclusion of future HR from predictor inputs, prediction validation, and numerical scoring using a persistence probe. The emitted `task-synthetic-contract-smoke.json` is an integration profile; all its synthetic seeds have been inspected, so it is not an untouched final-test batch. No LLM comparison, training run, or synthetic-to-real prediction gain has been measured here.

## Reproduce

Dependencies are in `requirements-synthetic.txt`; the executed generator used NumPy 2.4.4, and the compatibility tool used PyYAML 6.0.3. An existing SoliderPhys-Bench checkout is needed only to reuse its source adapter and exercise its runner. Generator fit/generate/audit operate independently on processed `.npz` files.

```sh
python3 -m pip install -r requirements-synthetic.txt
# Download the official UCI archive to a local path, then reuse the existing adapter.
python3 scripts/prepare_dalia_reference.py /path/to/ppg-dalia.zip .benchmark-runs/new-dalia-source \
  --physbench-source /path/to/physbench-checkout
python3 -m benchmark.dalia_synthetic fit .benchmark-runs/new-dalia-source/processed \
  .benchmark-runs/new-dalia-fit.json --subjects S1 S4 S8 S12
python3 -m benchmark.dalia_synthetic generate .benchmark-runs/new-dalia-fit.json \
  .benchmark-runs/new-dalia-synthetic --count 12 --seed 1000
python3 -m benchmark.dalia_synthetic audit .benchmark-runs/new-dalia-source/processed \
  .benchmark-runs/new-dalia-synthetic .benchmark-runs/new-dalia-audit.json --subjects S2 S3 S7 S10
python3 scripts/check_dalia_synthetic.py .benchmark-runs/new-dalia-synthetic \
  --physbench-source /path/to/physbench-checkout --output .benchmark-runs/new-dalia-compatibility.json
```

Use fresh output locations; generation/preparation refuse an existing output directory. Bulk data, fitted private parameters, and run artifacts remain ignored by Git. They are available locally, not published as a downloadable release.

## Next decisive checks

Improve activity-specific and low-frequency fidelity using reference/development data while keeping final-test people untouched. Compare fixed predictors trained/fitted on synthetic versus real-reference data on real development cases, then freeze a protocol for untouched real evaluation. Measure meaningful difficulty with conventional and LLM competitors on this data profile. Richer real-grounded simulations are a better basis for those experiments, but do not by themselves establish post-training value or justify claiming a full-body model.

## Temperature semantics in the environment

`temp_wrist` is the PPG-DaLiA skin-temperature channel. It already participates in the joint statistical generator, which fits cross-channel and temporal relationships from permitted real reference recordings. Synthetic recordings now carry explicit `skin_temperature` provenance in their manifest and per-recording metadata. The metadata-explicit output is `.benchmark-runs/dalia-realstats-03/` (the earlier numerical audits refer to `02`; no new physiological validation is implied).

Every benchmark observation now distinguishes optional measured/synthetic wrist skin temperature, IMU chip temperature (`imu_chip_temp_c`), and a separately calibrated skin estimate (`estimated_skin_temp_c`). No channel is populated merely by defining it. Chip temperature is never converted into a claimed skin measurement by an arbitrary offset. Calibrating such an estimate requires synchronized chip and independent skin-reference recordings, validation across relevant contact/ambient/device-heating conditions, and reported error. That calibration is not implemented or validated here.

The current HR forecast score remains in bpm. Actual future skin measurements cannot be added as forecast covariates without a separate explicitly conditioned task. No skin/core-temperature prediction accuracy is claimed, and no hardware firmware, calibration, or sensor readings were changed.

## Current agent-facing temperature policy

Chip-temperature fields and the uncalibrated-estimate field are excluded from public observations, exported recordings, and predictor execution inputs. The public temperature contract describes only optional skin temperature and its forecast boundary. Generated skin metadata no longer mentions chip temperature. The current dataset is `.benchmark-runs/dalia-realstats-04/`; earlier runs remain historical artifacts. A regression test injects prohibited fields in both workspace types and confirms that skin temperature survives while those fields do not reach the participant or executor.
