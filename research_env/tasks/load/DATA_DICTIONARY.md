# Data dictionary — EVOBREATH 2015 load-carriage task

Source: Mendeley Data `rhmdzmx6kk` version 1, DOI 10.17632/rhmdzmx6kk.1, CC BY 4.0. Attribution: Ana Mateos, CENIEH (Burgos, Spain), *Energetic expenditure during walking and load-carriage in humans. Dataset from EVOBREATH 2015*. Files: `Mateos2020_Dataset_WW2015.xls` (sha256 `51b42232…58da`, 38,400 bytes) and the protocol PDF `Mateos2020_Dataset_WW2015.pdf` (sha256 `5410a21e…57f5`); both match the SHA-256 published by the Mendeley files API.

Protocol (from the PDF): fasted morning indirect calorimetry (Jaeger Master Screen CPX), hp COSMOS treadmill, no inclination, 4 km/h, 11 minutes per trial, backloads of 0, 5, 10 and 15 kg. Laboratory 21.5–23 °C. RMR measured supine for 30 minutes.

## Request fields (one scored condition)

| Field | Meaning | Unit |
| --- | --- | --- |
| sex | `M` or `F`; declared optional covariate, subgroup-reported | — |
| age_years | Age | years |
| stature_cm | Stature (`Height`) | cm |
| body_mass_kg | Body mass (`Weight`), excludes load | kg |
| load_kg | Carried backload: 0, 5, 10 or 15 | kg |
| speed_m_s | 4/3.6 = 1.1111… (constant) | m/s |
| grade_percent | 0 (constant) | % |
| calibration | `null` (primary) or `{"load_kg": 0, "gross_w": …}` (calibration tier only) | W |

Output: `{"mean_w", "lower_w", "upper_w"}` gross metabolic power and a 90% interval, watts.

## Training rows (`data/train.json`)

Request fields (without `calibration`) plus `person` (opaque pseudonym), `gross_w` (label) and `quality_flags`.

## Conversions and conventions

- `EE_WW00/05/10/15` are rate equivalents in kcal/day. `gross_w = kcal_per_day * 4184 / 86400`. Example: 4855 kcal/day = 235.108 W.
- Target is **gross** energy expenditure while walking. `net = gross - RMR` changes the target and is kept only as a separate unscored controller field.
- Excluded from every request and from public training rows: RMR, fat mass, fat-free mass (bioimpedance), femur length and bi-iliac breadth, and any other load condition's target (except the declared unloaded calibration value in the calibration tier).

## Parsing and exclusions

The `DATA` sheet stores many numeric cells as text (e.g. `Height` in 29 of 48 rows). Each cell is parsed from its stored type: numbers directly; text only if it fully matches a decimal number; the literal `-` is a recorded missing value; anything else stops the import. Original cell coordinates, type and value are preserved in controller records. Three people have `-` for `EE_WW15`; those three conditions are excluded (reason `source_missing_dash`); the people remain. 48 people, 189 of 192 possible condition observations. Quality flags (not exclusions): gross walking cost at or below 1.5 × that person's measured RMR, and non-increasing condition costs within a person. Costs are never forced to be monotonic.

## Splits

Person-level, sex-stratified, deterministic SHA-256 order with salt `evobreath-load-split-v1`: 28 train / 10 development / 10 private test. RL agent-training tasks partition only the 28 train people (18 fit / 5 feedback / 5 reward). Test people are reserved for final evaluation.
