# Data dictionary

Source: EVOBREATH 2015 (Mateos, Mendeley Data, DOI 10.17632/rhmdzmx6kk.1, CC BY 4.0).
Adults aged 23–50, fasted, treadmill at 4 km/h, 0% grade, 11 minutes per condition, backloads
of 0, 5, 10 and 15 kg; indirect calorimetry (Jaeger Master Screen CPX). Laboratory at 21.5–23 °C.

## Training rows (`data/train.jsonl`)

| Field | Unit | Notes |
|---|---|---|
| `person_key` | – | Opaque grouping key for grouped validation. Not a feature. |
| `sex` | `M`/`F` | Optional declared covariate. If your model uses it, say so; results are reported by sex regardless. |
| `body_mass_kg` | kg | Digital scale, 0.1 kg |
| `stature_cm` | cm | Stadiometer, 0.1 cm |
| `age_years` | years | |
| `load_kg` | kg | Carried backload: 0, 5, 10 or 15 |
| `speed_m_s` | m/s | Always 4 / 3.6 ≈ 1.111 |
| `grade_fraction` | – | Always 0 |
| `reference_w` | W | Gross metabolic power. Source kcal/day × 4184 / 86400 |

Rows whose source target cell was non-numeric (`-`) are absent; they were not imputed.

## Requests

Same fields as a training row except `person_key` and `reference_w`, plus `request_id`, `track`, and
`calibration` (null unless the track reveals the person's measured 0 kg power as
`{"load_kg": 0, "measured_w": ...}`).

## Not provided

Femur length, bi-iliac breadth, fat mass, fat-free mass and resting metabolic rate exist in the source
but are laboratory measurements outside this task's input tier. No HR, motion, terrain or speed variation.
