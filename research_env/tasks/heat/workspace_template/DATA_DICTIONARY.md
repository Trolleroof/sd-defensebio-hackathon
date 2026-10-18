# Data dictionary

Source: CORE cycling heat dataset, Figshare 29120024 v1 (CC BY 4.0). Time is integer minutes since exercise start (1..60).

## Request (`data/example_request.json`)

| Field | Meaning |
|---|---|
| `request_id` | Opaque ID. Not a feature. |
| `tier` | `wearable` (primary) or `calibrated` |
| `cutoff_min` | Forecast origin. History includes every value stamped `<= cutoff_min`, so the current minute is available. |
| `horizons_min` | Subset of 5/10/20 with `cutoff + horizon <= 60`. It depends only on the cutoff, never on target availability. |
| `history.<field>` | `[[minute, value], ...]` ascending. Masked or missing values are absent, not filled. |
| `context` | `sex` (M/F), `age_yr`, `height_cm`, `mass_kg`, `bmi_kg_m2` |
| `calibration` | Calibrated tier only: `rectal_c` at minute 1 |

## History fields

| Field | Unit | Timing | Notes |
|---|---|---|---|
| `hr_bpm` | beats/min | Interval (t-1, t] stamped at t | **Sparse**: recorded only around minutes 6–10, 21–25, 36–40, and 51–55, so early cutoffs have no HR. |
| `device_skin_c` | °C | Every minute | Skin temperature reported by the CORE wearable. It is not core temperature and not a reference. |
| `work_rate_w` | W | Every minute | Prescribed cycling load; nearly constant within a person. |
| `ambient_c` | °C | Sparse (about 8 readings) | Measured chamber temperature |
| `ambient_rh_pct` | % | Sparse | Measured chamber relative humidity |

## Training manifest

`people[]` has `person` (opaque), `context`, `inputs` (`minute`, `field`, `value`, where `null` means masked), and `reference` (`minute`, `rectal_c`, `eligible`). Ineligible reference minutes are never scored.

## Excluded

These are never available: vendor CORE core-temperature estimates, laboratory reference skin thermistors, VO2 and VO2peak, RPE, ultrasound subcutaneous fat, and rectal history (except the calibrated-tier minute-1 value). Future values after the cutoff are never available.

## Quality masks (heat-quality-v1)

Masked values are omitted, never replaced. Three HR cells are implausible (1176, 1556, 1152 bpm), and one isolated rectal dropout (35.0 °C between values near 38.4 °C) is ineligible as a target. One person's recording ends at minute 56.
