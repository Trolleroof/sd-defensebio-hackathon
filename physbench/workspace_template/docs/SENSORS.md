# Sensors and channels

Recordings come from a public field study of wearable heart-rate estimation during daily
activities (chest strap + wrist device). All channels are resampled onto one 0.5 Hz grid.
Each value at time `t` is computed only from data at or before `t`.

| Channel | Unit | Device | Meaning |
|---|---|---|---|
| `hr_ecg` | bpm | chest ECG strap | Heart rate from ECG R-peaks over the preceding 8 s. **Target.** Independent reference. |
| `acc_wrist_enmo` | g | wrist 3-axis accelerometer, 32 Hz | Mean of max(‖a‖ − 1 g, 0) over the preceding 2 s — dynamic movement load at the wrist |
| `acc_wrist_std` | g | wrist accelerometer | Std of ‖a‖ over the preceding 2 s |
| `acc_chest_enmo` | g | chest accelerometer, 700 Hz | Same as wrist, at the chest (trunk movement) |
| `acc_chest_std` | g | chest accelerometer | Std of ‖a‖ over the preceding 2 s |
| `activity` | id | protocol annotation | Activity being performed (see `docs/channels.json` → `activity_names`); 0 = transition between activities |

Notes:

- Wrist acceleration under-represents leg-driven activity (e.g. cycling with hands on a handlebar);
  the chest accelerometer captures trunk movement.
- Heart rate responds to load with a lag: it rises over tens of seconds after exertion starts and
  recovers over a minute or more afterwards; recovery depends on fitness and prior effort.
- Activity transitions (`activity` = 0) often involve walking between locations.
- Disabled channels (exist in the study, not provided in this task): wrist electrodermal activity,
  wrist skin temperature, chest respiration.
