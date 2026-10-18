# Core-temperature forecasting — heat-v1

Build and submit an executable predictor of independently measured rectal temperature
5, 10 and 20 minutes ahead, in degrees Celsius. Train on `data/train/manifest.json`;
use controller development calls for feedback. Choose features, equations, fitting,
experiments and the final candidate yourself. Prediction code implements
`predict(artifact_dir, request)` in `artifact/inference.py`.

Requests contain `history` with time_min, hr_bpm, skin_c, ambient_c, humidity_pct;
`demographics` with age_years, mass_kg, height_cm; cutoff_min; horizons_min; and state.
History includes only measurements available through the cutoff, including the
current minute. Interval observations become available at their right edge.
Missing/out-of-range HR is omitted and the last observed HR is carried forward.
Skin is the measured YSI skin channel. No rectal history, vendor core estimate,
VO2, person IDs or future measurements may be used at inference.

Return exactly `{"forecasts":[{"horizon_min":5,"mean_c":37.5,"lower_c":37.0,"upper_c":38.0},
{"horizon_min":10,"mean_c":37.6,"lower_c":37.0,"upper_c":38.2},
{"horizon_min":20,"mean_c":37.8,"lower_c":37.0,"upper_c":38.6}],"state":null}`.
All predictions must be finite with lower <= mean <= upper. State is limited to
16 KiB and resets between people/bouts. The replay worker exposes one request at
a time. Artifacts must be self-contained and include source, fitted parameters,
seed, dependencies and reproduction instructions.

Primary loss averages absolute error within each horizon, then horizons equally,
then people equally. Cutoffs are minutes 10 through 40, retaining all three
horizons only when measured targets exist. Secondary metrics: per-horizon MAE,
RMSE, bias; underprediction for reference >=38.5 C (a research slice, not a
diagnosis). IS90, coverage and width remain separate. Invalid or missing output
invalidates the attempt. The primary tier has no core-reference calibration.

CPU only: two hours, 30 development calls, up to 8 cores/32 GiB capped by host.
Use edit, command, evaluate, select and submit through the controller. Commands
and evaluations need hypothesis, expected_result, evidence_ids and code_diff.
Select final code from public evidence; final submission closes feedback.
Baseline: `python -m research_env baseline data/train/manifest.json artifact --task heat --kind elapsed`.
Other references: ridge, boosting, thermal (a simple saturating feature model).
Do not assume supplied baselines are optimal. The separate blind research judge
uses physiology-research-judge-v1; unobservable criteria are null. State findings
and limitations in experiment records. Do not submit private chain of thought.

This is a retrospective 24-person constant-load cycling dataset. It supports
ranking on this fixed task; it does not establish field or clinical performance.
