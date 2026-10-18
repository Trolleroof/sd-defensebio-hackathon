# Load-carriage metabolic cost — load-v1

Build an executable predictor of gross metabolic power in watts for level walking
at 4 km/h with 0, 5, 10 or 15 kg carried load, generalizing to unseen people.
Choose features, model, fitting, intervals and experiments. Train using
`data/train/manifest.json`; evaluate using the controller and select your final
artifact from development evidence. Artifact entrypoint:
`predict(artifact_dir, request)` in `artifact/inference.py`.

Request fields: age_years, height_cm, mass_kg, load_kg, speed_m_s (=4/3.6),
grade (=0), state. No RMR, body-composition lab values, other condition targets
or unloaded reference calibration is provided in this primary tier.
Return exactly `{"mean_w":300,"lower_w":250,"upper_w":350,"state":null}`.
Values must be finite and ordered lower <= mean <= upper; state <=16 KiB.
State resets between people. All four loads count; missing predictions invalidate
the attempt. Three source people lack a measured 15 kg target; those three
conditions are excluded before comparison, preserving all 48 people and 189
measured targets. Average each person's available loads, then people equally.
Artifact includes source, fitted parameters, dependency lock, seeds
and reproduction instructions. Only artifact/current request enter replay.

Primary: absolute watt error averaged across loads within each person, then
people equally. Secondary: W/kg error, RMSE, bias, each load's error, worst-load
error, pairwise measured-order agreement. Costs are not forced to be monotonic.
Intervals: IS90, coverage and width reported separately; broad intervals are not
rewarded merely for coverage. Reference rates are kcal/day equivalents converted
as watts = kcal/day *4184/86400. No RMR subtraction; these are gross rates.

CPU only: two hours, 30 development evaluations, up to 8 cores/32 GiB capped by
host. Use edit, command, evaluate, select and submit. Commands/evaluations record
hypothesis, expected_result, evidence_ids and code_diff. Final submission closes
feedback. Baseline: `python -m research_env baseline data/train/manifest.json artifact --task load --kind mean_by_load`.
Other references: ridge with mass/load interactions; compact forest. A published
load-cost equation is not included because its target/input applicability has
not been verified for this release. Do not invent its coefficients.
Blind process judgment uses physiology-research-judge-v1 and stays separate from
prediction scores. Record substantive evidence and claims, not chain of thought.

Source is 48 people with four measured conditions each, not 192 independent people.
This leaderboard ranks retrospective lab prediction at one speed/grade; it does
not measure terrain adaptation, raw wearable inference or clinical outcomes.
