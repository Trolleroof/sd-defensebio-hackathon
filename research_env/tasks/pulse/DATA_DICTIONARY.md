# Data and causal window contract

Original: GalaxyPPG Zenodo record 14635823, version 1.0.0, CC BY 4.0. Retain the original archive, metadata, license text, Zenodo MD5 and locally calculated SHA-256. Attribution: Sangjun Park, Dejiang Zheng and Uichin Lee, *GalaxyPPG: A PPG Signal Dataset Collected in Semi-Naturalistic Settings Using Galaxy Watch*. Original author code documents Polar phone-time correction of minus 32,400,000 milliseconds. This importer uses raw recordings rather than published window labels.

| Request field | Meaning |
| --- | --- |
| timestamps_s | 201 timestamps from -8 to 0 inclusive at 25 Hz, relative to cutoff |
| ppg | Galaxy optical pulse values in source device units; 201 samples |
| acceleration | 201 triples [x,y,z], source Galaxy acceleration in m/s² |
| sample_rates_hz | ppg=25, acceleration=25 after within-window resampling |
| cutoff_s | 0; absolute timestamps and person IDs are not supplied to predictor |
| state | Previous JSON prediction state, null at bout/session start; max 16 KiB |

Output is `{mean_bpm, lower_bpm, upper_bpm, state}`. Means and interval endpoints are bpm. Request IDs are opaque controller associations, not predictor features. No reference channel appears in the request.

Source sensor timestamps are milliseconds since epoch; Galaxy timestamps divide by 1000. Polar ECG uses (phoneTimestamp-32400000)/1000 seconds. Sort stably and retain the first sample at each duplicate timestamp; log removed duplicates. Event ENTER/EXIT times establish bouts. Transition gaps are excluded rather than assigned guessed activities. First eligible update is ENTER+8 s, then every 2 s while cutoff<=EXIT. Raw support samples satisfy cutoff-8<=timestamp<=cutoff; endpoints are inclusive. Interpolate only from this support onto 201 equally spaced points, with constant extrapolation of at most 0.1 s for endpoint jitter. No sample outside the closed window supports interpolation. Sensor clock alignment follows author documentation, with sample ranges/rates audited; this does not measure physical inter-device synchronization error.

Eligibility is decided before comparing any candidate: finite PPG/ACC/ECG, boundary coverage <=0.1 s, PPG/ACC gaps <=0.2 s, ECG gaps <=0.05 s, nonflat PPG. ECG rate must be 110–150 Hz. Within each window, a third-order 5–25 Hz Butterworth filter and forward/backward filtering support R-peak detection. Filtering never reaches beyond the reference window. Orient using the larger absolute 1st/99th percentile; peak distance >=0.28 s; minimum height 0.4 times the oriented 99th percentile. Prominence=max(0.5*SD,0.15*(p99-p01)); compare detectors with 1.0 and 1.5 times this prominence. Require >=4 peaks, all RR in [0.28,1.5] s, RR CV<=0.25 and detector estimates within 5 bpm. Label is 60/mean(RR), with both peaks of each RR within the window. Synthetic known-rate signals validate arithmetic only. The detector is provisional and requires broader human-reviewed beat annotations before benchmark release.

`private/audit.json` records exclusions by reason, person counts, source-member hashes, timestamp ranges/rates and example peak timestamps. `private/splits.json` freezes all sessions/devices of each person together. P01 lacks Galaxy/event files; the default audited 23-person split is 13/5/5. Its deterministic hash order uses `galaxy-pulse-split-v1` before model comparison. An explicit split JSON must exactly cover the audited people. Existing DaLiA exposures remain documented and unchanged; no DaLiA waveform or synthetic feature sequence is substituted.

Signal NPZ files contain only `ppg` [time,value] and `acceleration` [time,x,y,z]. Training manifests include reference_bpm and evaluator slice labels; public development manifests remove those fields. Full private archives/ECG, private development labels and test recordings stay outside the worker. NPZ loading always disables pickle. Do not mount the project root or prepared dataset root into a participant container: use only `public/` exports. The private scorer creates fresh replay workers containing only the frozen artifact and inference entrypoint.
