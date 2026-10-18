# Heat data and provenance

Source: D'Lugos (2025), Figshare 29120024 v1, CC BY 4.0,
https://doi.org/10.6084/m9.figshare.29120024.v1.
Original workbook MD5: 01214bfc835215fe1a5d3f5da1e9ad52.
Person split: 14 train, 5 development, 5 private test; no windows cross people.
The private audit retains source cells and quality flags. Public training
records include request/targets; public development includes requests only.
Target: Core Temp / Rectal. Skin input: Skin Temp / YSI TSK.
HR input: Exercise Data / HR, interval labels aligned to their right edge.
HR outside [30,240] is omitted, including Exercise Data!AK9=1176.
Environmental observations are carried forward only after observation.
No backward interpolation. Demographic features exclude laboratory VO2peak.
Full workbook, private development labels and all test people stay outside the
participant mount. Known source identities in training are grouping metadata,
not predictor features. The dataset's conditions are not varied terrain/field heat.
