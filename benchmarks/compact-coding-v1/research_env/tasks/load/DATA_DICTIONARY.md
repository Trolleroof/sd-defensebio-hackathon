# Load data and provenance

Mateos (2020), EVOBREATH 2015, Mendeley rhmdzmx6kk v1, CC BY 4.0,
https://doi.org/10.17632/rhmdzmx6kk.1.
Workbook SHA-256: 51b42232fd63e15f2566f1d1053c92fe2de88477c3c4f14ce95fdc75784c58da.
DATA sheet: Age (years), Height (cm), Weight (kg) and assigned carried load.
Numeric strings are parsed explicitly; originals remain in private lineage.
Reference columns EE_WW00/05/10/15 are gross kcal/day equivalent rates, as
specified in Variables. Convert by 4184/86400 to watts. Speed 4/3.6 m/s; grade 0.
No RMR subtraction or inference-time reference features. Sex is omitted in v1.
Person grouping: 28 train /10 development /10 test; all four loads stay together.
Public development has no targets; private labels/original workbooks are never
mounted in participant workers. No reference-calibrated or held-out-load tier is
mixed into the primary leaderboard. Workbook has 48 people, no excluded people.
Three source '-' targets (BE0002, BE0058, BE0063 at 15 kg) are excluded before
evaluation. All 48 people remain, with 189 measured condition targets; these
exclusions are frozen for every competitor. Missing targets are not imputed.
