# IAT Data

The raw and preprocessed IAT data files are not included in this repository due to their size (~6.5 GB).

## Source

All data come from the Project Implicit OSF repository:
- URL: https://osf.io/52qxl
- Years used: 2003–2023
- Task: Race IAT

## Preprocessing

The raw data were preprocessed into 329 chunks of ~15,000–30,000 participants each (pickle files named `prepared_*.p`). Each chunk contains a `data_array` of shape `(N_persons, 120, 3)` where the three columns are:
- `[:, :, 0]` — signed RT in seconds (positive = correct, negative = error)
- `[:, :, 1]` — absolute RT in seconds
- `[:, :, 2]` — condition (0 = congruent, 1 = incongruent)

Trials were excluded if RT < 200 ms or > 10 s.

To reproduce the preprocessing, see the original analysis notebooks in `iat/legacy/iat_analyes.ipynb`.

## Included files

The following summary files are included in this repository:
- `estimates/iat_model_comparison_results.csv` — posterior model probabilities (DDM / OUM) per person
- `estimates/iat_demographics.csv` — age, gender, education, country per person
