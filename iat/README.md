# Study 2 — Race IAT (Project Implicit)

This folder contains all code for Study 2 of the manuscript. We fit a DDM and OUM to Race IAT data from Project Implicit (*N* ≈ 5.6 million participants, years 2003–2023) and examine cross-sectional age trends in cognitive parameters.

## Pipeline

```bash
bash run_all_training.sh
```

| Step | Script | Output |
|------|--------|--------|
| 1 | `run_model_comparison.py` | `figure4_model_comparison_iat.pdf`, calibration + confusion figures (Appendix C1/C2) |
| 2 | `run_parameter_estimation.py` | `iat_results_{ddm,oum}.csv`, recovery figures (Appendix C3/C4) |
| 3 | `run_analyses.py` | `figure5_age_trends_iat.pdf`, `figure6_k_subgroups_iat.pdf`, age correlation table |
| 4 | `run_ppc_analysis.py` | `figureC5_ppc_comparison_iat.pdf` |

## Training settings

### Model comparison
- Online training: 1,000 batches × 32 samples/batch = 32,000 samples per epoch
- Epochs: 100
- Optimizer: AdamW with cosine decay (1e-4 → 1e-5)
- Summary network: SetTransformer (20 dimensions)
- Classifier: MLP (16 × 256 units, ReLU)
- Validation: 10,000 simulated data sets

### Parameter estimation
- Training set: 64,000 offline simulations + 1,000 validation
- Epochs: 100
- Summary network: SetTransformer (20 dimensions)
- Inference network: CouplingFlow (spline transform)
- Initial learning rate: 5e-5
- Posterior samples per participant: 3,000
- Recovery diagnostics: 500 simulated data sets

## Data

See `iat_data/README_data.md` for data availability and preprocessing details.

The IAT data files (~6.5 GB) are not included in this repository. Results are saved in:
- `iat_data/estimates/iat_model_comparison_results.csv` — posterior model probabilities per person
- `iat_data/estimates/iat_demographics.csv` — age, gender, education, country
- `iat_data/estimates/iat_results_oum.csv` — OUM posterior medians per person (large, not tracked)
- `iat_data/estimates/iat_results_ddm.csv` — DDM posterior medians per person (large, not tracked)

## Priors

Priors are defined in `iat_functions.py`. The IAT uses the same priors as fast tasks in Study 1, plus a separate error non-decision time.

| Parameter | DDM | OUM |
|-----------|-----|-----|
| *ν*₁, *ν*₂ (drift, congruent/incongruent) | Γ(3.5, 1.0) | Γ(3.5, 1.0) |
| *a*₁, *a*₂ (boundary, congruent/incongruent) | 8·σ(𝒩(−1.0, 1.0)) | 8·σ(𝒩(0.0, 1.0)) |
| *τ*_c (correct NDT) | U(0.1, 1.0) | U(0.1, 1.0) |
| *τ*_e (error NDT) | Γ(2.0, 0.3) | Γ(2.0, 0.3) |
| *k* (self-excitation) | — | Γ(4.0, 0.5) + 1 |

The error NDT prior Γ(2.0, 0.3) accounts for the additional response-correction time recorded for errors in Project Implicit.
