# Study 1 — 18 Binary Decision Tasks

This folder contains all code and data for Study 1 of the manuscript. We fit a DDM and OUM to data from 18 reaction-time tasks (9 fast, mean RT ≈ 800 ms; 9 slow, mean RT ≈ 3 s) collected from *N* = 125 participants and examine how cognitive parameters correlate with age.

## Pipeline

Run all steps in order:

```bash
bash run_all_training.sh
```

Or run individual steps manually:

| Step | Script | Output |
|------|--------|--------|
| 1 | `run_model_comparison_fast.py` | `figure2_model_comparison_fast.pdf`, calibration + confusion figures (Appendix A1/A2) |
| 2 | `run_model_comparison_slow.py` | `figure2_model_comparison_slow.pdf`, calibration + confusion figures (Appendix A1/A2) |
| 3 | `run_parameter_estimation_fast.py` | Posterior `.npy` files, recovery figures (Appendix A3) |
| 4 | `run_parameter_estimation_slow.py` | Posterior `.npy` files, recovery figures (Appendix A4) |
| 5 | `run_analyses.py` | `figure3a_age_correlations_fast.pdf`, `figure3b_age_correlations_slow.pdf`, age correlation table |
| 6 | `run_ppc_analysis.py` | `figureA5_ppc_comparison.pdf`, PPC heatmaps |
| 7 | `run_figure1_schematic.py` | `figure1_schematic.pdf` |

## Training settings

### Model comparison (both fast and slow)
- Online training: 1,000 batches × 32 samples/batch = 32,000 samples per epoch
- Epochs: 100
- Optimizer: AdamW with cosine decay (1e-4 → 1e-5)
- Summary network: SetTransformer (20 dimensions)
- Classifier: MLP (16 × 256 units, ReLU)
- Validation: 10,000 separately simulated data sets

### Parameter estimation (both fast and slow)
- Training set: 64,000 offline simulations + 1,000 validation
- Epochs: 100
- Summary network: SetTransformer (20 dimensions)
- Inference network: CouplingFlow (spline transform)
- Initial learning rate: 5e-5
- Posterior samples per participant: 3,000
- Recovery diagnostics: 500 simulated data sets

## Data

- `sfi_data/FF1.txt` … `sfi_data/FV3.txt` — 9 fast task files (raw RTs)
- `sfi_data/SF1.txt` … `sfi_data/SV3.txt` — 9 slow task files (raw RTs)
- `sfi_data/estimates/questionnaires.csv` — participant demographics (age, gender, etc.)

Each `.txt` file has columns `pp` (participant ID), `block`, `acc` (accuracy), `RT` (ms). Posterior estimates are saved as `.npy` files alongside the raw data files after running the pipeline.

## Priors

All priors are defined in `sfi_functions.py`. Boundary separation uses a sigmoid-transformed normal:
*a* = *L* · σ(𝒩(*μ*, *σ*²)), bounded on (0, *L*).

| Parameter | Fast DDM | Fast OUM | Slow DDM | Slow OUM |
|-----------|----------|----------|----------|----------|
| *ν* | Γ(3.5, 1.0) | Γ(3.5, 1.0) | Γ(6.0, 0.25) | Γ(6.0, 0.25) |
| *a* | 8·σ(𝒩(−1.0, 1.0)) | 8·σ(𝒩(0.0, 1.0)) | 10·σ(𝒩(−1.3, 1.0)) | 10·σ(𝒩(1.3, 2.0)) |
| *τ* | U(0.1, 1.0) | U(0.1, 1.0) | U(0.1, 3.0) | U(0.1, 3.0) |
| *k* | — | Γ(4.0, 0.5) + 1 | — | Γ(4.0, 0.5) + 1 |
