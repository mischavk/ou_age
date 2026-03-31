# Age Differences in the Ornstein-Uhlenbeck Model of Decision Making

Code and data for the manuscript:

> von Krause, M., Wang, J.-S., Donkin, C., & Radev, S. T. (in prep.). *An investigation of age differences in the Ornstein-Uhlenbeck model of decision making.*

All analyses use simulation-based inference (SBI) via [BayesFlow 2.0.10](https://bayesflow.org). The Ornstein-Uhlenbeck model (OUM) extends the standard diffusion decision model (DDM) with a self-excitation parameter *k* that captures the degree to which accumulated evidence accelerates further accumulation. We compare the DDM and OUM across two data sets and examine how *k* and other cognitive parameters relate to age.

---

## Repository Structure

```
.
├── sfi/                  # Study 1 — 18 binary decision tasks (N = 125)
│   ├── sfi_functions.py          # Model definitions, priors, simulators, utilities
│   ├── run_model_comparison_fast.py
│   ├── run_model_comparison_slow.py
│   ├── run_parameter_estimation_fast.py
│   ├── run_parameter_estimation_slow.py
│   ├── run_analyses.py
│   ├── run_ppc_analysis.py
│   ├── run_figure1_schematic.py
│   ├── run_all_training.sh       # Runs all 7 steps sequentially
│   ├── sfi_data/                 # Raw task data (.txt) and posterior estimates (.npy)
│   ├── figures/                  # Generated figures (PDF)
│   ├── prior_exploration/        # Prior selection scripts and documentation
│   └── legacy/                   # Original Jupyter notebooks (superseded by scripts)
│
├── iat/                  # Study 2 — Race IAT (N ≈ 5.6 million)
│   ├── iat_functions.py          # Model definitions, priors, simulators, utilities
│   ├── run_model_comparison.py
│   ├── run_parameter_estimation.py
│   ├── run_analyses.py
│   ├── run_ppc_analysis.py
│   ├── run_all_training.sh       # Runs all 4 steps sequentially
│   ├── iat_data/                 # Preprocessed IAT data chunks (.p) and results (.csv)
│   ├── figures/                  # Generated figures (PDF)
│   └── legacy/                   # Original Jupyter notebooks (superseded by scripts)
│
├── overleaf/             # Manuscript source (LaTeX, figures, tables)
│   ├── manuscript.tex
│   ├── references.bib
│   ├── figures/                  # All publication-ready figures
│   └── table*.tex                # Auto-generated tables
│
└── legacy/               # Root-level exploratory files not used in final analyses
```

---

## Requirements

### Conda environment (recommended)

```bash
conda env create -f environment.yml
conda activate bfdev
```

### Key dependencies

| Package | Version |
|---------|---------|
| Python | 3.11 |
| bayesflow | 2.0.10 |
| jax / jaxlib | ≥ 0.4 (GPU build recommended) |
| keras | ≥ 3.0 |
| numpy | ≥ 1.26 |
| pandas | ≥ 2.0 |
| scipy | ≥ 1.11 |
| matplotlib | ≥ 3.8 |
| seaborn | ≥ 0.13 |
| numba | ≥ 0.59 |

All neural networks use the JAX backend (`KERAS_BACKEND=jax`). A CUDA-capable GPU is strongly recommended (tested on RTX 4080 Super).

---

## Reproducing the Analyses

### Study 1 (SFI — 18 tasks)

```bash
cd sfi
bash run_all_training.sh
```

This runs all 7 steps sequentially (~8–12 h on an RTX 4080 Super):
1. Model comparison — fast tasks
2. Model comparison — slow tasks
3. Parameter estimation — fast tasks
4. Parameter estimation — slow tasks
5. Age correlation analyses
6. Posterior predictive checks
7. Figure 1 (DDM vs. OUM schematic)

Raw task data is in `sfi/sfi_data/*.txt`. Posterior estimates are saved as `.npy` files in the same folder. Figures are saved to `sfi/figures/`.

### Study 2 (IAT — 5.6 million participants)

```bash
cd iat
bash run_all_training.sh
```

This runs all 4 steps sequentially (~6–10 h on an RTX 4080 Super):
1. Model comparison
2. Parameter estimation (all 329 data chunks processed serially)
3. Age trend analyses and Figure 5/6
4. Posterior predictive checks

The preprocessed IAT data chunks are in `iat/iat_data/` (not included in this repository due to file size; see Data Availability below). Results are saved as CSV files in `iat/iat_data/estimates/` and figures to `iat/figures/`.

---

## Data Availability

**Study 1:** Raw response time data for the 18 tasks are in `sfi/sfi_data/*.txt`. These data were originally collected by [Lerche et al. (2020)](https://doi.org/10.1037/xge0000780) and are included here with permission.

**Study 2:** The Race IAT data are publicly available from the [Project Implicit OSF repository](https://osf.io/52qxl). The preprocessed pickle files used in this analysis are not included in this repository due to their size (~6.5 GB). See `iat/iat_data/README_data.md` for instructions on how to reproduce the preprocessing step.

---

## Models

Both studies compare two evidence accumulation models:

**DDM:** `dX = ν dt + σ dW`

**OUM:** `dX = (ν + k·X) dt + σ dW`

When *k* > 0, self-excitation accelerates accumulation as evidence builds up. Both models use the same three core parameters (drift rate *ν*, boundary separation *a*, non-decision time *τ*); the OUM adds *k*.

Simulations use Euler-Maruyama integration (implemented in Numba for speed). All priors are documented in `sfi/sfi_functions.py`, `iat/iat_functions.py`, and Table 2 of the manuscript.

---

## Citation

If you use this code, please cite the manuscript and BayesFlow:

```
von Krause, M., Wang, J.-S., Donkin, C., & Radev, S. T. (in prep.).
An investigation of age differences in the Ornstein-Uhlenbeck model of decision making.
```

```
Radev, S. T., et al. (2023). BayesFlow: Amortized Bayesian workflows with neural networks.
arXiv preprint arXiv:2306.16015.
```
