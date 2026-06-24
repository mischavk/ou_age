"""Shared helpers for IAT parameter-recovery diagnostic figures.

Produces per-model diagnostic figures (recovery, calibration ECDF, coverage,
z-score contraction) with the parameters laid out across two rows and labelled
with their model symbols. 
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import bayesflow as bf

# Parameter labels in the order the adapter concatenates them into
# ``inference_variables`` (drifts[2], thresholds[2], ndt_correct, ndt_error[, k]).
DDM_LABELS = [
    r"$v_1$ (drift, cong.)", r"$v_2$ (drift, inc.)",
    r"$a_1$ (boundary, cong.)", r"$a_2$ (boundary, inc.)",
    r"$\tau_c$ (NDT, correct)", r"$\tau_e$ (NDT, error)",
]
OUM_LABELS = DDM_LABELS + [r"$k$ (self-excitation)"]

# Each per-parameter diagnostic and any extra kwargs.
_DIAGNOSTICS = [
    ("recovery", bf.diagnostics.plots.recovery, {}),
    ("calibration_ecdf", bf.diagnostics.plots.calibration_ecdf, {"difference": True}),
    ("coverage", bf.diagnostics.plots.coverage, {}),
    ("z_score_contraction", bf.diagnostics.plots.z_score_contraction, {}),
]


def sample_estimates(approximator, simulator, cond_keys, param_keys,
                     n_test=500, n_samples=1000, chunk=250):
    """Simulate test data and draw posterior samples (in chunks, to bound memory).

    Returns ``(estimates, targets)`` dicts keyed by ``param_keys``, suitable for
    the BayesFlow diagnostic plot functions.
    """
    test = simulator.sample(n_test)
    parts = []
    for start in range(0, n_test, chunk):
        end = min(start + chunk, n_test)
        cond = {k: test[k][start:end] for k in cond_keys}
        parts.append(approximator.sample(num_samples=n_samples, conditions=cond))
    estimates = {k: np.concatenate([p[k] for p in parts], axis=0) for k in param_keys}
    targets = {k: test[k] for k in param_keys}
    return estimates, targets


def save_param_diagnostics(estimates, targets, variable_names, prefix,
                           figures_dir="figures/", num_row=2):
    """Save the four per-parameter diagnostic figures, parameters laid out over
    ``num_row`` rows and labelled with ``variable_names``."""
    n = len(variable_names)
    num_col = -(-n // num_row)  # ceil
    for name, fn, extra in _DIAGNOSTICS:
        fig = fn(
            estimates=estimates, targets=targets,
            variable_names=variable_names,
            num_row=num_row, figsize=(4.3 * num_col, 4.0 * num_row),
            label_fontsize=13, title_fontsize=14, tick_fontsize=11, **extra,
        )
        path = f"{figures_dir}{prefix}_{name}.pdf"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {path}")
