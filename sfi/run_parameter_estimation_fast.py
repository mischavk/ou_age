"""Run parameter estimation (DDM + OUM) for fast tasks.

Trains BayesFlow inference networks for both DDM and OUM, runs diagnostics,
then estimates parameters for all participants across 9 fast tasks.
Saves .npy estimate files and Appendix A3 diagnostics figures.

Usage:
    python run_parameter_estimation_fast.py
"""

import numpy as np
import os

os.environ["KERAS_BACKEND"] = "jax"

import keras
import bayesflow as bf
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sfi_functions import (
    sfi_ddm_fast_prior, sfi_oum_fast_prior,
    sfi_likelihood_ddm, sfi_likelihood_oum,
    summarize_empirical_data, compute_rmses
)

FIGURES_DIR = "figures/"
os.makedirs(FIGURES_DIR, exist_ok=True)

MODELS_DIR = "models/"      # trained networks are saved here for reuse

# Parameter labels for the diagnostic figures (order matches the adapter).
DDM_LABELS = [r"$v$ (drift)", r"$a$ (boundary)", r"$\tau$ (NDT)"]
OUM_LABELS = DDM_LABELS + [r"$k$ (self-excitation)"]

# ── Training hyperparameters ──
NUM_TRAIN = 64000
NUM_VAL = 1000
EPOCHS = 100  
N_SAMPLES = 3000       # posterior samples per participant
N_PPC_SAMPLES = 100    # PPC simulations per participant


def make_summary_net() -> "bf.networks.SetTransformer":
    """Large SetTransformer summary network.
    """
    return bf.networks.SetTransformer(
        embed_dims=(128, 128, 128, 128),
        num_heads=(8, 8, 8, 8),
        mlp_depths=(2, 2, 2, 2),
        mlp_widths=(128, 128, 128, 128),
        num_seeds=16,
        summary_dim=12,
    )

def make_inference_net() -> "bf.networks.CouplingFlow":
    """Affine CouplingFlow (default settings) — inference network used across
    all experiments (IAT + SFI fast/slow)."""
    return bf.networks.CouplingFlow()

FILES = sorted([
    os.path.join("sfi_data", f)
    for f in os.listdir("sfi_data")
    if f.endswith(".txt") and f.startswith("F")
])


def load_task_data(file: str) -> dict:
    """Load and clean RT data for one task.

    Returns a conditions dict ready for ``workflow.sample(conditions=...)``:
    ``rts`` with shape ``(n_persons, 100, 1)``.
    """

    df = pd.read_csv(file, sep=" ")
    df.loc[df["RT"] < 200, "RT"] = 0
    df.loc[df["RT"] > 10000, "RT"] = 0
    df["rt_mod"] = (df["RT"] * df["acc"].replace(0, -1)) / 1000
    df = df[df["block"] == "test"]
    df_last100 = df.groupby("pp", group_keys=False).tail(100)

    grouped = df_last100.groupby("pp")
    n_persons = grouped.ngroups

    rts = np.zeros((n_persons, 100, 1), dtype=np.float32)
    for i, (_, group) in enumerate(grouped):
        row = group["rt_mod"].values
        rts[i, : len(row), 0] = row

    return {"rts": rts}


def run_ppc(result: np.ndarray, real_samples: dict,
            likelihood_fn, param_keys: list) -> dict:
    """Run posterior predictive checks for one task."""
    n_persons = result.shape[0]
    rmses = np.empty((n_persons, 10))
    acc = np.empty(n_persons)   # |posterior-predictive accuracy - empirical accuracy|

    for person in range(n_persons):
        if (person + 1) % 20 == 0 or person == n_persons - 1:
            print(f"    PPC: person {person + 1}/{n_persons}", end="\r")
        emp = result[person, :, 0]
        emp_summary = summarize_empirical_data(emp)
        n_trials = emp.shape[0]

        sim_rts = np.empty((N_PPC_SAMPLES, n_trials))
        for s in range(N_PPC_SAMPLES):
            kwargs = {k: np.squeeze(real_samples[k][person, s]) for k in param_keys}
            kwargs["ndt"] = float(kwargs["ndt"])
            sim_rts[s] = likelihood_fn(**kwargs)["rts"]

        rmses[person] = compute_rmses(sim_rts, emp_summary["c_qs"], emp_summary["e_qs"])

        emp_acc = (emp > 0).sum() / max((emp != 0).sum(), 1)
        sim_acc = np.array([
            (sim_rts[s] > 0).sum() / max((sim_rts[s] != 0).sum(), 1)
            for s in range(N_PPC_SAMPLES)
        ])
        acc[person] = abs(sim_acc.mean() - emp_acc)

    print()  # newline after \r
    return {"rmse": rmses, "acc": acc}


# ═══════════════════════════════════════════════════════════
# DDM
# ═══════════════════════════════════════════════════════════

print("=" * 60)
print("DDM — Fast tasks")
print("=" * 60)

sfi_ddm_simulator = bf.make_simulator([sfi_ddm_fast_prior, sfi_likelihood_ddm])

sfi_ddm_adapter = (
    bf.Adapter()
    .as_set(["rts"])
    .constrain("v", lower=0.0)
    .constrain("a", lower=0.0, upper=8.0)
    .constrain("ndt", lower=0.1, upper=1.0)
    .convert_dtype(from_dtype="float64", to_dtype="float32")
    .concatenate(["v", "a", "ndt"], into="inference_variables")
    .rename("rts", "summary_variables")
)

print(f"Sampling validation data ({NUM_VAL} samples)...")
val_data = sfi_ddm_simulator.sample(NUM_VAL)

sfi_ddm_workflow = bf.BasicWorkflow(
    simulator=sfi_ddm_simulator,
    adapter=sfi_ddm_adapter,
    inference_network=make_inference_net(),
    summary_network=make_summary_net(),
    standardize="all",
    initial_learning_rate=5e-4,
)

os.makedirs(MODELS_DIR, exist_ok=True)
_ddm_path = f"{MODELS_DIR}sfi_fast_ddm.keras"
if os.path.exists(_ddm_path):
    print(f"Loading saved DDM network from {_ddm_path}...")
    sfi_ddm_workflow.approximator = keras.saving.load_model(_ddm_path)
else:
    print(f"Training DDM inference network offline ({NUM_TRAIN} sims, {EPOCHS} epochs)...")
    train_data = sfi_ddm_simulator.sample(NUM_TRAIN)
    sfi_ddm_workflow.fit_offline(
        train_data, epochs=EPOCHS, batch_size=32, validation_data=val_data
    )
    sfi_ddm_workflow.approximator.save(_ddm_path)
    print(f"  Saved {_ddm_path}")

# Diagnostics (Appendix A3)
print("Generating DDM diagnostics...")
figures = sfi_ddm_workflow.plot_default_diagnostics(
    test_data=500,
    variable_names=DDM_LABELS,
    loss_kwargs={"figsize": (15, 3), "label_fontsize": 12},
    recovery_kwargs={"figsize": (15, 6), "label_fontsize": 12},
    calibration_ecdf_kwargs={
        "figsize": (15, 6), "legend_fontsize": 8,
        "difference": True, "label_fontsize": 12,
    },
    z_score_contraction_kwargs={"figsize": (15, 6), "label_fontsize": 12},
)
for fig_name, fig in figures.items():
    path = f"{FIGURES_DIR}figureA3_recovery_fast_ddm_{fig_name}.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")

# Per-task estimation + PPC
print(f"\nEstimating DDM posteriors for {len(FILES)} fast tasks...")
for file in FILES:
    task_name = os.path.basename(file).replace(".txt", "")
    print(f"\n  Task: {task_name}")

    conditions = load_task_data(file)
    rts = conditions["rts"]
    print(f"    {rts.shape[0]} participants")

    real_samples = sfi_ddm_workflow.sample(
        conditions=conditions, num_samples=N_SAMPLES
    )
    np.save(file.replace(".txt", "_sfi_fast_ddm_estimates.npy"), real_samples)
    print(f"    Saved estimates ({N_SAMPLES} posterior samples)")

    ppc = run_ppc(rts, real_samples,
                  sfi_likelihood_ddm, ["v", "a", "ndt"])
    np.save(file.replace(".txt", "_sfi_fast_ddm_ppc.npy"), ppc)
    print(f"    Saved PPC metrics")


# ═══════════════════════════════════════════════════════════
# OUM
# ═══════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("OUM — Fast tasks")
print("=" * 60)

sfi_oum_simulator = bf.make_simulator([sfi_oum_fast_prior, sfi_likelihood_oum])

sfi_oum_adapter = (
    bf.Adapter()
    .as_set(["rts"])
    .constrain("v", lower=0.0)
    .constrain("a", lower=0.0, upper=8.0)
    .constrain("ndt", lower=0.1, upper=1.0)
    .constrain("k", lower=0.0)
    .convert_dtype(from_dtype="float64", to_dtype="float32")
    .concatenate(["v", "a", "ndt", "k"], into="inference_variables")
    .rename("rts", "summary_variables")
)

print(f"Sampling validation data ({NUM_VAL} samples)...")
val_data = sfi_oum_simulator.sample(NUM_VAL)

sfi_oum_workflow = bf.BasicWorkflow(
    simulator=sfi_oum_simulator,
    adapter=sfi_oum_adapter,
    inference_network=make_inference_net(),
    summary_network=make_summary_net(),
    standardize="all",
    initial_learning_rate=5e-4,
)

os.makedirs(MODELS_DIR, exist_ok=True)
_oum_path = f"{MODELS_DIR}sfi_fast_oum.keras"
if os.path.exists(_oum_path):
    print(f"Loading saved OUM network from {_oum_path}...")
    sfi_oum_workflow.approximator = keras.saving.load_model(_oum_path)
else:
    print(f"Training OUM inference network offline ({NUM_TRAIN} sims, {EPOCHS} epochs)...")
    train_data = sfi_oum_simulator.sample(NUM_TRAIN)
    sfi_oum_workflow.fit_offline(
        train_data, epochs=EPOCHS, batch_size=32, validation_data=val_data
    )
    sfi_oum_workflow.approximator.save(_oum_path)
    print(f"  Saved {_oum_path}")

# Diagnostics (Appendix A3)
print("Generating OUM diagnostics...")
figures = sfi_oum_workflow.plot_default_diagnostics(
    test_data=500,
    variable_names=OUM_LABELS,
    loss_kwargs={"figsize": (15, 3), "label_fontsize": 12},
    recovery_kwargs={"figsize": (15, 6), "label_fontsize": 12},
    calibration_ecdf_kwargs={
        "figsize": (15, 6), "legend_fontsize": 8,
        "difference": True, "label_fontsize": 12,
    },
    z_score_contraction_kwargs={"figsize": (15, 6), "label_fontsize": 12},
)
for fig_name, fig in figures.items():
    path = f"{FIGURES_DIR}figureA3_recovery_fast_oum_{fig_name}.pdf"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")

# Per-task estimation + PPC
print(f"\nEstimating OUM posteriors for {len(FILES)} fast tasks...")
for file in FILES:
    task_name = os.path.basename(file).replace(".txt", "")
    print(f"\n  Task: {task_name}")

    conditions = load_task_data(file)
    rts = conditions["rts"]
    print(f"    {rts.shape[0]} participants")

    real_samples = sfi_oum_workflow.sample(
        conditions=conditions, num_samples=N_SAMPLES
    )
    np.save(file.replace(".txt", "_sfi_fast_oum_estimates.npy"), real_samples)
    print(f"    Saved estimates ({N_SAMPLES} posterior samples)")

    ppc = run_ppc(rts, real_samples,
                  sfi_likelihood_oum, ["v", "a", "ndt", "k"])
    np.save(file.replace(".txt", "_sfi_fast_oum_ppc.npy"), ppc)
    print(f"    Saved PPC metrics")

print("\n" + "=" * 60)
print("Fast task parameter estimation complete.")
print("=" * 60)
