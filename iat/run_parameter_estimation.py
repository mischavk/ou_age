"""Run parameter estimation (DDM + OUM) for IAT data.

Trains BayesFlow inference networks for both DDM and OUM, runs diagnostics,
then estimates posterior parameters and PPC metrics for all participants
across all IAT data chunks. Saves CSV results and diagnostic figures.

Usage:
    python run_parameter_estimation.py
"""

import numpy as np
import os

os.environ["KERAS_BACKEND"] = "jax"

import gc
import bayesflow as bf
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from iat_functions import (
    iat_ddm_prior_fun, iat_oum_prior_fun,
    iat_likelihood,
    summarize_empirical_bins, compute_rmses_parallel_safe,
    safe_wasserstein,
)

FIGURES_DIR = "figures/"
os.makedirs(FIGURES_DIR, exist_ok=True)

DATA_DIR = "iat_data/"

# ── Training hyperparameters ──
NUM_TRAIN = 64000
NUM_VAL = 1000
EPOCHS = 100
BATCH_SIZE = 32
N_POSTERIOR_SAMPLES = 3000
N_MIN_VALID = 1000       # 1/3 of N_POSTERIOR_SAMPLES
N_PPC_SAMPLES = 100
N_PPC_SUBSAMPLE = 100   # persons per chunk for PPC (rest get NaN)

DATASETS = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".p")])


def run_ppc_person(emp, samples, param_keys, n_ppc_samples):
    """Run PPC for one person: compute RMSE (congruent) + WD (all 4 conditions)."""
    emp_sum = summarize_empirical_bins(emp)
    n_trials = emp.shape[0]

    sim_rts = np.empty((n_ppc_samples, n_trials))
    sim_cond = np.empty((n_ppc_samples, n_trials))

    for s in range(n_ppc_samples):
        kwargs = {}
        for k in param_keys:
            val = samples[k]
            kwargs[k] = np.squeeze(val[s])
        sims = iat_likelihood(**kwargs)["out"]
        sim_rts[s] = sims[:, 0]
        sim_cond[s] = sims[:, 2]

    # RMSE (congruent only)
    rmses = compute_rmses_parallel_safe(
        sim_rts, sim_cond,
        emp_sum["c_cong_med"], emp_sum["c_cong_qs"],
        emp_sum["e_cong_med"], emp_sum["e_cong_qs"],
    )

    # Wasserstein distance (all 4 conditions)
    wd_samples = np.empty((n_ppc_samples, 4))
    for s in range(n_ppc_samples):
        rt = sim_rts[s]
        cond = sim_cond[s]

        mask = (rt > 0) & (cond == 0)
        wd_samples[s, 0] = safe_wasserstein(emp_sum["c_cong_bin"], rt[mask])
        mask = (rt < 0) & (cond == 0)
        wd_samples[s, 1] = safe_wasserstein(emp_sum["e_cong_bin"], rt[mask])
        mask = (rt > 0) & (cond == 1)
        wd_samples[s, 2] = safe_wasserstein(emp_sum["c_inc_bin"], rt[mask])
        mask = (rt < 0) & (cond == 1)
        wd_samples[s, 3] = safe_wasserstein(emp_sum["e_inc_bin"], rt[mask])

    wd = np.nanmean(wd_samples, axis=0)
    return rmses, wd


def quality_check_oum(samples, person, n_min):
    """Set all parameters to NaN if any positive-constrained param has < n_min valid samples.

    Returns the fraction of posterior samples excluded due to impossible values
    (negative threshold or ndt) before person-level exclusion.
    """
    n_samples = samples["thresholds"].shape[1]

    # Count samples with any impossible value (negative threshold or ndt) before exclusion
    invalid_mask = (
        (samples["thresholds"][person, :, 0] < 0) |
        (samples["thresholds"][person, :, 1] < 0) |
        (samples["ndt_correct"][person, :].squeeze() < 0) |
        (samples["ndt_error"][person, :].squeeze() < 0)
    )
    frac_excluded = float(invalid_mask.sum()) / n_samples

    checks = [
        np.sum(samples["thresholds"][person, :, 0] > 0) < n_min,
        np.sum(samples["thresholds"][person, :, 1] > 0) < n_min,
        np.sum(samples["ndt_correct"][person, :] > 0) < n_min,
        np.sum(samples["ndt_error"][person, :] > 0) < n_min,
        np.sum(samples["k"][person, :] > 0) < n_min,
    ]
    if any(checks):
        for key in ["drifts", "thresholds", "ndt_correct", "ndt_error", "k"]:
            samples[key][person] = np.nan

    # Set negative values to NaN for positive-constrained parameters
    for key in ["drifts", "thresholds", "ndt_correct", "ndt_error", "k"]:
        neg_mask = samples[key][person] < 0
        samples[key][person][neg_mask] = np.nan

    return frac_excluded


def quality_check_ddm(samples, person, n_min):
    """Set all parameters to NaN if any positive-constrained param has < n_min valid samples.

    Returns the fraction of posterior samples excluded due to impossible values
    (negative threshold or ndt) before person-level exclusion.
    """
    n_samples = samples["thresholds"].shape[1]

    # Count samples with any impossible value before exclusion
    invalid_mask = (
        (samples["thresholds"][person, :, 0] < 0) |
        (samples["thresholds"][person, :, 1] < 0) |
        (samples["ndt_correct"][person, :].squeeze() < 0) |
        (samples["ndt_error"][person, :].squeeze() < 0)
    )
    frac_excluded = float(invalid_mask.sum()) / n_samples

    checks = [
        np.sum(samples["thresholds"][person, :, 0] > 0) < n_min,
        np.sum(samples["thresholds"][person, :, 1] > 0) < n_min,
        np.sum(samples["ndt_correct"][person, :] > 0) < n_min,
        np.sum(samples["ndt_error"][person, :] > 0) < n_min,
    ]
    if any(checks):
        for key in ["drifts", "thresholds", "ndt_correct", "ndt_error"]:
            samples[key][person] = np.nan

    for key in ["thresholds", "ndt_correct", "ndt_error"]:
        neg_mask = samples[key][person] < 0
        samples[key][person][neg_mask] = np.nan

    return frac_excluded


def estimate_model(model_name, prior_fn, param_keys, adapter, inference_variables,
                   quality_check_fn, figure_prefix):
    """Train one model (DDM or OUM), run diagnostics, estimate posteriors + PPC."""
    print("=" * 60)
    print(f"{model_name} — IAT")
    print("=" * 60)

    simulator = bf.make_simulator([prior_fn, iat_likelihood])

    print(f"Simulating training data ({NUM_TRAIN} samples)...")
    train_data = simulator.sample(NUM_TRAIN)
    val_data = simulator.sample(NUM_VAL)

    summary_net = bf.networks.SetTransformer(summary_dim=20)
    inference_net = bf.networks.CouplingFlow(transform="spline")

    workflow = bf.BasicWorkflow(
        simulator=simulator,
        adapter=adapter,
        inference_network=inference_net,
        summary_network=summary_net,
        inference_variables=inference_variables,
        initial_learning_rate=5e-5,
    )

    print(f"Training {model_name} inference network ({EPOCHS} epochs)...")
    history = workflow.fit_offline(
        train_data, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=val_data
    )

    # Diagnostics
    print(f"Generating {model_name} diagnostics...")
    figures = workflow.plot_default_diagnostics(
        test_data=500,
        loss_kwargs={"figsize": (15, 3), "label_fontsize": 12},
        recovery_kwargs={"figsize": (15, 6), "label_fontsize": 12},
        calibration_ecdf_kwargs={
            "figsize": (15, 6), "legend_fontsize": 8,
            "difference": True, "label_fontsize": 12,
        },
        z_score_contraction_kwargs={"figsize": (15, 6), "label_fontsize": 12},
    )
    for fig_name, fig in figures.items():
        path = f"{FIGURES_DIR}{figure_prefix}_{fig_name}.pdf"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {path}")

    # Per-chunk estimation + PPC
    print(f"\nEstimating {model_name} posteriors for {len(DATASETS)} data chunks...")
    result_rows = []

    for dataset_name in DATASETS:
        print(f"\n  Chunk: {dataset_name}")
        empirical_data = pd.read_pickle(os.path.join(DATA_DIR, dataset_name))
        emp_data = empirical_data["data_array"]
        id_array = empirical_data["outcome_array"][:, 0]
        age_array = empirical_data["outcome_array"][:, 1]

        # Sample posteriors in sub-chunks (memory)
        n_subchunks = 100
        looped = {}
        for counter, w in zip(range(n_subchunks), np.array_split(emp_data, n_subchunks)):
            looped[counter] = workflow.sample(
                conditions={"out": w}, num_samples=N_POSTERIOR_SAMPLES
            )

        # Concatenate sub-chunks and free memory
        samples = {}
        for key in looped[0].keys():
            samples[key] = np.concatenate([looped[c][key] for c in range(n_subchunks)])
        del looped
        gc.collect()

        n_persons = samples["drifts"].shape[0]
        rmses_all = np.full((n_persons, 10), np.nan)
        wd_all = np.full((n_persons, 4), np.nan)
        frac_excluded_all = np.zeros(n_persons)

        # Quality check all persons; track fraction of excluded samples per person
        for person in range(n_persons):
            frac_excluded_all[person] = quality_check_fn(samples, person, N_MIN_VALID)

        # PPC on random subsample only
        ppc_n = min(N_PPC_SUBSAMPLE, n_persons)
        ppc_indices = np.random.choice(n_persons, ppc_n, replace=False)
        print(f"    Running PPC on {ppc_n}/{n_persons} persons...")
        for i, person in enumerate(ppc_indices):
            if (i + 1) % 25 == 0 or i == ppc_n - 1:
                print(f"    PPC {i + 1}/{ppc_n}", end="\r")
            person_samples = {k: samples[k][person] for k in param_keys}
            rmses_all[person], wd_all[person] = run_ppc_person(
                emp_data[person], person_samples, param_keys, N_PPC_SAMPLES
            )

        print()

        # Build result DataFrame for this chunk
        df_chunk = pd.DataFrame({
            "v1": np.nanmedian(samples["drifts"][:, :, 0], axis=1),
            "v2": np.nanmedian(samples["drifts"][:, :, 1], axis=1),
            "a1": np.nanmedian(samples["thresholds"][:, :, 0], axis=1),
            "a2": np.nanmedian(samples["thresholds"][:, :, 1], axis=1),
            "ndt_correct": np.nanmedian(samples["ndt_correct"], axis=1).squeeze(),
            "ndt_error": np.nanmedian(samples["ndt_error"], axis=1).squeeze(),
        })

        if "k" in samples:
            df_chunk["k"] = np.nanmedian(samples["k"], axis=1).squeeze()

        df_chunk["age"] = age_array
        df_chunk["id"] = id_array
        df_chunk["frac_excluded_samples"] = frac_excluded_all

        # PPC metrics
        rmse_cols = [
            "rms_median_c_congruent", "rms_q1_c_congruent", "rms_q3_c_congruent",
            "rms_q7_c_congruent", "rms_q9_c_congruent",
            "rms_median_e_congruent", "rms_q1_e_congruent", "rms_q3_e_congruent",
            "rms_q7_e_congruent", "rms_q9_e_congruent",
        ]
        for i, col in enumerate(rmse_cols):
            df_chunk[col] = rmses_all[:, i]

        df_chunk["wd_c_congruent"] = wd_all[:, 0]
        df_chunk["wd_e_congruent"] = wd_all[:, 1]
        df_chunk["wd_c_incongruent"] = wd_all[:, 2]
        df_chunk["wd_e_incongruent"] = wd_all[:, 3]

        result_rows.append(df_chunk)
        print(f"    {dataset_name}: {n_persons} persons processed")

        del samples, rmses_all, wd_all, emp_data
        gc.collect()
        try:
            import jax
            jax.clear_caches()
        except Exception:
            pass

    # Save combined CSV
    final_df = pd.concat(result_rows, ignore_index=True)
    csv_path = os.path.join(DATA_DIR, "estimates", f"iat_results_{model_name.lower()}.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    final_df.to_csv(csv_path, index=False)
    print(f"\n  Saved {csv_path} ({len(final_df)} persons)")

    return workflow


# ═══════════════════════════════════════════════════════════
# OUM
# ═══════════════════════════════════════════════════════════

oum_adapter = (
    bf.Adapter()
    .as_set(["out"])
    .convert_dtype(from_dtype="float64", to_dtype="float32")
    .concatenate(["drifts", "thresholds", "ndt_correct", "ndt_error", "k"],
                 into="inference_variables")
    .concatenate(["out"], into="summary_variables")
)

oum_workflow = estimate_model(
    model_name="OUM",
    prior_fn=iat_oum_prior_fun,
    param_keys=["drifts", "thresholds", "ndt_correct", "ndt_error", "k"],
    adapter=oum_adapter,
    inference_variables=["drifts", "thresholds", "ndt_correct", "ndt_error", "k"],
    quality_check_fn=quality_check_oum,
    figure_prefix="figureC3_recovery_iat_oum",
)

# ═══════════════════════════════════════════════════════════
# DDM
# ═══════════════════════════════════════════════════════════

ddm_adapter = (
    bf.Adapter()
    .as_set(["out"])
    .convert_dtype(from_dtype="float64", to_dtype="float32")
    .concatenate(["drifts", "thresholds", "ndt_correct", "ndt_error"],
                 into="inference_variables")
    .concatenate(["out"], into="summary_variables")
)

ddm_workflow = estimate_model(
    model_name="DDM",
    prior_fn=iat_ddm_prior_fun,
    param_keys=["drifts", "thresholds", "ndt_correct", "ndt_error"],
    adapter=ddm_adapter,
    inference_variables=["drifts", "thresholds", "ndt_correct", "ndt_error"],
    quality_check_fn=quality_check_ddm,
    figure_prefix="figureC4_recovery_iat_ddm",
)

print("\n" + "=" * 60)
print("IAT parameter estimation complete.")
print("=" * 60)
