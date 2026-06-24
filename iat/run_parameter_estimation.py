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
import zlib

import keras
import bayesflow as bf
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from iat_functions import (
    iat_ddm_prior_fun, iat_oum_prior_fun,
    iat_likelihood,
    summarize_empirical_safe, compute_rmses_parallel_safe,
)
from diagnostics_utils import (
    sample_estimates, save_param_diagnostics, DDM_LABELS, OUM_LABELS,
)

FIGURES_DIR = "figures/"
os.makedirs(FIGURES_DIR, exist_ok=True)

DATA_DIR = "iat_data/"
MODELS_DIR = "models/"      # trained networks are saved here for reuse
N_DIAG_TEST = 500           # simulated data sets for the diagnostic figures

# ── Training hyperparameters ──
# Networks/training (shared with the SFI pipeline): a Large SetTransformer
# summary (summary_dim=20 for the IAT's larger parameter set), an affine
# CouplingFlow inference network, standardize="all", and offline training on a
# fixed 64k simulation set at lr=5e-4.

NUM_TRAIN = 64000
NUM_VAL = 1000
EPOCHS = 100  
BATCH_SIZE = 32
N_POSTERIOR_SAMPLES = 3000
N_MIN_VALID = 1000       # 1/3 of N_POSTERIOR_SAMPLES
N_PPC_SAMPLES = 100
N_PPC_SUBSAMPLE = 100   # persons per chunk for PPC 

# RT-quantile RMSE columns, in the order returned by compute_rmses_parallel_safe:
# congruent (correct then error) followed by incongruent (correct then error).
RMSE_COLS = [
    "rms_median_c_congruent", "rms_q1_c_congruent", "rms_q3_c_congruent",
    "rms_q7_c_congruent", "rms_q9_c_congruent",
    "rms_median_e_congruent", "rms_q1_e_congruent", "rms_q3_e_congruent",
    "rms_q7_e_congruent", "rms_q9_e_congruent",
    "rms_median_c_incongruent", "rms_q1_c_incongruent", "rms_q3_c_incongruent",
    "rms_q7_c_incongruent", "rms_q9_c_incongruent",
    "rms_median_e_incongruent", "rms_q1_e_incongruent", "rms_q3_e_incongruent",
    "rms_q7_e_incongruent", "rms_q9_e_incongruent",
]

def make_summary_net() -> "bf.networks.SetTransformer":
    """Large SetTransformer with summary_dim=20"""

    return bf.networks.SetTransformer(
        embed_dims=(128, 128, 128, 128),
        num_heads=(8, 8, 8, 8),
        mlp_depths=(2, 2, 2, 2),
        mlp_widths=(128, 128, 128, 128),
        num_seeds=16,
        summary_dim=20,
    )


def make_inference_net() -> "bf.networks.CouplingFlow":
    """Affine CouplingFlow (default settings)"""
    return bf.networks.CouplingFlow()


DATASETS = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".p")])


def run_ppc_person(emp, samples, param_keys, n_ppc_samples):
    """Run PPC for one person: RT-quantile RMSE (correct + error) for both the
    congruent and incongruent conditions, plus accuracy discrepancy per condition."""
    emp_sum = summarize_empirical_safe(emp)
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

    # RT-quantile RMSE — congruent (0-9) and incongruent (10-19), correct + error
    rmses = compute_rmses_parallel_safe(
        sim_rts, sim_cond,
        emp_sum["c_cong_med"], emp_sum["c_cong_qs"],
        emp_sum["e_cong_med"], emp_sum["e_cong_qs"],
        emp_sum["c_inc_med"], emp_sum["c_inc_qs"],
        emp_sum["e_inc_med"], emp_sum["e_inc_qs"],
    )

    # Accuracy per condition (congruent=0, incongruent=1):
    emp_rt, emp_cond = emp[:, 0], emp[:, 2]
    acc = np.empty(2)
    for ci, c in enumerate((0, 1)):
        ev = (emp_rt != 0) & (emp_cond == c)
        emp_acc = (emp_rt[ev] > 0).mean() if ev.any() else np.nan
        sim_acc = np.empty(n_ppc_samples)
        for s in range(n_ppc_samples):
            sv = (sim_rts[s] != 0) & (sim_cond[s] == c)
            sim_acc[s] = (sim_rts[s][sv] > 0).mean() if sv.any() else np.nan
        acc[ci] = abs(np.nanmean(sim_acc) - emp_acc)

    return rmses, acc


# A small number of out-of-distribution participants yield implausibly large or
# non-finite posterior medians in the unbounded (lower-bound-only) parameters.
# These participants are excluded so they do not distort aggregate statistics.

SANE_MAX = 100.0


def _implausible_person(samples, person, keys):
    """True if the per-dimension posterior median of any listed (unbounded) param
    is non-finite or exceeds SANE_MAX for this person."""
    for key in keys:
        m = np.nanmedian(samples[key][person], axis=0)
        if not np.all(np.isfinite(m)) or np.nanmax(np.abs(m)) > SANE_MAX:
            return True
    return False


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

    # Exclude participants with implausibly large or non-finite estimates.
    if _implausible_person(samples, person, ["drifts", "ndt_error", "k"]):
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

    # Exclude participants with implausibly large or non-finite estimates.
    if _implausible_person(samples, person, ["drifts", "ndt_error"]):
        for key in ["drifts", "thresholds", "ndt_correct", "ndt_error"]:
            samples[key][person] = np.nan

    for key in ["thresholds", "ndt_correct", "ndt_error"]:
        neg_mask = samples[key][person] < 0
        samples[key][person][neg_mask] = np.nan

    return frac_excluded


def estimate_model(model_name, prior_fn, param_keys, adapter,
                   quality_check_fn, figure_prefix, variable_names):
    """Train one model (DDM or OUM), save it, run diagnostics, estimate
    posteriors + PPC."""
    print("=" * 60)
    print(f"{model_name} — IAT")
    print("=" * 60)

    simulator = bf.make_simulator([prior_fn, iat_likelihood])

    workflow = bf.BasicWorkflow(
        simulator=simulator,
        adapter=adapter,
        inference_network=make_inference_net(),
        summary_network=make_summary_net(),
        standardize="all",
        initial_learning_rate=5e-4,
    )

    # Reuse the saved network if one exists, so the estimates, diagnostics, and
    # PPC are all based on the same trained network; otherwise train online and
    # save it for future runs.
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = f"{MODELS_DIR}iat_{model_name.lower()}.keras"
    history = None
    if os.path.exists(model_path):
        print(f"Loading saved {model_name} network from {model_path}...")
        workflow.approximator = keras.saving.load_model(model_path)
    else:
        print(f"Training {model_name} inference network offline "
              f"({NUM_TRAIN} sims, {EPOCHS} epochs)...")
        train_data = simulator.sample(NUM_TRAIN)
        history = workflow.fit_offline(
            train_data, epochs=EPOCHS, batch_size=BATCH_SIZE,
            validation_data=simulator.sample(NUM_VAL),
        )
        workflow.approximator.save(model_path)
        print(f"  Saved {model_path}")

    # Diagnostics: loss curve (only when freshly trained) + per-parameter figures.
    print(f"Generating {model_name} diagnostics...")
    if history is not None:
        f = bf.diagnostics.plots.loss(history=history, figsize=(15, 3))
        f.savefig(f"{FIGURES_DIR}{figure_prefix}_losses.pdf", bbox_inches="tight")
        plt.close(f)
    estimates, targets = sample_estimates(
        workflow.approximator, simulator,
        cond_keys=("out",), param_keys=param_keys, n_test=N_DIAG_TEST,
    )
    save_param_diagnostics(estimates, targets, variable_names, figure_prefix)

    # Per-chunk estimation + PPC. Each chunk's results are written to a partial
    # CSV so a long run can resume after an interruption: chunks whose partial
    # CSV already exists are skipped on restart. (The partials are valid only for
    # the currently saved network; clear _partial_* if the network changes.)
    print(f"\nEstimating {model_name} posteriors for {len(DATASETS)} data chunks...")
    partial_dir = os.path.join(DATA_DIR, "estimates", f"_partial_{model_name.lower()}")
    os.makedirs(partial_dir, exist_ok=True)

    for dataset_name in DATASETS:
        chunk_csv = os.path.join(partial_dir, dataset_name.replace(".p", ".csv"))
        if os.path.exists(chunk_csv):
            print(f"\n  Chunk: {dataset_name} — already done, skipping")
            continue
        print(f"\n  Chunk: {dataset_name}")
        empirical_data = pd.read_pickle(os.path.join(DATA_DIR, dataset_name))
        emp_data = empirical_data["data_array"]
        id_array = empirical_data["outcome_array"][:, 0]
        age_array = empirical_data["outcome_array"][:, 1]

        # Sample posteriors in sub-chunks (to avoid memory issues).
        n_subchunks = 200
        chunks_out = np.array_split(emp_data, n_subchunks)
        looped = {}
        for counter in range(n_subchunks):
            cond = {"out": chunks_out[counter]}
            looped[counter] = workflow.sample(
                conditions=cond, num_samples=N_POSTERIOR_SAMPLES
            )

        # Concatenate sub-chunks and free memory
        samples = {}
        for key in looped[0].keys():
            samples[key] = np.concatenate([looped[c][key] for c in range(n_subchunks)])
        del looped
        gc.collect()

        n_persons = samples["drifts"].shape[0]
        rmses_all = np.full((n_persons, 20), np.nan)
        acc_all = np.full((n_persons, 2), np.nan)   # accuracy error: [cong, inc]
        frac_excluded_all = np.zeros(n_persons)

        # Quality check all persons; track fraction of excluded samples per person
        for person in range(n_persons):
            frac_excluded_all[person] = quality_check_fn(samples, person, N_MIN_VALID)

        # PPC on a random subsample only. 
        
        ppc_n = min(N_PPC_SUBSAMPLE, n_persons)
        ppc_rng = np.random.default_rng(zlib.crc32(dataset_name.encode()))
        ppc_indices = ppc_rng.choice(n_persons, ppc_n, replace=False)
        print(f"    Running PPC on {ppc_n}/{n_persons} persons...")
        for i, person in enumerate(ppc_indices):
            if (i + 1) % 25 == 0 or i == ppc_n - 1:
                print(f"    PPC {i + 1}/{ppc_n}", end="\r")
            person_samples = {k: samples[k][person] for k in param_keys}
            rmses_all[person], acc_all[person] = run_ppc_person(
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

        # PPC metrics (RT-quantile RMSE: congruent 0-9, incongruent 10-19)
        for i, col in enumerate(RMSE_COLS):
            df_chunk[col] = rmses_all[:, i]

        df_chunk["acc_err_congruent"] = acc_all[:, 0]
        df_chunk["acc_err_incongruent"] = acc_all[:, 1]

        df_chunk.to_csv(chunk_csv, index=False)
        print(f"    {dataset_name}: {n_persons} persons processed -> {chunk_csv}")

        del samples, rmses_all, acc_all, emp_data, df_chunk
        gc.collect()
        try:
            import jax
            jax.clear_caches()
        except Exception:
            pass

    # Combine all partial chunk CSVs (in DATASETS order) into the final CSV.
    final_df = pd.concat(
        [pd.read_csv(os.path.join(partial_dir, d.replace(".p", ".csv")))
         for d in DATASETS],
        ignore_index=True,
    )
    csv_path = os.path.join(DATA_DIR, "estimates", f"iat_results_{model_name.lower()}.csv")
    final_df.to_csv(csv_path, index=False)
    print(f"\n  Saved {csv_path} ({len(final_df)} persons)")

    return workflow


# ═══════════════════════════════════════════════════════════
# DDM
# ═══════════════════════════════════════════════════════════

ddm_adapter = (
    bf.Adapter()
    .as_set(["out"])
    .constrain("drifts", lower=0.0)
    .constrain("thresholds", lower=0.0, upper=8.0)
    .constrain("ndt_correct", lower=0.1, upper=1.0)
    .constrain("ndt_error", lower=0.0)
    .convert_dtype(from_dtype="float64", to_dtype="float32")
    .concatenate(["drifts", "thresholds", "ndt_correct", "ndt_error"],
                 into="inference_variables")
    .concatenate(["out"], into="summary_variables")
)

# ═══════════════════════════════════════════════════════════
# OUM
# ═══════════════════════════════════════════════════════════

oum_adapter = (
    bf.Adapter()
    .as_set(["out"])
    .constrain("drifts", lower=0.0)
    .constrain("thresholds", lower=0.0, upper=8.0)
    .constrain("ndt_correct", lower=0.1, upper=1.0)
    .constrain("ndt_error", lower=0.0)
    .constrain("k", lower=0.0)
    .convert_dtype(from_dtype="float64", to_dtype="float32")
    .concatenate(["drifts", "thresholds", "ndt_correct", "ndt_error", "k"],
                 into="inference_variables")
    .concatenate(["out"], into="summary_variables")
)


if __name__ == "__main__":
    estimate_model(
        model_name="DDM",
        prior_fn=iat_ddm_prior_fun,
        param_keys=["drifts", "thresholds", "ndt_correct", "ndt_error"],
        adapter=ddm_adapter,
        quality_check_fn=quality_check_ddm,
        figure_prefix="figureC4_recovery_iat_ddm",
        variable_names=DDM_LABELS,
    )
    estimate_model(
        model_name="OUM",
        prior_fn=iat_oum_prior_fun,
        param_keys=["drifts", "thresholds", "ndt_correct", "ndt_error", "k"],
        adapter=oum_adapter,
        quality_check_fn=quality_check_oum,
        figure_prefix="figureC3_recovery_iat_oum",
        variable_names=OUM_LABELS,
    )

    print("\n" + "=" * 60)
    print("IAT parameter estimation complete.")
    print("=" * 60)
