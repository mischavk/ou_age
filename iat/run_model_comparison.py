"""Run model comparison (DDM vs OUM) for IAT data.

Trains a BayesFlow ModelComparisonApproximator, validates on simulated data,
then applies to all empirical IAT data chunks. Saves figures and CSV results.

Usage:
    python run_model_comparison.py
"""

import numpy as np
import os

os.environ["KERAS_BACKEND"] = "jax"

import gc
import keras
import seaborn as sns
import bayesflow as bf
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from iat_functions import (
    iat_ddm_prior_fun, iat_oum_prior_fun,
    iat_likelihood,
)

FIGURES_DIR = "figures/"
os.makedirs(FIGURES_DIR, exist_ok=True)

DATA_DIR = "iat_data/"

# ── Training hyperparameters ──
NUM_BATCHES = 1000
BATCH_SIZE = 32
EPOCHS = 100

# ── 1. Create simulators ──
print("Creating simulators...")
iat_ddm_simulator = bf.make_simulator([iat_ddm_prior_fun, iat_likelihood])
iat_oum_simulator = bf.make_simulator([iat_oum_prior_fun, iat_likelihood])

# ── 2. Set up adapter and network ──
# Same Large SetTransformer + standardize as the IAT parameter-estimation
# pipeline (summary_dim=20).

adapter = (
    bf.Adapter()
    .as_set(["out"])
    .convert_dtype(from_dtype="float64", to_dtype="float32")
    .drop(["drifts", "thresholds", "ndt_correct", "ndt_error", "k"])
    .rename("model_indices", "inference_variables")
    .concatenate(["out"], into="summary_variables")
)

simulator = bf.simulators.ModelComparisonSimulator(
    simulators=[iat_ddm_simulator, iat_oum_simulator],
    use_mixed_batches=True,
)

summary_network = bf.networks.SetTransformer(
    embed_dims=(128, 128, 128, 128),
    num_heads=(8, 8, 8, 8),
    mlp_depths=(2, 2, 2, 2),
    mlp_widths=(128, 128, 128, 128),
    num_seeds=16,
    summary_dim=20,
)

classifier_network = bf.networks.MLP(widths=(256,) * 16, activation="relu")

approximator = bf.approximators.ModelComparisonApproximator(
    num_models=2,
    classifier_network=classifier_network,
    summary_network=summary_network,
    adapter=adapter,
    standardize=["summary_variables"],
)

# ── 3. Train (or reuse the saved classifier) ──
MODELS_DIR = "models/"
mc_path = f"{MODELS_DIR}iat_model_comparison.keras"
if os.path.exists(mc_path):
    print(f"Loading saved model-comparison network from {mc_path}...")
    approximator = keras.saving.load_model(mc_path)
else:
    learning_rate = keras.optimizers.schedules.CosineDecay(
        1e-4, decay_steps=EPOCHS * NUM_BATCHES, alpha=1e-5
    )
    optimizer = keras.optimizers.AdamW(learning_rate=learning_rate, clipnorm=1.0)
    approximator.compile(optimizer=optimizer)
    print(f"Training model comparison network ({EPOCHS} epochs, {NUM_BATCHES} batches)...")
    history = approximator.fit(
        epochs=EPOCHS,
        num_batches=NUM_BATCHES,
        batch_size=BATCH_SIZE,
        simulator=simulator,
        adapter=adapter,
    )
    os.makedirs(MODELS_DIR, exist_ok=True)
    approximator.save(mc_path)
    print(f"  Saved {mc_path}")

    # Loss curve
    print("Plotting loss curve...")
    f = bf.diagnostics.plots.loss(history=history)
    f.savefig(f"{FIGURES_DIR}loss_model_comparison_iat.pdf", bbox_inches="tight")
    plt.close(f)

# ── 5. Validation on simulated data ──
print("Generating validation predictions (10,000 samples)...")
df_sampled = simulator.sample(10000)
pred_models = np.concatenate([
    approximator.predict(conditions={"out": df_sampled["out"][idx]})
    for idx in np.array_split(np.arange(len(df_sampled["out"])), 50)
], axis=0)

# Calibration (Appendix C1)
print("Plotting calibration...")
f = bf.diagnostics.plots.mc_calibration(
    pred_models=pred_models,
    true_models=df_sampled["model_indices"],
    model_names=["DDM", "OUM"],
)
f.savefig(f"{FIGURES_DIR}figureC1_calibration_iat.pdf", bbox_inches="tight")
plt.close(f)
print(f"  Saved {FIGURES_DIR}figureC1_calibration_iat.pdf")

# Confusion matrix (Appendix C2)
print("Plotting confusion matrix...")
f = bf.diagnostics.plots.mc_confusion_matrix(
    pred_models=pred_models,
    true_models=df_sampled["model_indices"],
    model_names=["DDM", "OUM"],
    normalize="true",
)
f.savefig(f"{FIGURES_DIR}figureC2_confusion_matrix_iat.pdf", bbox_inches="tight")
plt.close(f)
print(f"  Saved {FIGURES_DIR}figureC2_confusion_matrix_iat.pdf")

# ── 6. Empirical data analysis (Figure 4) ──
# Each chunk's per-person model probabilities are written to a partial CSV so the
# long pass is resumable: a chunk whose partial already exists is skipped on
# restart. (Clear _partial_mc if the classifier network changes.) Predictions use
# a FIXED batch size with the last batch padded to that size, so XLA compiles the
# prediction once and reuses it for every batch in every chunk; the earlier
# variable-size np.array_split forced a recompile per chunk, growing the JAX
# compilation cache until the process segfaulted.
PREDICT_BATCH = 1024
MAXCHUNKS = int(os.environ.get("MAXCHUNKS", "0"))  # 0 => all chunks (+ aggregate)

datasets = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".p")])
run_datasets = datasets[:MAXCHUNKS] if MAXCHUNKS else datasets
print(f"\nApplying to {len(datasets)} empirical data chunks...")

partial_dir = os.path.join(DATA_DIR, "estimates", "_partial_mc")
os.makedirs(partial_dir, exist_ok=True)


def predict_model_probs(emp_array):
    """Per-person [DDM, OUM] probabilities, batched at a single fixed shape."""
    n = emp_array.shape[0]
    out = np.empty((n, 2), dtype=np.float32)
    for start in range(0, n, PREDICT_BATCH):
        stop = min(start + PREDICT_BATCH, n)
        batch = emp_array[start:stop]
        m = batch.shape[0]
        if m < PREDICT_BATCH:  # pad the final batch so the compiled shape is reused
            batch = np.concatenate(
                [batch, np.repeat(batch[-1:], PREDICT_BATCH - m, axis=0)], axis=0)
        out[start:stop] = approximator.predict(conditions={"out": batch})[:m]
    return out


for di, dataset_name in enumerate(run_datasets):
    chunk_csv = os.path.join(partial_dir, dataset_name.replace(".p", ".csv"))
    if os.path.exists(chunk_csv):
        print(f"  {dataset_name}: already done, skipping")
        continue
    empirical_data = pd.read_pickle(os.path.join(DATA_DIR, dataset_name))
    emp_array = empirical_data["data_array"]
    id_array = empirical_data["outcome_array"][:, 0].astype(np.int64)

    probs = predict_model_probs(emp_array)
    pd.DataFrame({"id": id_array, "DDM": probs[:, 0], "OUM": probs[:, 1]}).to_csv(
        chunk_csv, index=False)
    print(f"  [{di + 1}/{len(run_datasets)}] {dataset_name}: {probs.shape[0]} persons, "
          f"median DDM={np.median(probs[:, 0]):.3f}, OUM={np.median(probs[:, 1]):.3f}",
          flush=True)
    del empirical_data, emp_array, probs
    gc.collect()

if MAXCHUNKS:
    print(f"\nMAXCHUNKS={MAXCHUNKS}: stopping before aggregation.")
    raise SystemExit(0)

# Combine all partial chunk CSVs (in datasets order)
all_df = pd.concat(
    [pd.read_csv(os.path.join(partial_dir, d.replace(".p", ".csv"))) for d in datasets],
    ignore_index=True)

# A few hundred participants appear twice (the same session_id was prepared into
# two data chunks). Keep the first occurrence so duplicates are not counted twice
# in the model-probability summaries (matches the dedup in run_analyses.py).
n_before = len(all_df)
all_df = all_df.drop_duplicates(subset="id").reset_index(drop=True)
all_probs = all_df[["DDM", "OUM"]].values

print(f"\nTotal: {len(all_df)} persons ({n_before - len(all_df)} duplicate sessions dropped)")
print(f"  Overall median: DDM={np.median(all_probs[:, 0]):.3f}, "
      f"OUM={np.median(all_probs[:, 1]):.3f}")
print(f"  DDM preferred: {np.sum(all_probs[:, 0] > 0.5)}")
print(f"  OUM preferred: {np.sum(all_probs[:, 1] > 0.5)}")

# Save CSV (id included so the results stay auditable and dedup-able)
csv_path = os.path.join(DATA_DIR, "estimates", "iat_model_comparison_results.csv")
os.makedirs(os.path.dirname(csv_path), exist_ok=True)
all_df.to_csv(csv_path, index=False)
print(f"  Saved {csv_path}")

# Violin plot (Figure 4)
fig, ax = plt.subplots(figsize=(6, 5))
sns.violinplot(data=all_probs, ax=ax, cut=0)
ax.set_xticks([0, 1])
ax.set_xticklabels(["DDM", "OUM"], fontsize=12)
ax.set_ylabel("Posterior model probability", fontsize=12)
ax.set_title("IAT — Model comparison", fontsize=14, fontweight="bold")
fig.savefig(f"{FIGURES_DIR}figure4_model_comparison_iat.pdf", bbox_inches="tight")
plt.close(fig)
print(f"  Saved {FIGURES_DIR}figure4_model_comparison_iat.pdf")

print("\nDone.")
