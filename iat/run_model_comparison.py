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
datasets = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".p")])
print(f"\nApplying to {len(datasets)} empirical data chunks...")

result_dict = {}
for dataset_name in datasets:
    empirical_data = pd.read_pickle(os.path.join(DATA_DIR, dataset_name))
    emp_array = empirical_data["data_array"]
    id_array = empirical_data["outcome_array"][:, 0]

    n_split = 50
    chunks_out = np.array_split(emp_array, n_split)

    gc.collect()
    pred_models_empirical = np.concatenate([
        approximator.predict(conditions={"out": chunks_out[i]})
        for i in range(n_split)
    ], axis=0)

    result_dict[dataset_name] = {"model_probs": pred_models_empirical, "ids": id_array}
    print(f"  {dataset_name}: {pred_models_empirical.shape[0]} persons, "
          f"median DDM={np.median(pred_models_empirical[:, 0]):.3f}, "
          f"OUM={np.median(pred_models_empirical[:, 1]):.3f}")

# Combine all chunks
all_probs = np.concatenate([v["model_probs"] for v in result_dict.values()], axis=0)
all_ids = np.concatenate([v["ids"] for v in result_dict.values()]).astype(np.int64)

# A few hundred participants appear twice (the same session_id was prepared into
# two data chunks). Keep the first occurrence so duplicates are not counted twice
# in the model-probability summaries (matches the dedup in run_analyses.py).
_, first_idx = np.unique(all_ids, return_index=True)
keep = np.sort(first_idx)
n_dup = len(all_ids) - len(keep)
all_probs = all_probs[keep]
all_ids = all_ids[keep]

print(f"\nTotal: {all_probs.shape[0]} persons ({n_dup} duplicate sessions dropped)")
print(f"  Overall median: DDM={np.median(all_probs[:, 0]):.3f}, "
      f"OUM={np.median(all_probs[:, 1]):.3f}")
print(f"  DDM preferred: {np.sum(all_probs[:, 0] > 0.5)}")
print(f"  OUM preferred: {np.sum(all_probs[:, 1] > 0.5)}")

# Save CSV (id included so the results stay auditable and dedup-able)
df_mc = pd.DataFrame({"id": all_ids, "DDM": all_probs[:, 0], "OUM": all_probs[:, 1]})
csv_path = os.path.join(DATA_DIR, "estimates", "iat_model_comparison_results.csv")
os.makedirs(os.path.dirname(csv_path), exist_ok=True)
df_mc.to_csv(csv_path, index=False)
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
