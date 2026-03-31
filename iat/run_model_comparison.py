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
print("Setting up adapter and network...")
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

summary_network = bf.networks.SetTransformer(summary_dim=20)
classifier_network = bf.networks.MLP(widths=(256,) * 16, activation="relu")

approximator = bf.approximators.ModelComparisonApproximator(
    num_models=2,
    classifier_network=classifier_network,
    summary_network=summary_network,
    adapter=adapter,
    standardize="summary_variables",
)

learning_rate = keras.optimizers.schedules.CosineDecay(
    1e-4, decay_steps=EPOCHS * NUM_BATCHES, alpha=1e-5
)
optimizer = keras.optimizers.AdamW(learning_rate=learning_rate, clipnorm=1.0)
approximator.compile(optimizer=optimizer)

# ── 3. Train ──
print(f"Training model comparison network ({EPOCHS} epochs, {NUM_BATCHES} batches)...")
history = approximator.fit(
    epochs=EPOCHS,
    num_batches=NUM_BATCHES,
    batch_size=BATCH_SIZE,
    simulator=simulator,
    adapter=adapter,
)

# ── 4. Loss curve ──
print("Plotting loss curve...")
f = bf.diagnostics.plots.loss(history=history)
f.savefig(f"{FIGURES_DIR}loss_model_comparison_iat.pdf", bbox_inches="tight")
plt.close(f)

# ── 5. Validation on simulated data ──
print("Generating validation predictions (10,000 samples)...")
df_sampled = simulator.sample(10000)
pred_models = np.concatenate([
    approximator.predict(conditions={"out": x})
    for x in np.array_split(df_sampled["out"], 50)
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
    emp_data = {"out": empirical_data["data_array"]}

    gc.collect()
    pred_models_empirical = np.concatenate([
        approximator.predict(conditions={"out": x})
        for x in np.array_split(emp_data["out"], 50)
    ], axis=0)

    result_dict[dataset_name] = {"model_probs": pred_models_empirical}
    print(f"  {dataset_name}: {pred_models_empirical.shape[0]} persons, "
          f"median DDM={np.median(pred_models_empirical[:, 0]):.3f}, "
          f"OUM={np.median(pred_models_empirical[:, 1]):.3f}")

# Combine all chunks
all_probs = np.concatenate([v["model_probs"] for v in result_dict.values()], axis=0)
print(f"\nTotal: {all_probs.shape[0]} persons")
print(f"  Overall median: DDM={np.median(all_probs[:, 0]):.3f}, "
      f"OUM={np.median(all_probs[:, 1]):.3f}")
print(f"  DDM preferred: {np.sum(all_probs[:, 0] > 0.5)}")
print(f"  OUM preferred: {np.sum(all_probs[:, 1] > 0.5)}")

# Save CSV
df_mc = pd.DataFrame(all_probs, columns=["DDM", "OUM"])
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
