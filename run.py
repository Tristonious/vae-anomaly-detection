"""
run.py — Top-level entry point for VAE anomaly detection on Fashion-MNIST.

Trains a convolutional VAE on clean images, evaluates reconstruction-based
anomaly detection across corruption types and severities, compares against
a baseline autoencoder, and runs a training-duration study.

Usage:
    python run.py
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback

from src import (
    load_data, make_anomaly_split,
    add_black_box, add_gaussian_noise, apply_rotation_only, make_corrupted_set,
    build_vae, build_vanilla_autoencoder,
    reconstruction_errors, compute_auroc, evaluate_at_threshold,
    evaluate_corruption, severity_sweep,
    plot_reconstructions, plot_error_histogram, plot_roc_curve,
    plot_multiple_roc_curves, plot_sensitivity_analysis,
    plot_vae_vs_ae_comparison, plot_epoch_study,
)

os.makedirs("figures", exist_ok=True)
os.makedirs("models",  exist_ok=True)
os.makedirs("results", exist_ok=True)

# -----------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------
LATENT_DIM  = 16
BETA        = 1.0
EPOCHS      = 30
BATCH_SIZE  = 128
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# -----------------------------------------------------------------------
# 1. Load data
# -----------------------------------------------------------------------
print("\n=== LOADING DATA ===")
x_train, y_train, x_test, y_test = load_data()
print(f"  Train: {x_train.shape}  Test: {x_test.shape}")

x_train_normal = x_train
x_test_normal  = x_test

anomaly_indices, y_normal, y_anom = make_anomaly_split(x_test, seed=RANDOM_SEED)
x_test_anom = make_corrupted_set(x_test[anomaly_indices])
print(f"  Normal test: {x_test_normal.shape}  Anomaly test: {x_test_anom.shape}")

# -----------------------------------------------------------------------
# 2. Train VAE
# -----------------------------------------------------------------------
print("\n=== TRAINING VAE ===")
batches_per_epoch = len(x_train_normal) // BATCH_SIZE

class ProgressCallback(Callback):
    def on_epoch_begin(self, epoch, logs=None):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")

    def on_batch_end(self, batch, logs=None):
        if batch % 10 == 0 or batch == batches_per_epoch - 1:
            end = "\n" if batch == batches_per_epoch - 1 else "\r"
            print(
                f"  Batch {batch}/{batches_per_epoch} | "
                f"Loss: {logs.get('loss',0):.4f} | "
                f"Recon: {logs.get('recon_loss',0):.4f} | "
                f"KL: {logs.get('kl_loss',0):.4f}",
                end=end,
            )

    def on_epoch_end(self, epoch, logs=None):
        if logs:
            print(
                f"  Epoch {epoch+1}/{EPOCHS} complete | "
                f"Loss: {logs.get('loss',0):.4f} | "
                f"Val Loss: {logs.get('val_loss',0):.4f}"
            )

vae = build_vae(latent_dim=LATENT_DIM, beta=BETA)
vae.fit(
    x_train_normal, None,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.1,
    shuffle=True,
    callbacks=[ProgressCallback()],
    verbose=0,
)
vae.save_weights(os.path.join("models", "vae_weights.h5"))
print("  VAE weights saved to models/vae_weights.h5")

# -----------------------------------------------------------------------
# 3. Main evaluation (combined corruption)
# -----------------------------------------------------------------------
print("\n=== MAIN EVALUATION (COMBINED CORRUPTION) ===")
err_normal = reconstruction_errors(vae, x_test_normal)
err_anom   = reconstruction_errors(vae, x_test_anom)

auroc, labels, scores = compute_auroc(err_normal, err_anom)
print(f"  AUROC: {auroc:.4f}")

metrics = evaluate_at_threshold(labels, scores, err_normal)
print(f"  Threshold (95th pct): {metrics['threshold']:.6f}")
print(f"  Accuracy:  {metrics['accuracy']:.4f}")
print(f"  Precision: {metrics['precision']:.4f}")
print(f"  Recall:    {metrics['recall']:.4f}")
print(f"  F1:        {metrics['f1']:.4f}")
print(f"  Confusion matrix:\n{metrics['confusion_matrix']}")

plot_error_histogram(err_normal, err_anom, corruption_name="combined")
plot_roc_curve(labels, scores, corruption_name="combined", auc_score=auroc)

# -----------------------------------------------------------------------
# 4. Reconstruction visualizations
# -----------------------------------------------------------------------
print("\n=== RECONSTRUCTION VISUALIZATIONS ===")
rng = np.random.default_rng(RANDOM_SEED)
x_samples = x_test[rng.choice(len(x_test), 8, replace=False)]

for name, fn in [
    ("occluded", lambda x: add_black_box(x, box_size=8)),
    ("noisy",    lambda x: add_gaussian_noise(x, sigma=0.3)),
    ("rotated",  lambda x: apply_rotation_only(x, max_angle=1.2)),
    ("combined", make_corrupted_set),
]:
    plot_reconstructions(vae, x_samples, fn, corruption_name=name)

# -----------------------------------------------------------------------
# 5. ROC by corruption type
# -----------------------------------------------------------------------
print("\n=== ROC BY CORRUPTION TYPE ===")
x_test_pool = x_test[anomaly_indices]

corruption_sets = {
    "Combined (Occlusion + Noise)": x_test_anom,
    "Occlusion Only":               add_black_box(x_test_pool, box_size=8),
    "Noise Only":                   add_gaussian_noise(x_test_pool, sigma=0.3),
    "Rotation Only":                apply_rotation_only(x_test_pool, max_angle=1.2),
}

roc_data          = []
vae_auroc_by_type = {}

for name, x_corr in corruption_sets.items():
    result = evaluate_corruption(vae, err_normal, x_corr, y_normal)
    short  = name.split(" (")[0].replace(" Only", "")
    vae_auroc_by_type[short] = result["auroc"]
    roc_data.append({"labels": result["labels"], "scores": result["scores"],
                     "name": name, "auc": result["auroc"]})
    print(f"  {name}: AUROC = {result['auroc']:.4f}")

plot_multiple_roc_curves(roc_data)

# -----------------------------------------------------------------------
# 6. Sensitivity analyses
# -----------------------------------------------------------------------
print("\n=== SENSITIVITY ANALYSES ===")

plot_sensitivity_analysis(
    severity_sweep(vae, err_normal, x_test_pool,
                   lambda x, s: add_gaussian_noise(x, sigma=s),
                   [0.1, 0.2, 0.3, 0.4], y_normal),
    "Noise",
)
plot_sensitivity_analysis(
    severity_sweep(vae, err_normal, x_test_pool,
                   lambda x, s: add_black_box(x, box_size=s),
                   [4, 8, 12], y_normal),
    "Occlusion",
)
plot_sensitivity_analysis(
    severity_sweep(vae, err_normal, x_test_pool,
                   lambda x, s: apply_rotation_only(x, max_angle=s),
                   [0.3, 0.6, 1.2], y_normal),
    "Rotation",
)

# -----------------------------------------------------------------------
# 7. VAE vs. autoencoder
# -----------------------------------------------------------------------
print("\n=== VAE vs AUTOENCODER ===")
autoencoder = build_vanilla_autoencoder(latent_dim=LATENT_DIM)
autoencoder.fit(
    x_train_normal, x_train_normal,
    epochs=EPOCHS, batch_size=BATCH_SIZE,
    validation_split=0.1, shuffle=True, verbose=0,
)

ae_err_normal    = reconstruction_errors(autoencoder, x_test_normal)
ae_auroc_by_type = {}

for name, x_corr in corruption_sets.items():
    short = name.split(" (")[0].replace(" Only", "")
    ae_err = reconstruction_errors(autoencoder, x_corr)
    ae_auc, _, _ = compute_auroc(ae_err_normal, ae_err)
    ae_auroc_by_type[short] = ae_auc

vae_aligned = {k: vae_auroc_by_type[k] for k in ae_auroc_by_type}
print(f"  {'Corruption':<15} {'VAE':>8} {'AE':>8} {'Diff':>8}")
for k in ae_auroc_by_type:
    diff = vae_aligned[k] - ae_auroc_by_type[k]
    print(f"  {k:<15} {vae_aligned[k]:>8.4f} {ae_auroc_by_type[k]:>8.4f} {diff:>+8.4f}")

plot_vae_vs_ae_comparison(vae_aligned, ae_auroc_by_type)

# -----------------------------------------------------------------------
# 8. Epoch study
# -----------------------------------------------------------------------
print("\n=== EPOCH STUDY ===")
epoch_results = []
for num_epochs in [10, 30, 60]:
    print(f"  Training {num_epochs} epochs...")
    vae_temp = build_vae(latent_dim=LATENT_DIM, beta=BETA)
    vae_temp.fit(
        x_train_normal, None,
        epochs=num_epochs, batch_size=BATCH_SIZE,
        validation_split=0.1, shuffle=True, verbose=0,
    )
    err_n = reconstruction_errors(vae_temp, x_test_normal)
    err_a = reconstruction_errors(vae_temp, make_corrupted_set(x_test[anomaly_indices]))
    auc_t, _, _ = compute_auroc(err_n, err_a)
    epoch_results.append({"epochs": num_epochs, "auroc": auc_t})
    print(f"    {num_epochs} epochs -> AUROC = {auc_t:.4f}")

plot_epoch_study(epoch_results)

print("\n=== DONE - all figures saved to figures/ ===")
