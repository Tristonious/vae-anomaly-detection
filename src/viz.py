"""
viz.py — Visualization functions for VAE anomaly detection results.

All figures are saved to the figures/ directory.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)


def _save(filename):
    path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_reconstructions(vae_model, x_clean_samples, corruption_fn, corruption_name, num_samples=8):
    """
    4-row comparison grid: clean | corrupted | clean reconstruction | corrupted reconstruction.

    Args:
        vae_model:      trained VAE
        x_clean_samples: (N, 28, 28, 1) clean sample images
        corruption_fn:  callable(x) → corrupted images
        corruption_name: label string used in title and filename
        num_samples:    number of columns to show
    """
    x_corrupted    = corruption_fn(x_clean_samples)
    recon_clean    = vae_model.predict(x_clean_samples, verbose=0)
    recon_corrupt  = vae_model.predict(x_corrupted,     verbose=0)

    fig, axes = plt.subplots(4, num_samples, figsize=(num_samples * 1.5, 6))
    row_labels = ["Clean\nOriginal", "Corrupted\nInput", "Clean\nRecon", "Corrupted\nRecon"]
    sources    = [x_clean_samples, x_corrupted, recon_clean, recon_corrupt]

    for row, (data, label) in enumerate(zip(sources, row_labels)):
        for col in range(num_samples):
            axes[row, col].imshow(data[col, :, :, 0], cmap="gray", vmin=0, vmax=1)
            axes[row, col].axis("off")
        axes[row, 0].set_ylabel(label, fontsize=9, rotation=0, ha="right", va="center")

    plt.suptitle(f"VAE Reconstructions: Clean vs {corruption_name.capitalize()} Images",
                 fontsize=14, y=0.98)
    plt.tight_layout()
    _save(f"reconstructions_{corruption_name}.png")


def plot_error_histogram(errors_clean, errors_corrupted, corruption_name="corrupted"):
    """
    Overlapping histogram of reconstruction errors for clean vs. corrupted images.

    Args:
        errors_clean:     1-D array of MSE for clean images
        errors_corrupted: 1-D array of MSE for corrupted images
        corruption_name:  label string used in legend and filename
    """
    plt.figure(figsize=(10, 6))
    plt.hist(errors_clean,     bins=50, alpha=0.6, label="Clean Images",   color="blue",  density=True)
    plt.hist(errors_corrupted, bins=50, alpha=0.6,
             label=f"{corruption_name.capitalize()} Images", color="red", density=True)
    plt.axvline(np.mean(errors_clean), color="blue", linestyle="--", linewidth=2,
                label=f"Clean Mean: {np.mean(errors_clean):.6f}")
    plt.axvline(np.mean(errors_corrupted), color="red", linestyle="--", linewidth=2,
                label=f"{corruption_name.capitalize()} Mean: {np.mean(errors_corrupted):.6f}")
    plt.xlabel("Reconstruction Error (MSE)", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.title("Distribution of Reconstruction Errors: Clean vs Corrupted Images", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(f"error_histogram_{corruption_name}.png")


def plot_roc_curve(labels, scores, corruption_name="corrupted", auc_score=None):
    """
    Single ROC curve for one corruption type.

    Args:
        labels:          binary ground-truth labels
        scores:          anomaly scores
        corruption_name: label string used in legend and filename
        auc_score:       pre-computed AUROC (computed from scores if None)
    """
    fpr, tpr, _ = roc_curve(labels, scores)
    if auc_score is None:
        auc_score = roc_auc_score(labels, scores)

    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, linewidth=2,
             label=f"{corruption_name.capitalize()} (AUC = {auc_score:.4f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Classifier")
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curve: Anomaly Detection via Reconstruction Error", fontsize=14)
    plt.legend(fontsize=11, loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(f"roc_curve_{corruption_name}.png")


def plot_multiple_roc_curves(roc_data_list):
    """
    Overlay multiple ROC curves on a single plot.

    Args:
        roc_data_list: list of dicts, each with keys:
                       'labels', 'scores', 'name', optionally 'auc'
    """
    colors = ["red", "blue", "green", "orange", "purple"]
    plt.figure(figsize=(8, 8))

    for i, data in enumerate(roc_data_list):
        fpr, tpr, _ = roc_curve(data["labels"], data["scores"])
        auc_val = data.get("auc", roc_auc_score(data["labels"], data["scores"]))
        plt.plot(fpr, tpr, linewidth=2, color=colors[i % len(colors)],
                 label=f"{data['name']} (AUC = {auc_val:.4f})")

    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Classifier")
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curves: Anomaly Detection for Different Corruption Types", fontsize=14)
    plt.legend(fontsize=10, loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    _save("roc_curves_comparison.png")


def plot_sensitivity_analysis(severity_results, corruption_type):
    """
    Two-panel plot: AUROC and accuracy vs. corruption severity.

    Args:
        severity_results: list of dicts with keys 'severity', 'auroc', 'accuracy'
        corruption_type:  label string (e.g. 'Noise', 'Occlusion', 'Rotation')
    """
    severities = [r["severity"] for r in severity_results]
    aurocs     = [r["auroc"]    for r in severity_results]
    accuracies = [r["accuracy"] for r in severity_results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(severities, aurocs, "o-", linewidth=2, markersize=8, color="blue")
    ax1.set_xlabel(f"{corruption_type} Severity", fontsize=12)
    ax1.set_ylabel("AUROC", fontsize=12)
    ax1.set_title(f"Anomaly Detection Performance vs {corruption_type} Severity", fontsize=13)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0.5, 1.05])

    ax2.plot(severities, accuracies, "o-", linewidth=2, markersize=8, color="green")
    ax2.set_xlabel(f"{corruption_type} Severity", fontsize=12)
    ax2.set_ylabel("Accuracy (95th percentile threshold)", fontsize=12)
    ax2.set_title(f"Detection Accuracy vs {corruption_type} Severity", fontsize=13)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0.5, 1.05])

    plt.tight_layout()
    _save(f"sensitivity_{corruption_type.lower()}.png")


def plot_vae_vs_ae_comparison(vae_results, ae_results):
    """
    Side-by-side bar chart comparing VAE and autoencoder AUROC.

    Args:
        vae_results: dict mapping corruption_type → AUROC (VAE)
        ae_results:  dict mapping corruption_type → AUROC (autoencoder)
    """
    corruption_types = list(vae_results.keys())
    vae_aurocs = [vae_results[c] for c in corruption_types]
    ae_aurocs  = [ae_results[c]  for c in corruption_types]

    x     = np.arange(len(corruption_types))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, vae_aurocs, width, label="VAE",         color="blue",   alpha=0.8)
    bars2 = ax.bar(x + width / 2, ae_aurocs,  width, label="Autoencoder", color="orange", alpha=0.8)

    ax.set_xlabel("Corruption Type", fontsize=12)
    ax.set_ylabel("AUROC", fontsize=12)
    ax.set_title("VAE vs Autoencoder: Anomaly Detection Performance", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(corruption_types, rotation=15, ha="right")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim([0.5, 1.05])

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, h,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    _save("vae_vs_ae_comparison.png")


def plot_epoch_study(epoch_results):
    """
    Line plot of AUROC vs. training duration.

    Args:
        epoch_results: list of dicts with keys 'epochs', 'auroc'
    """
    epochs_list = [r["epochs"] for r in epoch_results]
    aurocs      = [r["auroc"]  for r in epoch_results]

    plt.figure(figsize=(8, 6))
    plt.plot(epochs_list, aurocs, "o-", linewidth=2, markersize=10, color="purple")
    plt.xlabel("Training Epochs", fontsize=12)
    plt.ylabel("AUROC (Combined Corruption)", fontsize=12)
    plt.title("Anomaly Detection Performance vs Training Duration", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.ylim([0.5, 1.05])

    for ep, auc in zip(epochs_list, aurocs):
        plt.text(ep, auc + 0.02, f"{auc:.4f}", ha="center", fontsize=10)

    plt.tight_layout()
    _save("epoch_study.png")
