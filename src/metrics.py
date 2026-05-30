"""
metrics.py — Reconstruction error computation and anomaly detection evaluation.
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def reconstruction_errors(model, x, batch_size=256):
    """
    Compute per-image MSE between input and VAE reconstruction.

    Args:
        model: trained VAE or autoencoder with a predict() method
        x: (N, H, W, C) float32 array
        batch_size: batch size for model.predict()

    Returns:
        1-D float array of length N with per-image MSE values
    """
    recon  = model.predict(x, batch_size=batch_size, verbose=0)
    errors = np.mean((x - recon) ** 2, axis=(1, 2, 3))
    return errors


def compute_auroc(err_normal, err_anomaly):
    """
    Compute AUROC treating reconstruction error as the anomaly score.

    Args:
        err_normal:  1-D array of MSE values for clean images  (label 0)
        err_anomaly: 1-D array of MSE values for corrupted images (label 1)

    Returns:
        auroc: float
        labels: concatenated label array
        scores: concatenated score array
    """
    labels = np.concatenate([np.zeros(len(err_normal)), np.ones(len(err_anomaly))])
    scores = np.concatenate([err_normal, err_anomaly])
    auroc  = roc_auc_score(labels, scores)
    return auroc, labels, scores


def evaluate_at_threshold(labels, scores, err_normal, percentile=95):
    """
    Evaluate binary classification at a fixed operating point.

    Threshold is set to the given percentile of clean-image errors,
    allowing approximately (100 - percentile) % false positives on
    normal data.

    Args:
        labels:     ground-truth binary labels (0=normal, 1=anomaly)
        scores:     anomaly scores (higher = more anomalous)
        err_normal: reconstruction errors for clean images only
                    (used to set the threshold)
        percentile: percentile of err_normal used as the threshold

    Returns:
        dict with keys: threshold, accuracy, precision, recall, f1, confusion_matrix
    """
    threshold = np.percentile(err_normal, percentile)
    y_pred    = (scores > threshold).astype(int)

    return {
        "threshold":        threshold,
        "accuracy":         float(np.mean(y_pred == labels)),
        "precision":        float(precision_score(labels, y_pred)),
        "recall":           float(recall_score(labels, y_pred)),
        "f1":               float(f1_score(labels, y_pred)),
        "confusion_matrix": confusion_matrix(labels, y_pred),
    }


def evaluate_corruption(model, err_normal, x_corrupted, y_normal, percentile=95):
    """
    Full evaluation pipeline for one corruption type.

    Computes reconstruction errors for the corrupted set, then returns
    AUROC and threshold-based metrics.

    Args:
        model:       trained model
        err_normal:  pre-computed clean reconstruction errors
        x_corrupted: corrupted test images
        y_normal:    zero labels for the clean test set
        percentile:  threshold percentile (default 95)

    Returns:
        dict with keys: auroc, labels, scores, threshold_metrics
    """
    err_anom            = reconstruction_errors(model, x_corrupted)
    auroc, labels, scores = compute_auroc(err_normal, err_anom)
    threshold_metrics   = evaluate_at_threshold(labels, scores, err_normal, percentile)

    return {
        "auroc":             auroc,
        "labels":            labels,
        "scores":            scores,
        "err_anomaly":       err_anom,
        "threshold_metrics": threshold_metrics,
    }


def severity_sweep(model, err_normal, x_base, corrupt_fn, severities, y_normal, percentile=95):
    """
    Evaluate anomaly detection over a range of corruption severities.

    Args:
        model:       trained model
        err_normal:  pre-computed clean reconstruction errors
        x_base:      base images to corrupt (the anomaly pool)
        corrupt_fn:  callable(x, severity) → corrupted images
        severities:  list of severity values to test
        y_normal:    zero labels for the clean test set
        percentile:  threshold percentile

    Returns:
        list of dicts with keys: severity, auroc, accuracy
    """
    results = []
    for sev in severities:
        x_corrupted = corrupt_fn(x_base, sev)
        err_anom    = reconstruction_errors(model, x_corrupted)
        auroc, labels, scores = compute_auroc(err_normal, err_anom)
        threshold   = np.percentile(err_normal, percentile)
        y_pred      = (scores > threshold).astype(int)
        accuracy    = float(np.mean(y_pred == labels))
        results.append({"severity": sev, "auroc": auroc, "accuracy": accuracy})
    return results
