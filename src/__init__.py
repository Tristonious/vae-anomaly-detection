"""
src — VAE anomaly detection package.

Public API
----------
Data
    load_data, make_anomaly_split,
    add_black_box, add_gaussian_noise, apply_rotation_only, make_corrupted_set

Models
    build_vae, build_vanilla_autoencoder, VAE

Metrics
    reconstruction_errors, compute_auroc, evaluate_at_threshold,
    evaluate_corruption, severity_sweep

Visualization
    plot_reconstructions, plot_error_histogram, plot_roc_curve,
    plot_multiple_roc_curves, plot_sensitivity_analysis,
    plot_vae_vs_ae_comparison, plot_epoch_study
"""

from .data import (
    load_data,
    make_anomaly_split,
    add_black_box,
    add_gaussian_noise,
    apply_rotation_only,
    make_corrupted_set,
)

from .model import (
    build_encoder,
    build_decoder,
    build_vae,
    build_vanilla_autoencoder,
    VAE,
)

from .metrics import (
    reconstruction_errors,
    compute_auroc,
    evaluate_at_threshold,
    evaluate_corruption,
    severity_sweep,
)

from .viz import (
    plot_reconstructions,
    plot_error_histogram,
    plot_roc_curve,
    plot_multiple_roc_curves,
    plot_sensitivity_analysis,
    plot_vae_vs_ae_comparison,
    plot_epoch_study,
)
