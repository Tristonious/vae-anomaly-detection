# VAE Anomaly Detection on Fashion-MNIST

Convolutional Variational Autoencoder trained on clean Fashion-MNIST images for reconstruction-based anomaly detection. Anomalies are synthetic corruptions (occlusion, Gaussian noise, rotation, combined) applied at test time. Reconstruction error (MSE) is used as the anomaly score.

## Model Architecture

| Component | Details |
|---|---|
| Encoder | Conv2D(32) → Conv2D(64) → Dense(128) → z_mean, z_log_var |
| Latent dim | 16 |
| Reparameterization | z = z_mean + exp(0.5 · z_log_var) · ε |
| Decoder | Dense → Reshape(7×7×64) → ConvTranspose(64) → ConvTranspose(32) → ConvTranspose(1, sigmoid) |
| Loss | Binary cross-entropy + KL divergence (β=1.0) |
| Baseline AE | Same architecture, MSE loss, no KL term |
| Training data | 60,000 Fashion-MNIST (all 10 classes, clean only) |

## Corruption Types

| Type | Description | Default Severity |
|---|---|---|
| Noise | Additive Gaussian noise | σ = 0.3 |
| Occlusion | Black-box mask placed on object region | 8×8 px |
| Rotation | Random rotation per image | max ±1.2 rad (~69°) |
| Combined | Occlusion + Noise applied simultaneously | defaults above |

## Key Results

### AUROC by Corruption Type (VAE vs. Autoencoder)

| Corruption | VAE AUROC | AE AUROC |
|---|---|---|
| Noise | 0.9988 | 0.9988 |
| Combined | 0.9988 | 0.9988 |
| Rotation | 0.8831 | 0.8800 |
| Occlusion | 0.8775 | 0.8770 |

### Sensitivity to Severity (VAE)

| Corruption | Severity | AUROC | Accuracy (95th pct threshold) |
|---|---|---|---|
| Noise | σ=0.10 | 0.940 | 0.77 |
| Noise | σ=0.20 | 0.990 | 0.96 |
| Noise | σ=0.30 | 1.000 | 0.97 |
| Noise | σ=0.40 | 1.000 | 0.97 |
| Occlusion | 4 px | 0.735 | 0.52 |
| Occlusion | 8 px | 0.879 | 0.71 |
| Occlusion | 12 px | 0.903 | 0.77 |
| Rotation | 0.3 rad | 0.715 | 0.58 |
| Rotation | 0.6 rad | 0.820 | 0.69 |
| Rotation | 1.2 rad | 0.880 | 0.76 |

### Training Duration (Combined Corruption)

| Epochs | AUROC |
|---|---|
| 10 | 0.9990 |
| 30 | 0.9989 |
| 60 | 0.9990 |

## Figures

| File | Description |
|---|---|
| [`error_histogram_combined.png`](figures/error_histogram_combined.png) | Reconstruction error distribution: clean vs. combined corruption |
| [`roc_curve_combined.png`](figures/roc_curve_combined.png) | ROC curve, combined corruption (AUROC = 0.9988) |
| [`roc_curves_comparison.png`](figures/roc_curves_comparison.png) | ROC curves for all four corruption types |
| [`vae_vs_ae_comparison.png`](figures/vae_vs_ae_comparison.png) | AUROC comparison: VAE vs. autoencoder |
| [`reconstructions_combined.png`](figures/reconstructions_combined.png) | VAE reconstructions: clean vs. combined |
| [`reconstructions_noisy.png`](figures/reconstructions_noisy.png) | VAE reconstructions: clean vs. noisy |
| [`reconstructions_occluded.png`](figures/reconstructions_occluded.png) | VAE reconstructions: clean vs. occluded |
| [`reconstructions_rotated.png`](figures/reconstructions_rotated.png) | VAE reconstructions: clean vs. rotated |
| [`sensitivity_noise.png`](figures/sensitivity_noise.png) | AUROC and accuracy vs. noise severity |
| [`sensitivity_occlusion.png`](figures/sensitivity_occlusion.png) | AUROC and accuracy vs. occlusion size |
| [`sensitivity_rotation.png`](figures/sensitivity_rotation.png) | AUROC and accuracy vs. rotation angle |
| [`epoch_study.png`](figures/epoch_study.png) | AUROC vs. training duration (10 / 30 / 60 epochs) |

## Project Structure

```
vae-anomaly-detection/
├── run.py                  ← main script: train, evaluate, generate all figures
├── src/                    ← modular components (future refactor)
├── figures/                ← tracked; all output figures for README rendering
├── results/                ← gitignored: metrics CSVs, text logs
├── models/                 ← gitignored: .h5 checkpoints
├── docs/
│   └── vae_anomaly_detection_paper.pdf
├── requirements.txt
├── README.md
└── .gitignore
```

## Usage

**Dataset** — Fashion-MNIST is downloaded automatically by Keras on first run:

```python
from tensorflow.keras.datasets import fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
```

**Install dependencies:**

```bash
pip install -r requirements.txt
```

**Train and evaluate:**

```bash
python run.py
```

All figures are saved to `figures/`. Model checkpoints are saved to `models/` (gitignored).

## Paper

Covers the full methodology, experiment design, sensitivity analyses, VAE vs. autoencoder comparison, and discussion of reconstruction-based anomaly detection behavior on Fashion-MNIST.

[`Final Project VAE Anomaly Detection Paper`](docs/vae_anomaly_detection_paper.pdf)

## Note on AI Assistance

The original implementation for this project was developed as coursework for CSCI 8110. The code in this repository has been refactored with the assistance of Claude (Anthropic) for clarity, structure, and readability. The underlying model architecture, training procedure, corruption generation, anomaly detection methodology, experimental design, and analysis are my own work.

## References

1. D. P. Kingma and M. Welling, "Auto-Encoding Variational Bayes," arXiv:1312.6114, 2014.
2. T. Schlegl et al., "Unsupervised Anomaly Detection with Generative Adversarial Networks," IPMI, 2017.
3. H. Xiao, K. Rasul, and R. Vollgraf, "Fashion-MNIST," arXiv:1708.07747, 2017.
4. P. Bergmann et al., "MVTec AD," CVPR, 2019.
5. L. Ruff et al., "Deep One-Class Classification," ICML, 2018.
