#==============================================================================
# Final Project CSCI 8110 - VAE Anomaly Detection
# Tristan Jones
# 
# This project uses a Variational Autoencoder (VAE) to detect anomalies in
# Fashion-MNIST images. The VAE is trained on clean images from all 10 classes,
# then tested on corrupted images (with occlusions, noise, and rotations) to
# evaluate anomaly detection performance.
#==============================================================================

# Utilized Chat-GPT and CoPilot to assit me in creating this code and especially with
# adding pertinent documentation. Essentially I came up with the model and utilized it to 
# assist me in creating it faster. The idea for the project came from the promt and after 
# doing some research I decided on utilizing the VAE for anomaly detection using the 
# fashion MNist dataset.Utilized some refrences matrerials as well from previous assignments, 
# and lectrues. Copilot helped extensively with creating the plotting functions and adding debugging 
# statements throughout the code to make it more user friendly. ChatGPT helped me with getting started
# on the idea of the project and then also for cleaning up my writing with the project writetup. 


# Import required libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import os


# Create output directory for saving all visualization figures
os.makedirs("vae_visualizations", exist_ok=True)

# ========================================
# Corruption Functions for Anomaly Generation
# ========================================
# These functions artificially corrupt clean images to create anomalies.
# The VAE should produce higher reconstruction errors on corrupted images,
# enabling anomaly detection.

def add_black_box(x, box_size=8):
    """
    Add black square occlusions to images, targeting areas with actual content.
    
    This simulates missing data or sensor failures. The algorithm intelligently
    places boxes on the clothing items rather than the black background.
    
    Args:
        x: numpy array of shape (N, 28, 28, 1) with values in [0,1]
        box_size: side length of the square occlusion in pixels
    
    Returns:
        Corrupted images with black boxes placed on content regions
    """
    x_corrupt = x.copy()
    N, H, W, C = x.shape

    for i in range(N):
        # Analyze each image individually to find content regions
        img = x[i, :, :, 0]
        
        # Create binary mask of content (non-background pixels)
        content_mask = img > 0.1  # Threshold to separate clothing from background
        # Only use intelligent placement if image has enough content
        if np.sum(content_mask) > box_size * box_size:
            # Search for positions where box would overlap with content (clothing)
            valid_positions = []
            for top in range(H - box_size):
                for left in range(W - box_size):
                    box_region = content_mask[top:top+box_size, left:left+box_size]
                    # Require at least 30% of box to overlap with content for meaningful occlusion
                    if np.sum(box_region) > (box_size * box_size * 0.3):
                        valid_positions.append((top, left))
            
            if valid_positions:
                # Randomly select from valid positions
                top, left = valid_positions[np.random.randint(len(valid_positions))]
            else:
                # Fallback: random placement if no good positions found
                top = np.random.randint(0, H - box_size)
                left = np.random.randint(0, W - box_size)
        else:
            # Fallback: random placement for sparse/empty images
            top = np.random.randint(0, H - box_size)
            left = np.random.randint(0, W - box_size)
        
        # Apply black square occlusion at selected position
        x_corrupt[i, top:top+box_size, left:left+box_size, :] = 0.0

    return x_corrupt

def add_gaussian_noise(x, sigma=0.3):
    noise = np.random.normal(0.0, sigma, size=x.shape)
    x_noisy = np.clip(x + noise, 0.0, 1.0)
    return x_noisy

def rotate_images(x, max_angle=1.2):  # ~1.2 rad ≈ 70 degrees - much more significant
    # x: (N, 28, 28, 1)
    # Apply different random rotation to each image
    x_rot = x.copy()
    N = x.shape[0]
    
    for i in range(N):
        # Generate unique random angle for each image (between -max_angle and +max_angle)
        angle = np.random.uniform(-max_angle, max_angle)
        angle_degrees = angle * 180 / np.pi  # Convert radians to degrees
        
        # Create rotation layer with this specific angle
        # Setting factor min=max gives fixed rotation instead of random range
        rotation_layer = tf.keras.layers.RandomRotation(
            factor=(angle_degrees/360.0, angle_degrees/360.0),  # Rotation as fraction of full circle
            fill_mode="constant",  # Fill empty areas with constant value
            fill_value=0.0  # Black fill for rotated-out regions
        )
        # Apply rotation to single image and store result
        x_rot[i:i+1] = rotation_layer(x[i:i+1], training=True).numpy()
    
    return x_rot

def apply_rotation_only(x, max_angle=1.2):
    """Apply only rotation corruption with significant angle"""
    return rotate_images(x, max_angle=max_angle)

def make_corrupted_set(x_clean):
    """Apply combined corruption: black box + noise"""
    x_occ  = add_black_box(x_clean, box_size=8)
    x_noisy = add_gaussian_noise(x_occ, sigma=0.3)
    return x_noisy

# ========================================
# Visualization Functions
# ========================================
def plot_side_by_side_reconstructions(vae_model, x_clean_samples, corruption_name="occluded", num_samples=8):
    """
    Creates a 4-row visualization:
    Row 1: Original clean images
    Row 2: Corrupted versions
    Row 3: VAE reconstructions of clean
    Row 4: VAE reconstructions of corrupted
    """
    # Apply the specified corruption type to clean samples
    if corruption_name == "occluded":
        x_corrupted = add_black_box(x_clean_samples, box_size=8)
    elif corruption_name == "noisy":
        x_corrupted = add_gaussian_noise(x_clean_samples, sigma=0.3)
    elif corruption_name == "rotated":
        x_corrupted = apply_rotation_only(x_clean_samples, max_angle=1.2)
    elif corruption_name == "combined":
        x_corrupted = make_corrupted_set(x_clean_samples)
    else:
        x_corrupted = x_clean_samples
    
    # Get VAE reconstructions for both clean and corrupted images
    recon_clean = vae_model.predict(x_clean_samples, verbose=0)
    recon_corrupted = vae_model.predict(x_corrupted, verbose=0)
    
    # Create 4-row grid: original, corrupted, and their reconstructions
    fig, axes = plt.subplots(4, num_samples, figsize=(num_samples * 1.5, 6))
    
    for i in range(num_samples):
        # Row 1: Original clean images (ground truth)
        axes[0, i].imshow(x_clean_samples[i, :, :, 0], cmap='gray', vmin=0, vmax=1)
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_ylabel('Clean\nOriginal', fontsize=10, rotation=0, ha='right', va='center')
        
        # Row 2: Corrupted versions (anomalies)
        axes[1, i].imshow(x_corrupted[i, :, :, 0], cmap='gray', vmin=0, vmax=1)
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_ylabel('Corrupted\nInput', fontsize=10, rotation=0, ha='right', va='center')
        
        # Row 3: VAE reconstruction of clean (should look good)
        axes[2, i].imshow(recon_clean[i, :, :, 0], cmap='gray', vmin=0, vmax=1)
        axes[2, i].axis('off')
        if i == 0:
            axes[2, i].set_ylabel('Clean\nRecon', fontsize=10, rotation=0, ha='right', va='center')
        
        # Row 4: VAE reconstruction of corrupted (should look distorted/poor)
        axes[3, i].imshow(recon_corrupted[i, :, :, 0], cmap='gray', vmin=0, vmax=1)
        axes[3, i].axis('off')
        if i == 0:
            axes[3, i].set_ylabel('Corrupted\nRecon', fontsize=10, rotation=0, ha='right', va='center')
    
    plt.suptitle(f'VAE Reconstructions: Clean vs {corruption_name.capitalize()} Images', fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig(f'vae_visualizations/reconstructions_{corruption_name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: vae_visualizations/reconstructions_{corruption_name}.png")

def plot_error_histogram(errors_clean, errors_corrupted, corruption_name="corrupted"):
    """
    Plot histogram comparing reconstruction errors for clean vs corrupted images
    """
    plt.figure(figsize=(10, 6))
    
    # Plot overlapping histograms of reconstruction errors
    # Clean images (blue) should cluster at low errors (left side)
    # Corrupted images (red) should cluster at high errors (right side)
    plt.hist(errors_clean, bins=50, alpha=0.6, label='Clean Images', color='blue', density=True)
    plt.hist(errors_corrupted, bins=50, alpha=0.6, label=f'{corruption_name.capitalize()} Images', color='red', density=True)
    
    # Add vertical lines showing mean errors for each distribution
    plt.axvline(np.mean(errors_clean), color='blue', linestyle='--', linewidth=2, 
                label=f'Clean Mean: {np.mean(errors_clean):.6f}')
    plt.axvline(np.mean(errors_corrupted), color='red', linestyle='--', linewidth=2,
                label=f'{corruption_name.capitalize()} Mean: {np.mean(errors_corrupted):.6f}')
    
    plt.xlabel('Reconstruction Error (MSE)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('Distribution of Reconstruction Errors: Clean vs Corrupted Images', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'vae_visualizations/error_histogram_{corruption_name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: vae_visualizations/error_histogram_{corruption_name}.png")

def plot_roc_curve(labels, scores, corruption_name="corrupted", auc_score=None):
    """
    Plot ROC curve for anomaly detection
    """
    # Compute ROC curve: TPR (sensitivity) vs FPR (1-specificity) at all thresholds
    fpr, tpr, thresholds = roc_curve(labels, scores)
    if auc_score is None:
        auc_score = roc_auc_score(labels, scores)
    
    plt.figure(figsize=(8, 8))
    # Plot ROC curve (closer to top-left corner = better performance)
    plt.plot(fpr, tpr, linewidth=2, label=f'{corruption_name.capitalize()} (AUC = {auc_score:.4f})')
    # Diagonal line represents random guessing (AUC = 0.5)
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve: Anomaly Detection via Reconstruction Error', fontsize=14)
    plt.legend(fontsize=11, loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'vae_visualizations/roc_curve_{corruption_name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: vae_visualizations/roc_curve_{corruption_name}.png")

def plot_multiple_roc_curves(roc_data_list):
    """
    Plot multiple ROC curves on the same plot
    roc_data_list: list of dicts with keys 'labels', 'scores', 'name', 'auc'
    """
    plt.figure(figsize=(8, 8))
    
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    # Plot each corruption type's ROC curve with different color
    for i, data in enumerate(roc_data_list):
        fpr, tpr, _ = roc_curve(data['labels'], data['scores'])
        auc_val = data.get('auc', roc_auc_score(data['labels'], data['scores']))
        plt.plot(fpr, tpr, linewidth=2, color=colors[i % len(colors)],
                label=f"{data['name']} (AUC = {auc_val:.4f})")
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves: Anomaly Detection for Different Corruption Types', fontsize=14)
    plt.legend(fontsize=10, loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('vae_visualizations/roc_curves_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: vae_visualizations/roc_curves_comparison.png")

def plot_sensitivity_analysis(severity_results, corruption_type):
    """
    Plot AUROC vs corruption severity
    severity_results: list of dicts with 'severity', 'auroc', 'accuracy'
    """
    # Extract metrics from results list
    severities = [r['severity'] for r in severity_results]
    aurocs = [r['auroc'] for r in severity_results]
    accuracies = [r['accuracy'] for r in severity_results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left plot: AUROC vs severity
    # Higher severity should generally lead to higher AUROC (easier to detect)
    ax1.plot(severities, aurocs, 'o-', linewidth=2, markersize=8, color='blue')
    ax1.set_xlabel(f'{corruption_type} Severity', fontsize=12)
    ax1.set_ylabel('AUROC', fontsize=12)
    ax1.set_title(f'Anomaly Detection Performance vs {corruption_type} Severity', fontsize=13)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0.5, 1.05])
    
    # Right plot: Accuracy vs severity
    # Accuracy based on 95th percentile threshold (5% FPR)
    ax2.plot(severities, accuracies, 'o-', linewidth=2, markersize=8, color='green')
    ax2.set_xlabel(f'{corruption_type} Severity', fontsize=12)
    ax2.set_ylabel('Accuracy (95th percentile threshold)', fontsize=12)
    ax2.set_title(f'Detection Accuracy vs {corruption_type} Severity', fontsize=13)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0.5, 1.05])
    
    plt.tight_layout()
    plt.savefig(f'vae_visualizations/sensitivity_{corruption_type.lower()}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: vae_visualizations/sensitivity_{corruption_type.lower()}.png")

def plot_vae_vs_ae_comparison(vae_results, ae_results):
    """
    Compare VAE vs Autoencoder performance
    """
    # Extract AUROC values for each corruption type
    corruption_types = list(vae_results.keys())
    vae_aurocs = [vae_results[c] for c in corruption_types]
    ae_aurocs = [ae_results[c] for c in corruption_types]
    
    # Set up side-by-side bar positions
    x = np.arange(len(corruption_types))
    width = 0.35
    
    # Create bar chart comparing VAE (blue) vs vanilla AE (orange)
    # VAE includes KL divergence in loss, AE only has reconstruction loss
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, vae_aurocs, width, label='VAE', color='blue', alpha=0.8)
    bars2 = ax.bar(x + width/2, ae_aurocs, width, label='Autoencoder', color='orange', alpha=0.8)
    
    ax.set_xlabel('Corruption Type', fontsize=12)
    ax.set_ylabel('AUROC', fontsize=12)
    ax.set_title('VAE vs Autoencoder: Anomaly Detection Performance', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(corruption_types, rotation=15, ha='right')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0.5, 1.05])
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('vae_visualizations/vae_vs_ae_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: vae_visualizations/vae_vs_ae_comparison.png")

def plot_epoch_study(epoch_results):
    """
    Plot AUROC vs training epochs
    """
    # Extract epochs and corresponding AUROC values
    epochs_list = [r['epochs'] for r in epoch_results]
    aurocs = [r['auroc'] for r in epoch_results]
    
    # Plot performance vs training duration
    # Shows diminishing returns as training continues beyond certain point
    plt.figure(figsize=(8, 6))
    plt.plot(epochs_list, aurocs, 'o-', linewidth=2, markersize=10, color='purple')
    plt.xlabel('Training Epochs', fontsize=12)
    plt.ylabel('AUROC (Combined Corruption)', fontsize=12)
    plt.title('Anomaly Detection Performance vs Training Duration', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.ylim([0.5, 1.05])
    
    # Add value labels on each data point for exact reading
    for i, (ep, auc) in enumerate(zip(epochs_list, aurocs)):
        plt.text(ep, auc + 0.02, f'{auc:.4f}', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('vae_visualizations/epoch_study.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: vae_visualizations/epoch_study.png")

# ========================================
# 1) Load Fashion-MNIST
# ========================================
# Dataset: 60,000 training + 10,000 test images (28×28 grayscale)
# 10 classes: T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot
print("\nLOADING FASHION-MNIST DATASET")
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
print(f"✓ Dataset loaded successfully")
print(f"  Train data shape: {x_train.shape}")
print(f"  Test data shape: {x_test.shape}")

# Normalize and add channel dimension
print("\nPREPROCESSING DATA")
x_train = x_train.astype("float32") / 255.0  # Convert to [0, 1] range
x_test  = x_test.astype("float32") / 255.0
x_train = np.expand_dims(x_train, -1)  # (N, 28, 28, 1) for Conv2D
x_test  = np.expand_dims(x_test, -1)
print(f"✓ Normalized pixel values to [0, 1]")
print(f"✓ Added channel dimension")
print(f"  Train shape: {x_train.shape}")
print(f"  Test shape: {x_test.shape}")

# ========================================
# 2) Train on ALL classes (normal data)
#    Create corrupted images as anomalies
# ========================================
# Strategy: Learn representation of clean clothing, detect corrupted images as anomalies
print("\nPREPARING TRAINING DATA")
print("Training set: ALL 10 clothing classes (clean images)")

# Use all training data as normal (all classes, no corruption)
x_train_normal = x_train  # All classes, clean images

# For test set: use clean images as normal
x_test_normal = x_test  # All classes, clean images

# Create corrupted test images as anomalies
# Use same number of anomalies as normal samples for balanced evaluation
print("\nCREATING CORRUPTED ANOMALIES")
num_anomalies = len(x_test_normal)  # Same number as normal test samples
# Randomly sample from test set to corrupt (without replacement)
anomaly_indices = np.random.choice(len(x_test), num_anomalies, replace=False)
x_test_anom = make_corrupted_set(x_test[anomaly_indices])

# Binary labels: 0 = normal (clean), 1 = anomaly (corrupted)
y_test_normal = np.zeros(len(x_test_normal))  # label 0 = normal
y_test_anom   = np.ones(len(x_test_anom))     # label 1 = anomaly

print(f"✓ Train normal (all classes, clean): {x_train_normal.shape}")
print(f"✓ Test normal (all classes, clean): {x_test_normal.shape}")
print(f"✓ Test anomalies (corrupted images): {x_test_anom.shape}")
print(f"  Anomaly ratio: {len(x_test_anom)/(len(x_test_normal)+len(x_test_anom)):.2%}")


# ========================================
# Helper function to build plain Autoencoder
# ========================================
def build_vanilla_autoencoder(latent_dim=16):
    """Build a vanilla autoencoder (no VAE, no KL divergence)
    Used for comparison analysis to show VAE advantages"""
    # Encoder: compress 28×28×1 → latent_dim
    encoder_inputs = layers.Input(shape=(28, 28, 1))
    x = layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(encoder_inputs)  # 14×14×32
    x = layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(x)  # 7×7×64
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    z = layers.Dense(latent_dim, activation="relu", name="latent")(x)  # Deterministic encoding
    
    encoder = models.Model(encoder_inputs, z, name="ae_encoder")
    
    # Decoder: decompress latent_dim → 28×28×1
    latent_inputs = layers.Input(shape=(latent_dim,))
    x = layers.Dense(7 * 7 * 64, activation="relu")(latent_inputs)
    x = layers.Reshape((7, 7, 64))(x)
    x = layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu")(x)  # 14×14×64
    x = layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu")(x)  # 28×28×32
    decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)  # 28×28×1
    
    decoder = models.Model(latent_inputs, decoder_outputs, name="ae_decoder")
    
    # Full autoencoder: direct reconstruction without sampling
    ae_outputs = decoder(z)
    autoencoder = models.Model(encoder_inputs, ae_outputs, name="autoencoder")
    
    return autoencoder

# ========================================
# 3) Build VAE Architecture
# ========================================
print("\nBUILDING VAE ARCHITECTURE")
latent_dim = 16  # Dimension of latent space (compression factor ~98%)
print(f"Latent dimension: {latent_dim}")

# Encoder: Maps 28×28×1 images to latent distribution parameters (μ, σ²)
print("\n[1/3] Building Encoder...")
encoder_inputs = layers.Input(shape=(28, 28, 1))
x = layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(encoder_inputs)  # → 14×14×32
x = layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(x)  # → 7×7×64
x = layers.Flatten()(x)  # → 3136
x = layers.Dense(128, activation="relu")(x)

# Output mean and log-variance for each latent dimension
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)

# Sampling function: Reparameterization trick
# z = μ + σ * ε, where ε ~ N(0,1)
# This allows backpropagation through the stochastic sampling
def sampling(args):
    z_mean, z_log_var = args
    epsilon = tf.random.normal(shape=tf.shape(z_mean))  # Random noise from standard normal
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon  # σ = exp(0.5 * log(σ²))

z = layers.Lambda(sampling, name="z")([z_mean, z_log_var])

encoder = models.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
print("✓ Encoder built successfully")
encoder.summary()

# Decoder: Maps latent vector back to image space
print("\n[2/3] Building Decoder...")
latent_inputs = layers.Input(shape=(latent_dim,))
x = layers.Dense(7 * 7 * 64, activation="relu")(latent_inputs)
x = layers.Reshape((7, 7, 64))(x)  # Reshape to spatial dimensions
x = layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu")(x)  # → 14×14×64
x = layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu")(x)  # → 28×28×32
decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)  # → 28×28×1

decoder = models.Model(latent_inputs, decoder_outputs, name="decoder")
print("✓ Decoder built successfully")
decoder.summary()

# Full VAE model
print("\n[3/3] Building Full VAE Model...")

# Custom VAE class with proper loss computation
# Necessary because VAE loss includes both reconstruction and KL divergence
# which must be computed during training, not at model construction time
class VAE(models.Model):
    def __init__(self, encoder, decoder, beta=1.0, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta  # β-VAE parameter: weight for KL divergence term
        # Trackers for monitoring each loss component
        self.total_loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="recon_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
    
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]
    
    def train_step(self, data):
        # Forward pass with gradient tracking
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            
            # Reconstruction loss: binary cross-entropy per pixel, summed then averaged
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.keras.losses.binary_crossentropy(data, reconstruction),
                    axis=(1, 2)
                )
            )
            
            # KL divergence: how much latent distribution deviates from N(0,1)
            # Formula: -0.5 * Σ(1 + log(σ²) - μ² - σ²)
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
            )
            
            # Total VAE loss = reconstruction + β * KL
            total_loss = reconstruction_loss + self.beta * kl_loss
        
        # Backpropagation
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        # Update metrics
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        
        return {
            "loss": self.total_loss_tracker.result(),
            "recon_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
    
    def test_step(self, data):
        # Evaluation without gradient tracking
        z_mean, z_log_var, z = self.encoder(data)
        reconstruction = self.decoder(z)
        
        # Reconstruction loss
        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(
                tf.keras.losses.binary_crossentropy(data, reconstruction),
                axis=(1, 2)
            )
        )
        
        # KL divergence
        kl_loss = -0.5 * tf.reduce_mean(
            tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
        )
        
        total_loss = reconstruction_loss + self.beta * kl_loss
        
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        
        return {
            "loss": self.total_loss_tracker.result(),
            "recon_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
    
    def call(self, inputs):
        # Forward pass for prediction (encode → sample → decode)
        z_mean, z_log_var, z = self.encoder(inputs)
        return self.decoder(z)

# Instantiate and compile VAE
beta = 1.0  # β=1.0 is standard VAE; β>1 encourages better disentanglement
vae = VAE(encoder, decoder, beta=beta)
vae.compile(optimizer=tf.keras.optimizers.Adam(1e-3))  # Adam with learning rate 0.001
print("✓ VAE compiled successfully")
print(f"  Beta (KL weight): {beta}")
print(f"  Optimizer: Adam (lr=1e-3)")
# Build the model by calling it once (initializes all layers)
_ = vae(x_train_normal[:1])
vae.summary()





# ========================================
# 4) Train VAE on Clean Images
# ========================================
print("\nTRAINING VAE MODEL")
batch_size = 128  # Trade-off between speed and gradient stability
epochs = 30  # Typical convergence point for Fashion-MNIST
print(f"Batch size: {batch_size}")
print(f"Epochs: {epochs}")
print(f"Training samples: {len(x_train_normal)}")
batches_per_epoch = len(x_train_normal) // batch_size
print(f"Steps per epoch: {batches_per_epoch}")

# Custom callback for progress tracking during training
class ProgressCallback(Callback):
    """Provides real-time training progress updates with in-place printing"""
    def __init__(self, batches_per_epoch):
        super().__init__()
        self.batches_per_epoch = batches_per_epoch
        self.current_epoch = 0
    
    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch + 1
        print(f"\nEpoch {self.current_epoch}/{epochs}")
    
    def on_batch_end(self, batch, logs=None):
        # Show progress every 10 batches (less frequent printing = faster training)
        if batch % 10 == 0 or batch == self.batches_per_epoch - 1:
            loss = logs.get('loss', 0)
            recon = logs.get('recon_loss', 0)
            kl = logs.get('kl_loss', 0)
            
            # Use \r for in-place update (GAN-style), except on last batch
            end_char = '\n' if batch == self.batches_per_epoch - 1 else '\r'
            print(f"Epoch {self.current_epoch}/{epochs} | Batch {batch}/{self.batches_per_epoch} | "
                  f"Loss: {loss:.4f} | Recon: {recon:.4f} | KL: {kl:.4f}", end=end_char)
    
    def on_epoch_end(self, epoch, logs=None):
        # Print final epoch metrics
        if logs:
            print(f">>> Epoch {self.current_epoch}/{epochs} COMPLETE | "
                  f"Loss: {logs.get('loss', 0):.4f} | "
                  f"Recon: {logs.get('recon_loss', 0):.4f} | "
                  f"KL: {logs.get('kl_loss', 0):.4f}")
            if 'val_loss' in logs:
                print(f"    Val Loss: {logs.get('val_loss', 0):.4f} | "
                      f"Val Recon: {logs.get('val_recon_loss', 0):.4f} | "
                      f"Val KL: {logs.get('val_kl_loss', 0):.4f}")

# Train VAE on clean images only
# Model learns to reconstruct normal clothing, will struggle with corrupted images
history = vae.fit(
    x_train_normal,  # Only clean training images
    None,  # VAE computes its own targets in train_step()
    epochs=epochs,
    batch_size=batch_size,
    validation_split=0.1,  # Hold out 10% for validation
    shuffle=True,
    callbacks=[ProgressCallback(batches_per_epoch)],
    verbose=0  # Disable default Keras output (use custom callback instead)
)

# ========================================
# 5) Anomaly Detection: Compute Reconstruction Errors
# ========================================
print("\nCOMPUTING RECONSTRUCTION ERRORS")

def reconstruction_errors(model, x):
    """Compute mean squared error between input and reconstruction for each image"""
    print(f"  Computing reconstruction for {len(x)} samples...")
    recon = model.predict(x, batch_size=256, verbose=0)
    # Per-image MSE: average over all pixels (H×W×C)
    errors = np.mean((x - recon) ** 2, axis=(1, 2, 3))
    print(f"  ✓ Complete. Mean error: {np.mean(errors):.6f}")
    return errors

# Compute reconstruction error for clean test images
print("\n[1/2] Computing errors for normal test samples...")
err_normal = reconstruction_errors(vae, x_test_normal)
print(f"  Normal error stats - Min: {np.min(err_normal):.6f}, Max: {np.max(err_normal):.6f}, Std: {np.std(err_normal):.6f}")

# Compute reconstruction error for corrupted test images
print("\n[2/2] Computing errors for anomaly test samples...")
err_anom   = reconstruction_errors(vae, x_test_anom)
print(f"  Anomaly error stats - Min: {np.min(err_anom):.6f}, Max: {np.max(err_anom):.6f}, Std: {np.std(err_anom):.6f}")

# Expectation: anomaly errors should be significantly higher than normal errors
print("\nEVALUATING ANOMALY DETECTION PERFORMANCE")

# Combine all errors and labels for evaluation
scores = np.concatenate([err_normal, err_anom])  # Higher score = more anomalous
labels = np.concatenate([y_test_normal, y_test_anom])  # 0=normal, 1=anomaly
auc = roc_auc_score(labels, scores)
print(f"\n✓ AUROC Score: {auc:.4f} (higher is better, 0.5=random, 1.0=perfect)")

# Set threshold to accept 5% false positive rate on normal data
print("\nCOMPUTING CLASSIFICATION METRICS")
threshold = np.percentile(err_normal, 95)  # 95th percentile = top 5% of normal errors
print(f"Threshold (95th percentile of normal errors): {threshold:.6f}")
y_pred = (scores > threshold).astype(int)  # Flag images with error > threshold

accuracy = np.mean(y_pred == labels)
print(f"\n✓ Accuracy with 95th percentile threshold: {accuracy:.4f}")

# Additional classification metrics
precision = precision_score(labels, y_pred)  # What % of flagged images are actually anomalies?
recall = recall_score(labels, y_pred)  # What % of actual anomalies are detected?
f1 = f1_score(labels, y_pred)  # Harmonic mean of precision and recall
cm = confusion_matrix(labels, y_pred)

print(f"\nFINAL RESULTS")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print(f"AUROC:     {auc:.4f}")
print(f"\nConfusion Matrix:")
print(f"  [[TN={cm[0,0]:4d}, FP={cm[0,1]:4d}]")  # TN=correct normal, FP=false alarm
print(f"   [FN={cm[1,0]:4d}, TP={cm[1,1]:4d}]]")  # FN=missed anomaly, TP=correct anomaly

# ========================================
# 6) Generate Visualizations
# ========================================
print("\n" + "="*60)
print("GENERATING VISUALIZATIONS")
print("="*60)

# 1. Side-by-side reconstructions for different corruption types
print("\n[1/4] Generating side-by-side reconstruction visualizations...")

# Sample random test images for visualization
np.random.seed(42)
sample_indices = np.random.choice(len(x_test), 8, replace=False)
x_samples = x_test[sample_indices]

# Generate 4-row comparison grids (clean, corrupted, reconstruction, error map)
plot_side_by_side_reconstructions(vae, x_samples, corruption_name="occluded", num_samples=8)
plot_side_by_side_reconstructions(vae, x_samples, corruption_name="noisy", num_samples=8)
plot_side_by_side_reconstructions(vae, x_samples, corruption_name="rotated", num_samples=8)
plot_side_by_side_reconstructions(vae, x_samples, corruption_name="combined", num_samples=8)

# 2. Error histogram showing distribution separation
print("\n[2/4] Generating reconstruction error histogram...")
plot_error_histogram(err_normal, err_anom, corruption_name="combined")

# 3. ROC curve for main combined corruption result
print("\n[3/4] Generating ROC curve...")
plot_roc_curve(labels, scores, corruption_name="combined", auc_score=auc)

# 4. Multiple ROC curves comparing different corruption types
print("\n[4/4] Generating comparison ROC curves for different corruption types...")

# Compute reconstruction errors for each corruption type individually
print("  Computing errors for occlusion-only anomalies...")
x_test_occluded = add_black_box(x_test[anomaly_indices], box_size=8)
err_occluded = reconstruction_errors(vae, x_test_occluded)

print("  Computing errors for noise-only anomalies...")
x_test_noisy = add_gaussian_noise(x_test[anomaly_indices], sigma=0.3)
err_noisy = reconstruction_errors(vae, x_test_noisy)

print("  Computing errors for rotation-only anomalies...")
x_test_rotated = apply_rotation_only(x_test[anomaly_indices], max_angle=1.2)
err_rotated = reconstruction_errors(vae, x_test_rotated)

# Prepare data for multi-curve comparison plot
# Each entry compares normal errors vs. specific corruption type errors
roc_data = [
    {
        'labels': np.concatenate([y_test_normal, y_test_anom]),
        'scores': np.concatenate([err_normal, err_anom]),
        'name': 'Combined (Occlusion + Noise)',
        'auc': auc
    },
    {
        'labels': np.concatenate([y_test_normal, np.ones(len(err_occluded))]),
        'scores': np.concatenate([err_normal, err_occluded]),
        'name': 'Occlusion Only',
        'auc': roc_auc_score(np.concatenate([y_test_normal, np.ones(len(err_occluded))]),
                            np.concatenate([err_normal, err_occluded]))
    },
    {
        'labels': np.concatenate([y_test_normal, np.ones(len(err_noisy))]),
        'scores': np.concatenate([err_normal, err_noisy]),
        'name': 'Noise Only',
        'auc': roc_auc_score(np.concatenate([y_test_normal, np.ones(len(err_noisy))]),
                            np.concatenate([err_normal, err_noisy]))
    },
    {
        'labels': np.concatenate([y_test_normal, np.ones(len(err_rotated))]),
        'scores': np.concatenate([err_normal, err_rotated]),
        'name': 'Rotation Only',
        'auc': roc_auc_score(np.concatenate([y_test_normal, np.ones(len(err_rotated))]),
                            np.concatenate([err_normal, err_rotated]))
    }
]

# Generate comparison plot with all 4 ROC curves
plot_multiple_roc_curves(roc_data)

# Summary of detection performance across corruption types
print("\n" + "="*60)
print("AUROC SCORES BY CORRUPTION TYPE")
print("="*60)
for data in roc_data:
    print(f"{data['name']:30s}: {data['auc']:.4f}")

print("\n" + "="*60)
print("ALL VISUALIZATIONS SAVED TO: vae_visualizations/")
print("="*60)

# ========================================
# 7) Additional Analyses
# ========================================

# ========================================
# Analysis 1: Sensitivity to Corruption Strength
# ========================================
# Question: How does detection performance change with corruption severity?
# Expectation: Stronger corruption → easier detection → higher AUROC
print("\n" + "="*60)
print("ANALYSIS 1: SENSITIVITY TO CORRUPTION STRENGTH")
print("="*60)

# Test different noise levels (σ = standard deviation of Gaussian noise)
print("\n[1/3] Testing sensitivity to noise levels...")
noise_levels = [0.1, 0.2, 0.3, 0.4]  # From subtle to severe
noise_results = []

for sigma in noise_levels:
    print(f"  Testing noise σ={sigma}...")
    # Apply noise and compute reconstruction errors
    x_noisy_test = add_gaussian_noise(x_test[anomaly_indices], sigma=sigma)
    err_noisy_test = np.mean((x_noisy_test - vae.predict(x_noisy_test, verbose=0)) ** 2, axis=(1, 2, 3))
    
    # Evaluate detection performance at this noise level
    labels_temp = np.concatenate([y_test_normal, np.ones(len(err_noisy_test))])
    scores_temp = np.concatenate([err_normal, err_noisy_test])
    auroc_temp = roc_auc_score(labels_temp, scores_temp)
    
    # Compute accuracy using 95th percentile threshold
    threshold_temp = np.percentile(err_normal, 95)
    y_pred_temp = (scores_temp > threshold_temp).astype(int)
    accuracy_temp = np.mean(y_pred_temp == labels_temp)
    
    noise_results.append({'severity': sigma, 'auroc': auroc_temp, 'accuracy': accuracy_temp})
    print(f"    σ={sigma}: AUROC={auroc_temp:.4f}, Accuracy={accuracy_temp:.4f}")

plot_sensitivity_analysis(noise_results, 'Noise')

# Test different occlusion box sizes (side length in pixels)
print("\n[2/3] Testing sensitivity to occlusion sizes...")
box_sizes = [4, 8, 12]  # Small, medium, large
box_results = []

for box_size in box_sizes:
    print(f"  Testing box size={box_size}...")
    # Apply occlusion and compute reconstruction errors
    x_occluded_test = add_black_box(x_test[anomaly_indices], box_size=box_size)
    err_occluded_test = np.mean((x_occluded_test - vae.predict(x_occluded_test, verbose=0)) ** 2, axis=(1, 2, 3))
    
    # Evaluate detection performance at this occlusion size
    labels_temp = np.concatenate([y_test_normal, np.ones(len(err_occluded_test))])
    scores_temp = np.concatenate([err_normal, err_occluded_test])
    auroc_temp = roc_auc_score(labels_temp, scores_temp)
    
    threshold_temp = np.percentile(err_normal, 95)
    y_pred_temp = (scores_temp > threshold_temp).astype(int)
    accuracy_temp = np.mean(y_pred_temp == labels_temp)
    
    box_results.append({'severity': box_size, 'auroc': auroc_temp, 'accuracy': accuracy_temp})
    print(f"    box_size={box_size}: AUROC={auroc_temp:.4f}, Accuracy={accuracy_temp:.4f}")

plot_sensitivity_analysis(box_results, 'Occlusion')

# Test different rotation angles (in radians, converted to degrees for display)
print("\n[3/3] Testing sensitivity to rotation angles...")
rotation_angles = [0.3, 0.6, 1.2]  # ~17°, ~34°, ~69°
rotation_results = []

for max_angle in rotation_angles:
    print(f"  Testing rotation angle={max_angle:.1f} rad ({max_angle*180/np.pi:.0f}°)...")
    # Apply rotation and compute reconstruction errors
    x_rotated_test = apply_rotation_only(x_test[anomaly_indices], max_angle=max_angle)
    err_rotated_test = np.mean((x_rotated_test - vae.predict(x_rotated_test, verbose=0)) ** 2, axis=(1, 2, 3))
    
    # Evaluate detection performance at this rotation angle
    labels_temp = np.concatenate([y_test_normal, np.ones(len(err_rotated_test))])
    scores_temp = np.concatenate([err_normal, err_rotated_test])
    auroc_temp = roc_auc_score(labels_temp, scores_temp)
    
    threshold_temp = np.percentile(err_normal, 95)
    y_pred_temp = (scores_temp > threshold_temp).astype(int)
    accuracy_temp = np.mean(y_pred_temp == labels_temp)
    
    rotation_results.append({'severity': max_angle, 'auroc': auroc_temp, 'accuracy': accuracy_temp})
    print(f"    max_angle={max_angle:.1f}: AUROC={auroc_temp:.4f}, Accuracy={accuracy_temp:.4f}")

plot_sensitivity_analysis(rotation_results, 'Rotation')

# ========================================
# Analysis 2: VAE vs Plain Autoencoder
# ========================================
# Question: Does VAE's KL divergence term improve anomaly detection vs vanilla AE?
# Expectation: VAE should generalize better due to regularized latent space
print("\n" + "="*60)
print("ANALYSIS 2: VAE vs VANILLA AUTOENCODER COMPARISON")
print("="*60)

# Train vanilla autoencoder with same architecture (but no KL divergence)
print("\nTraining vanilla autoencoder (30 epochs)...")
autoencoder = build_vanilla_autoencoder(latent_dim=16)
autoencoder.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='mse')

ae_history = autoencoder.fit(
    x_train_normal,
    x_train_normal,  # Autoencoder reconstructs input
    epochs=30,
    batch_size=128,
    validation_split=0.1,
    shuffle=True,
    verbose=0
)
print("✓ Autoencoder training complete")

# Compute AE reconstruction errors
print("\nComputing autoencoder reconstruction errors...")
ae_err_normal = np.mean((x_test_normal - autoencoder.predict(x_test_normal, verbose=0)) ** 2, axis=(1, 2, 3))
ae_err_occluded = np.mean((x_test_occluded - autoencoder.predict(x_test_occluded, verbose=0)) ** 2, axis=(1, 2, 3))
ae_err_noisy = np.mean((x_test_noisy - autoencoder.predict(x_test_noisy, verbose=0)) ** 2, axis=(1, 2, 3))
ae_err_rotated = np.mean((x_test_rotated - autoencoder.predict(x_test_rotated, verbose=0)) ** 2, axis=(1, 2, 3))
ae_err_combined = np.mean((x_test_anom - autoencoder.predict(x_test_anom, verbose=0)) ** 2, axis=(1, 2, 3))

# Compute AUROC for autoencoder
vae_results_dict = {
    'Occlusion': roc_auc_score(np.concatenate([y_test_normal, np.ones(len(err_occluded))]),
                               np.concatenate([err_normal, err_occluded])),
    'Noise': roc_auc_score(np.concatenate([y_test_normal, np.ones(len(err_noisy))]),
                          np.concatenate([err_normal, err_noisy])),
    'Rotation': roc_auc_score(np.concatenate([y_test_normal, np.ones(len(err_rotated))]),
                             np.concatenate([err_normal, err_rotated])),
    'Combined': auc
}

ae_results_dict = {
    'Occlusion': roc_auc_score(np.concatenate([y_test_normal, np.ones(len(ae_err_occluded))]),
                               np.concatenate([ae_err_normal, ae_err_occluded])),
    'Noise': roc_auc_score(np.concatenate([y_test_normal, np.ones(len(ae_err_noisy))]),
                          np.concatenate([ae_err_normal, ae_err_noisy])),
    'Rotation': roc_auc_score(np.concatenate([y_test_normal, np.ones(len(ae_err_rotated))]),
                             np.concatenate([ae_err_normal, ae_err_rotated])),
    'Combined': roc_auc_score(np.concatenate([y_test_normal, np.ones(len(ae_err_combined))]),
                             np.concatenate([ae_err_normal, ae_err_combined]))
}

print("\nVAE vs Autoencoder AUROC Comparison:")
print(f"{'Corruption Type':<15} {'VAE AUROC':<12} {'AE AUROC':<12} {'Difference':<12}")
print("-" * 55)
for corruption_type in vae_results_dict.keys():
    vae_auc = vae_results_dict[corruption_type]
    ae_auc = ae_results_dict[corruption_type]
    diff = vae_auc - ae_auc
    print(f"{corruption_type:<15} {vae_auc:<12.4f} {ae_auc:<12.4f} {diff:+.4f}")

plot_vae_vs_ae_comparison(vae_results_dict, ae_results_dict)

# ========================================
# Analysis 3: Epoch Study
# ========================================
print("\n" + "="*60)
print("ANALYSIS 3: TRAINING DURATION STUDY")
print("="*60)

epoch_configs = [10, 30, 60]
epoch_study_results = []

for num_epochs in epoch_configs:
    print(f"\nTraining VAE for {num_epochs} epochs...")
    
    # Build fresh VAE
    encoder_temp = models.Model(encoder_inputs, [z_mean, z_log_var, z], name=f"encoder_{num_epochs}")
    decoder_temp = models.Model(latent_inputs, decoder_outputs, name=f"decoder_{num_epochs}")
    vae_temp = VAE(encoder_temp, decoder_temp, beta=1.0)
    vae_temp.compile(optimizer=tf.keras.optimizers.Adam(1e-3))
    
    # Train
    vae_temp.fit(
        x_train_normal,
        None,
        epochs=num_epochs,
        batch_size=128,
        validation_split=0.1,
        shuffle=True,
        verbose=0
    )
    
    # Evaluate on combined corruption
    err_normal_temp = np.mean((x_test_normal - vae_temp.predict(x_test_normal, verbose=0)) ** 2, axis=(1, 2, 3))
    
    # Create fresh corrupted samples for fair comparison
    x_test_anom_temp = make_corrupted_set(x_test[anomaly_indices])
    err_anom_temp = np.mean((x_test_anom_temp - vae_temp.predict(x_test_anom_temp, verbose=0)) ** 2, axis=(1, 2, 3))
    
    labels_temp = np.concatenate([y_test_normal, y_test_anom])
    scores_temp = np.concatenate([err_normal_temp, err_anom_temp])
    auroc_temp = roc_auc_score(labels_temp, scores_temp)
    
    epoch_study_results.append({'epochs': num_epochs, 'auroc': auroc_temp})
    print(f"✓ {num_epochs} epochs: AUROC = {auroc_temp:.4f}")

print("\nEpoch Study Results:")
print(f"{'Epochs':<10} {'AUROC':<10}")
print("-" * 20)
for result in epoch_study_results:
    print(f"{result['epochs']:<10} {result['auroc']:<10.4f}")

plot_epoch_study(epoch_study_results)

print("\n" + "="*60)
print("ALL ANALYSES COMPLETE!")
print("="*60)
print("\nTRAINING COMPLETE!\n")
