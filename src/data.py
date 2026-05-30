"""
data.py — Data loading, preprocessing, and corruption/anomaly generation.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist


# -----------------------------------------------------------------------
# Loading and preprocessing
# -----------------------------------------------------------------------

def load_data():
    """
    Load and preprocess Fashion-MNIST.

    Returns:
        x_train: (60000, 28, 28, 1) float32 in [0, 1]
        x_test:  (10000, 28, 28, 1) float32 in [0, 1]
        y_train, y_test: integer class labels
    """
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test  = x_test.astype("float32")  / 255.0
    x_train = np.expand_dims(x_train, -1)
    x_test  = np.expand_dims(x_test,  -1)
    return x_train, y_train, x_test, y_test


def make_anomaly_split(x_test, num_anomalies=None, seed=None):
    """
    Sample indices from x_test to use as the anomaly pool.

    Args:
        x_test: full clean test array
        num_anomalies: how many anomaly samples to draw (default: len(x_test))
        seed: optional random seed for reproducibility

    Returns:
        anomaly_indices: 1-D int array of sampled indices
        y_normal: zero labels for all of x_test
        y_anom:   one  labels for the anomaly pool
    """
    if num_anomalies is None:
        num_anomalies = len(x_test)
    rng = np.random.default_rng(seed)
    anomaly_indices = rng.choice(len(x_test), num_anomalies, replace=False)
    y_normal = np.zeros(len(x_test))
    y_anom   = np.ones(num_anomalies)
    return anomaly_indices, y_normal, y_anom


# -----------------------------------------------------------------------
# Corruption / anomaly generation
# -----------------------------------------------------------------------

def add_black_box(x, box_size=8):
    """
    Place a black square on each image, targeting content regions.

    Simulates missing data or a sensor obstruction. The box is placed
    on a region where at least 30 % of pixels exceed the background
    threshold; falls back to random placement when no such region exists.

    Args:
        x: (N, 28, 28, 1) float32 array in [0, 1]
        box_size: side length of the square occlusion in pixels

    Returns:
        Corrupted copy of x with black boxes applied.
    """
    x_corrupt = x.copy()
    N, H, W, _ = x.shape

    for i in range(N):
        img = x[i, :, :, 0]
        content_mask = img > 0.1

        if np.sum(content_mask) > box_size * box_size:
            valid_positions = [
                (top, left)
                for top  in range(H - box_size)
                for left in range(W - box_size)
                if np.sum(content_mask[top:top+box_size, left:left+box_size])
                   > box_size * box_size * 0.3
            ]
            if valid_positions:
                idx = np.random.randint(len(valid_positions))
                top, left = valid_positions[idx]
            else:
                top  = np.random.randint(0, H - box_size)
                left = np.random.randint(0, W - box_size)
        else:
            top  = np.random.randint(0, H - box_size)
            left = np.random.randint(0, W - box_size)

        x_corrupt[i, top:top+box_size, left:left+box_size, :] = 0.0

    return x_corrupt


def add_gaussian_noise(x, sigma=0.3):
    """
    Add i.i.d. Gaussian noise, clipped to [0, 1].

    Args:
        x: float32 array in [0, 1]
        sigma: standard deviation of the noise

    Returns:
        Noisy copy of x.
    """
    noise = np.random.normal(0.0, sigma, size=x.shape)
    return np.clip(x + noise, 0.0, 1.0)


def rotate_images(x, max_angle=1.2):
    """
    Apply a unique random rotation to every image in x.

    Each image is rotated by an angle drawn uniformly from
    [-max_angle, +max_angle] radians. Empty regions are filled black.

    Args:
        x: (N, 28, 28, 1) float32 array
        max_angle: maximum rotation magnitude in radians (~1.2 rad ≈ 69°)

    Returns:
        Rotated copy of x.
    """
    x_rot = x.copy()
    N = x.shape[0]

    for i in range(N):
        angle = np.random.uniform(-max_angle, max_angle)
        angle_degrees = angle * 180 / np.pi
        rotation_layer = tf.keras.layers.RandomRotation(
            factor=(angle_degrees / 360.0, angle_degrees / 360.0),
            fill_mode="constant",
            fill_value=0.0,
        )
        x_rot[i:i+1] = rotation_layer(x[i:i+1], training=True).numpy()

    return x_rot


def apply_rotation_only(x, max_angle=1.2):
    """Convenience wrapper for rotate_images."""
    return rotate_images(x, max_angle=max_angle)


def make_corrupted_set(x_clean, box_size=8, sigma=0.3):
    """
    Apply combined corruption: black-box occlusion followed by Gaussian noise.

    Args:
        x_clean: (N, 28, 28, 1) float32 array of clean images
        box_size: occlusion box side length in pixels
        sigma: Gaussian noise standard deviation

    Returns:
        Combined-corrupted copy of x_clean.
    """
    x_occ   = add_black_box(x_clean, box_size=box_size)
    x_noisy = add_gaussian_noise(x_occ, sigma=sigma)
    return x_noisy
