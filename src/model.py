"""
model.py — VAE and baseline autoencoder architecture definitions.
"""

import tensorflow as tf
from tensorflow.keras import layers, models


# -----------------------------------------------------------------------
# VAE components
# -----------------------------------------------------------------------

def build_encoder(latent_dim=16):
    """
    Convolutional encoder: 28×28×1 → (z_mean, z_log_var, z).

    Architecture:
        Conv2D(32, 3, stride=2) → 14×14×32
        Conv2D(64, 3, stride=2) →  7×7×64
        Flatten → Dense(128) → z_mean, z_log_var  (each latent_dim)
        z sampled via reparameterization trick

    Args:
        latent_dim: dimensionality of the latent space

    Returns:
        Keras Model with outputs [z_mean, z_log_var, z]
    """
    encoder_inputs = layers.Input(shape=(28, 28, 1))
    x = layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(encoder_inputs)
    x = layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)

    z_mean    = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)

    def sampling(args):
        z_mean, z_log_var = args
        epsilon = tf.random.normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    z = layers.Lambda(sampling, name="z")([z_mean, z_log_var])

    return models.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")


def build_decoder(latent_dim=16):
    """
    Convolutional decoder: latent vector → 28×28×1 image.

    Architecture:
        Dense → Reshape(7×7×64)
        ConvTranspose(64, 3, stride=2) → 14×14×64
        ConvTranspose(32, 3, stride=2) → 28×28×32
        ConvTranspose( 1, 3, sigmoid)  → 28×28×1

    Args:
        latent_dim: dimensionality of the latent space

    Returns:
        Keras Model mapping latent vector to reconstructed image
    """
    latent_inputs = layers.Input(shape=(latent_dim,))
    x = layers.Dense(7 * 7 * 64, activation="relu")(latent_inputs)
    x = layers.Reshape((7, 7, 64))(x)
    x = layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu")(x)
    decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)

    return models.Model(latent_inputs, decoder_outputs, name="decoder")


class VAE(models.Model):
    """
    Variational Autoencoder with a custom training loop.

    Uses binary cross-entropy reconstruction loss plus a weighted
    KL divergence term (β-VAE formulation, default β=1.0).

    Args:
        encoder: Keras Model returning [z_mean, z_log_var, z]
        decoder: Keras Model mapping z → reconstructed image
        beta: weight on the KL divergence term
    """

    def __init__(self, encoder, decoder, beta=1.0, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.beta    = beta
        self.total_loss_tracker        = tf.keras.metrics.Mean(name="loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="recon_loss")
        self.kl_loss_tracker           = tf.keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def _compute_loss(self, data):
        z_mean, z_log_var, z = self.encoder(data)
        reconstruction = self.decoder(z)

        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(
                tf.keras.losses.binary_crossentropy(data, reconstruction),
                axis=(1, 2),
            )
        )
        kl_loss = -0.5 * tf.reduce_mean(
            tf.reduce_sum(
                1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
                axis=1,
            )
        )
        total_loss = reconstruction_loss + self.beta * kl_loss
        return total_loss, reconstruction_loss, kl_loss

    def train_step(self, data):
        with tf.GradientTape() as tape:
            total_loss, recon_loss, kl_loss = self._compute_loss(data)
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss":       self.total_loss_tracker.result(),
            "recon_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss":    self.kl_loss_tracker.result(),
        }

    def test_step(self, data):
        total_loss, recon_loss, kl_loss = self._compute_loss(data)
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss":       self.total_loss_tracker.result(),
            "recon_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss":    self.kl_loss_tracker.result(),
        }

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        return self.decoder(z)


def build_vae(latent_dim=16, beta=1.0):
    """
    Convenience factory: build and compile a VAE.

    Args:
        latent_dim: latent space dimensionality
        beta: KL divergence weight

    Returns:
        Compiled VAE instance
    """
    encoder = build_encoder(latent_dim)
    decoder = build_decoder(latent_dim)
    vae = VAE(encoder, decoder, beta=beta)
    vae.compile(optimizer=tf.keras.optimizers.Adam(1e-3))
    return vae


# -----------------------------------------------------------------------
# Baseline autoencoder
# -----------------------------------------------------------------------

def build_vanilla_autoencoder(latent_dim=16):
    """
    Standard convolutional autoencoder (no KL divergence).

    Shares the same encoder–decoder architecture as the VAE but uses a
    deterministic latent representation and MSE reconstruction loss.
    Used as a baseline to isolate the effect of VAE latent regularization.

    Args:
        latent_dim: latent space dimensionality

    Returns:
        Compiled Keras Model
    """
    encoder_inputs = layers.Input(shape=(28, 28, 1))
    x = layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(encoder_inputs)
    x = layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    z = layers.Dense(latent_dim, activation="relu", name="latent")(x)

    latent_inputs = layers.Input(shape=(latent_dim,))
    y = layers.Dense(7 * 7 * 64, activation="relu")(latent_inputs)
    y = layers.Reshape((7, 7, 64))(y)
    y = layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu")(y)
    y = layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu")(y)
    decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(y)

    encoder = models.Model(encoder_inputs, z,              name="ae_encoder")
    decoder = models.Model(latent_inputs,  decoder_outputs, name="ae_decoder")

    ae_outputs   = decoder(z)
    autoencoder  = models.Model(encoder_inputs, ae_outputs, name="autoencoder")
    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    return autoencoder
