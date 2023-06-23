import numpy as np
import tensorflow as tf
from tensorflow import keras




class Sampling(keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def encoder_initializer(input_shape=(28, 28, 1), latent_dim=4):
    # input_img = keras.layers.Input(shape=input_shape)

    encoder_inputs = keras.layers.Input(shape=input_shape)
    x = keras.layers.Conv2D(8, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
    # x = keras.layers.BatchNormalization(x)
    x = keras.layers.Conv2D(16, 3, activation="relu", strides=2, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Conv2D(32, 3, activation="relu", strides=2, padding="valid")(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    z_mean = keras.layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = keras.layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    encoder.summary()
    return encoder


def decoder_initializer(latent_dim=4):
    latent_inputs = keras.Input(shape=(latent_dim,))
    x = keras.layers.Dense(128, activation="relu")(latent_inputs)
    x = keras.layers.Dense(3 * 3 * 32, activation="relu")(x)
    x = keras.layers.Reshape((3, 3, 32))(x)
    x = keras.layers.Conv2DTranspose(16, 3, strides=2, activation="relu", padding="valid", output_padding=0)(x)
    x = keras.layers.Conv2DTranspose(8, 3, strides=2, activation="relu", padding="same", output_padding=1)(x)
    decoder_outputs = keras.layers.Conv2DTranspose(1, 3, strides=2, activation="sigmoid", padding="same",
                                                   output_padding=1)(x)
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
    decoder.summary()

    return decoder


class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2))
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

def vae_initializer(latent_dim):
    encoder = encoder_initializer(latent_dim=latent_dim)
    decoder = decoder_initializer(latent_dim=latent_dim)
    vae = VAE(encoder, decoder)
    vae.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3, decay=1e-5))

    return vae
