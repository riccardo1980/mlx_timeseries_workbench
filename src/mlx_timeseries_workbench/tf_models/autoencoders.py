import tensorflow as tf
from tensorflow.keras import Sequential, layers
from tensorflow.keras.models import Model


class FixedLengthFullyConnectedAutoEncoder(Model):  # type: ignore
    """Fully connected autoencoder for fixed-length time series in TensorFlow/Keras."""

    def __init__(self) -> None:
        """Initialize the fixed-length fully connected autoencoder architecture."""
        super().__init__()
        self.encoder = Sequential(
            [
                layers.Dense(32, activation="relu"),
                layers.Dense(16, activation="relu"),
                layers.Dense(8, activation="relu"),
            ]
        )

        self.decoder = Sequential(
            [
                layers.Dense(16, activation="relu"),
                layers.Dense(32, activation="relu"),
                layers.Dense(140, activation="sigmoid"),
            ]
        )

    def call(self, x: tf.Tensor) -> tf.Tensor:  # type: ignore
        """Forward pass encoding and decoding the input tensor.

        :param x: Input tensor of shape (batch_size, 140).
        :type x: tf.Tensor
        :return: Reconstructed output tensor of shape (batch_size, 140).
        :rtype: tf.Tensor
        """
        encoded = self.encoder(x)
        decoded: tf.Tensor = self.decoder(encoded)
        return decoded
