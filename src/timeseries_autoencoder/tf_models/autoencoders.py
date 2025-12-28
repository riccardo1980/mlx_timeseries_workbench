import tensorflow as tf
from tensorflow.keras import Sequential, layers
from tensorflow.keras.models import Model


class FixedLengthFullyConnectedAutoEncoder(Model):  # type: ignore
    def __init__(self) -> None:
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
        encoded = self.encoder(x)
        decoded: tf.Tensor = self.decoder(encoded)
        return decoded
