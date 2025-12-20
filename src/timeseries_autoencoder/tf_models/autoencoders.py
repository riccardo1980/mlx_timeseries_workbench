from tensorflow.keras.models import Model
from tensorflow.keras import layers, Sequential


class FixedLengthFullyConnectedAutoEncoder(Model):
  def __init__(self):
    super(FixedLengthFullyConnectedAutoEncoder, self).__init__()
    self.encoder = Sequential([
      layers.Dense(32, activation="relu"),
      layers.Dense(16, activation="relu"),
      layers.Dense(8, activation="relu")])

    self.decoder = Sequential([
      layers.Dense(16, activation="relu"),
      layers.Dense(32, activation="relu"),
      layers.Dense(140, activation="sigmoid")])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded
