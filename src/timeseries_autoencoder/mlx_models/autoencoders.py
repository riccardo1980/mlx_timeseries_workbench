import mlx.nn as nn
from typing import List

import logging
logger = logging.getLogger(__name__)


class FullyConnected(nn.Module):
  def __init__(self, input_dim: int = 140, hidden_dims: List[int] = [32, 16], output_dim: int = 8):
    super().__init__()
    self.layer_sizes = [input_dim] + hidden_dims + [output_dim]
    logger.debug(f'layer sizes: {self.layer_sizes}')

    self.layers = [
      nn.Linear(idim, odim)
      for idim, odim in zip(self.layer_sizes[:-1], self.layer_sizes[1:])
    ]

  def __call__(self, x):
    for layer in self.layers[:-1]:
      x = nn.relu(layer(x))

    return self.layers[-1](x)
    


class FixedLengthFullyConnectedAutoEncoder(nn.Module):
  def __init__(self, input_dim: int = 140, hidden_dims: List[int] = [32, 16], code_dim: int = 8):
    super().__init__()
    
    self.encoder = FullyConnected(input_dim, hidden_dims, code_dim)
    self.decoder = FullyConnected(code_dim, hidden_dims[::-1], input_dim)

  def __call__(self, x):
    encoded = nn.relu(self.encoder(x))
    decoded = nn.sigmoid(self.decoder(encoded))
    return decoded
