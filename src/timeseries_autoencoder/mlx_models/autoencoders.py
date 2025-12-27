import logging

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)


class FullyConnected(nn.Module):  # type: ignore
    def __init__(
        self,
        input_dim: int = 140,
        hidden_dims: list[int] = [32, 16],
        output_dim: int = 8,
    ):
        super().__init__()
        self.layer_sizes = [input_dim] + list(hidden_dims) + [output_dim]
        logger.debug(f"layer sizes: {self.layer_sizes}")

        self.layers = [
            nn.Linear(idim, odim)
            for idim, odim in zip(
                self.layer_sizes[:-1], self.layer_sizes[1:], strict=True
            )
        ]

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.layers[:-1]:
            x = nn.relu(layer(x))
        t: mx.array = self.layers[-1](x)
        return t


class FixedLengthFullyConnectedAutoEncoder(nn.Module):  # type: ignore
    def __init__(
        self, input_dim: int = 140, hidden_dims: list[int] = [32, 16], code_dim: int = 8
    ):
        super().__init__()

        self.encoder = FullyConnected(input_dim, hidden_dims, code_dim)
        self.decoder = FullyConnected(code_dim, hidden_dims[::-1], input_dim)

    def __call__(self, x: mx.array) -> mx.array:
        encoded = nn.relu(self.encoder(x))
        decoded: mx.array = nn.sigmoid(self.decoder(encoded))
        return decoded
