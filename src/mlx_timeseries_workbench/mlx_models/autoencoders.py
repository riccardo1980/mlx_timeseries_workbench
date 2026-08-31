import logging

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)


class FullyConnected(nn.Module):  # type: ignore
    """Fully connected neural network module in MLX."""

    def __init__(
        self,
        input_dim: int = 140,
        hidden_dims: list[int] = [32, 16],
        output_dim: int = 8,
    ):
        """Initialize the fully connected network.

        :param input_dim: Dimensionality of the input features, defaults to 140.
        :type input_dim: int, optional
        :param hidden_dims: List of dimensions for the hidden layers, defaults to [32, 16].
        :type hidden_dims: list[int], optional
        :param output_dim: Dimensionality of the output layer, defaults to 8.
        :type output_dim: int, optional
        """
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
        """Forward pass through the fully connected layers.

        :param x: Input array of shape (batch_size, input_dim).
        :type x: mx.array
        :return: Output array of shape (batch_size, output_dim).
        :rtype: mx.array
        """
        for layer in self.layers[:-1]:
            x = nn.relu(layer(x))
        t: mx.array = self.layers[-1](x)
        return t


class FixedLengthFullyConnectedAutoEncoder(nn.Module):  # type: ignore
    """Fully connected autoencoder for fixed-length time series in MLX."""

    def __init__(
        self, input_dim: int = 140, hidden_dims: list[int] = [32, 16], code_dim: int = 8
    ):
        """Initialize the fixed-length fully connected autoencoder.

        :param input_dim: Dimensionality of the input time series, defaults to 140.
        :type input_dim: int, optional
        :param hidden_dims: List of dimensions for the hidden layers, defaults to [32, 16].
        :type hidden_dims: list[int], optional
        :param code_dim: Dimensionality of the latent code representation, defaults to 8.
        :type code_dim: int, optional
        """
        super().__init__()

        self.encoder = FullyConnected(input_dim, hidden_dims, code_dim)
        self.decoder = FullyConnected(code_dim, hidden_dims[::-1], input_dim)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass encoding and decoding the input time series.

        :param x: Input time series array of shape (batch_size, input_dim).
        :type x: mx.array
        :return: Reconstructed time series array of shape (batch_size, input_dim).
        :rtype: mx.array
        """
        encoded = nn.relu(self.encoder(x))
        decoded: mx.array = nn.sigmoid(self.decoder(encoded))
        return decoded
