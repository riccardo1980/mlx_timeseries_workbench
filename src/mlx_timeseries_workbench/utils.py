import numpy as np
from matplotlib.axes import Axes


def plot_reconstruction(
    input: np.ndarray, ax: Axes, reconstruction: np.ndarray | None = None
) -> Axes:
    """Plot an input signal and optionally its reconstruction along with reconstruction error.

    :param input: The original input time series signal.
    :type input: np.ndarray
    :param ax: Matplotlib axes object where the plot will be drawn.
    :type ax: Axes
    :param reconstruction: The reconstructed time series signal, or None.
    :type reconstruction: np.ndarray | None
    :return: The Matplotlib axes object with the plotted data.
    :rtype: Axes
    """
    ax.plot(input, "b")

    if reconstruction is not None:
        ax.plot(reconstruction, "r")
        ax.fill_between(np.arange(140), reconstruction, input, color="lightcoral")
        ax.legend(labels=["Input", "Reconstruction", "Error"])

    return ax

