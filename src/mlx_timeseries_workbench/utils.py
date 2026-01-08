import numpy as np
from matplotlib.axes import Axes


def plot_reconstruction(
    input: np.ndarray, ax: Axes, reconstruction: np.ndarray | None = None
) -> Axes:
    ax.plot(input, "b")

    if reconstruction is not None:
        ax.plot(reconstruction, "r")
        ax.fill_between(np.arange(140), reconstruction, input, color="lightcoral")
        ax.legend(labels=["Input", "Reconstruction", "Error"])

    return ax
