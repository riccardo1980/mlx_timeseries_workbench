from typing import Optional
import matplotlib.pyplot as plt
import numpy as np

def plot_reconstruction(
        input: np.ndarray,
        ax: plt.Axes,
        reconstruction: Optional[np.ndarray] = None
    ) -> None:

    ax.plot(input, 'b')
    
    if reconstruction is not None:
        ax.plot(reconstruction, 'r')
        ax.fill_between(np.arange(140), reconstruction, input, color='lightcoral')
        ax.legend(labels=["Input", "Reconstruction", "Error"])

    return ax
