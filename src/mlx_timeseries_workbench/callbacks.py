import logging
import time
from abc import ABC
from typing import Any

from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)


class Callback(ABC):  # noqa: B024
    """Abstract base class used to build new callbacks."""

    def on_train_begin(self, logs: dict[str, Any] | None = None) -> None:  # noqa: B027
        """Called at the beginning of training.

        Subclasses should override for any actions to run.

        :param logs: Metric dictionary. Currently no data is passed to this argument for this
            method, defaults to None.
        :type logs: dict[str, Any] | None, optional
        :return: None
        :rtype: None
        """
        pass

    def on_train_end(self, logs: dict[str, Any] | None = None) -> None:  # noqa: B027
        """Called at the end of training.

        Subclasses should override for any actions to run.

        :param logs: Metric dictionary containing results from the last epoch, defaults to None.
        :type logs: dict[str, Any] | None, optional
        :return: None
        :rtype: None
        """
        pass

    def on_train_batch_begin(
        self, batch: int, logs: dict[str, Any] | None = None
    ) -> None:  # noqa: B027
        """Called at the beginning of a training batch in `fit` methods.

        Subclasses should override for any actions to run.

        :param batch: Index of batch within the current epoch.
        :type batch: int
        :param logs: Metric dictionary, defaults to None.
        :type logs: dict[str, Any] | None, optional
        :return: None
        :rtype: None
        """
        pass

    def on_train_batch_end(
        self, batch: int, logs: dict[str, Any] | None = None
    ) -> None:  # noqa: B027
        """Called at the end of a training batch in `fit` methods.

        Subclasses should override for any actions to run.

        :param batch: Index of batch within the current epoch.
        :type batch: int
        :param logs: Aggregated metric results up until this batch, defaults to None.
        :type logs: dict[str, Any] | None, optional
        :return: None
        :rtype: None
        """
        pass

    def on_epoch_begin(self, epoch: int, logs: dict[str, Any] | None = None) -> None:  # noqa: B027
        """Called at the start of an epoch.

        Subclasses should override for any actions to run. This function should
        only be called during train mode.

        :param epoch: Index of the epoch.
        :type epoch: int
        :param logs: Metric dictionary, defaults to None.
        :type logs: dict[str, Any] | None, optional
        :return: None
        :rtype: None
        """
        pass

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:  # noqa: B027
        """Called at the end of an epoch.

        Subclasses should override for any actions to run. This function should
        only be called during train mode.

        :param epoch: Index of the epoch.
        :type epoch: int
        :param logs: Metric results for this training epoch, and for the
            validation epoch if validation is performed, defaults to None.
        :type logs: dict[str, Any] | None, optional
        :return: None
        :rtype: None
        """
        pass


class CallbackList(Callback):
    """Container managing a list of callbacks."""

    def __init__(self, callbacks: list[Callback] | None) -> None:
        """Initialize callback list container.

        :param callbacks: List of callback instances, defaults to None.
        :type callbacks: list[Callback] | None
        """
        self.callbacks = callbacks if callbacks is not None else []

    def append(self, callback: Callback) -> None:
        """Append a callback to the list.

        :param callback: Callback instance to append.
        :type callback: Callback
        :return: None
        :rtype: None
        """
        self.callbacks.append(callback)

    def on_train_begin(self, logs: dict[str, Any] | None = None) -> None:
        """Dispatch on_train_begin to all registered callbacks.

        :param logs: Metric dictionary, defaults to None.
        :type logs: dict[str, Any] | None, optional
        :return: None
        :rtype: None
        """
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs: dict[str, Any] | None = None) -> None:
        """Dispatch on_train_end to all registered callbacks.

        :param logs: Metric dictionary, defaults to None.
        :type logs: dict[str, Any] | None, optional
        :return: None
        :rtype: None
        """
        for callback in self.callbacks:
            callback.on_train_end(logs)

    def on_train_batch_begin(
        self, batch: int, logs: dict[str, Any] | None = None
    ) -> None:
        """Dispatch on_train_batch_begin to all registered callbacks.

        :param batch: Index of the batch.
        :type batch: int
        :param logs: Metric dictionary, defaults to None.
        :type logs: dict[str, Any] | None, optional
        :return: None
        :rtype: None
        """
        for callback in self.callbacks:
            callback.on_train_batch_begin(batch, logs)

    def on_train_batch_end(
        self, batch: int, logs: dict[str, Any] | None = None
    ) -> None:
        """Dispatch on_train_batch_end to all registered callbacks.

        :param batch: Index of the batch.
        :type batch: int
        :param logs: Metric dictionary, defaults to None.
        :type logs: dict[str, Any] | None, optional
        :return: None
        :rtype: None
        """
        for callback in self.callbacks:
            callback.on_train_batch_end(batch, logs)

    def on_epoch_begin(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        """Dispatch on_epoch_begin to all registered callbacks.

        :param epoch: Index of the epoch.
        :type epoch: int
        :param logs: Metric dictionary, defaults to None.
        :type logs: dict[str, Any] | None, optional
        :return: None
        :rtype: None
        """
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        """Dispatch on_epoch_end to all registered callbacks.

        :param epoch: Index of the epoch.
        :type epoch: int
        :param logs: Metric dictionary, defaults to None.
        :type logs: dict[str, Any] | None, optional
        :return: None
        :rtype: None
        """
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)


class ProgressBarLogger(Callback):
    """Callback that prints training progress and metrics to stdout."""

    def __init__(self, epochs: int, verbose: int) -> None:
        """Initialize progress bar logger.

        :param epochs: Total number of epochs.
        :type epochs: int
        :param verbose: Verbosity level (0: silent, >0: print metrics).
        :type verbose: int
        """
        super().__init__()
        self.epochs = epochs
        self.verbose = verbose

        self._epoch_tic = 0.0

    def on_epoch_begin(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        """Record epoch start time.

        :param epoch: Index of the epoch.
        :type epoch: int
        :param logs: Metric dictionary, defaults to None.
        :type logs: dict[str, Any] | None, optional
        :return: None
        :rtype: None
        """
        self._epoch_tic = time.perf_counter()

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        """Compute epoch elapsed time and print epoch statistics.

        :param epoch: Index of the epoch.
        :type epoch: int
        :param logs: Metric dictionary for the completed epoch, defaults to None.
        :type logs: dict[str, Any] | None, optional
        :return: None
        :rtype: None
        """
        toc = time.perf_counter()
        epoch_elapsed_time_sec = toc - self._epoch_tic

        if self.verbose > 0:
            msg: list[str] = [f"Epoch {epoch: 4d} /{self.epochs: 4d}"]
            if logs is not None:
                for k, v in logs.items():
                    msg.append(f"{k}: {v:.2e}")
            msg.append(f"time: {epoch_elapsed_time_sec:.3f}[s]")

            print(" - ".join(msg))


class TensorBoardLogger(Callback):
    """Callback that writes training metrics to TensorBoard logs."""

    def __init__(self, log_dir: str) -> None:
        """Initialize TensorBoard logger.

        :param log_dir: Directory where TensorBoard events will be written.
        :type log_dir: str
        """
        super().__init__()
        self.writer = SummaryWriter(log_dir=log_dir)

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        """Log epoch metrics as scalar values to TensorBoard.

        :param epoch: Index of the epoch.
        :type epoch: int
        :param logs: Metric dictionary for the epoch, defaults to None.
        :type logs: dict[str, Any] | None, optional
        :return: None
        :rtype: None
        """
        if logs is not None:
            self.writer.add_scalars("losses", logs, epoch)

    def on_train_end(self, logs: dict[str, Any] | None = None) -> None:
        """Close the TensorBoard summary writer.

        :param logs: Final metrics dictionary, defaults to None.
        :type logs: dict[str, Any] | None, optional
        :return: None
        :rtype: None
        """
        self.writer.close()
