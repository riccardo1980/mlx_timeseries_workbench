import logging
import time
from abc import ABC
from typing import Any

from tensorboardX import SummaryWriter

logger = logging.getLogger(__name__)


class Callback(ABC):  # noqa: B024
    def on_train_begin(self, logs: dict[str, Any] | None = None) -> None:  # noqa: B027
        """Called at the beginning of training.

        Subclasses should override for any actions to run.

        Args:
            logs: Dict. Currently no data is passed to this argument for this
              method but that may change in the future.
        """
        pass

    def on_train_end(self, logs: dict[str, Any] | None = None) -> None:  # noqa: B027
        """Called at the end of training.

        Subclasses should override for any actions to run.

        Args:
            logs: Dict. Currently the output of the last call to
              `on_epoch_end()` is passed to this argument for this method but
              that may change in the future.
        """
        pass

    def on_train_batch_begin(
        self, batch: int, logs: dict[str, Any] | None = None
    ) -> None:  # noqa: B027
        """Called at the beginning of a training batch in `fit` methods.

        Subclasses should override for any actions to run.

        Note that if the `steps_per_execution` argument to `compile` in
        `Model` is set to `N`, this method will only be called every
        `N` batches.

        Args:
            batch: Integer, index of batch within the current epoch.
            logs: Dict. Currently no data is passed to this argument for this
              method but that may change in the future.
        """
        pass

    def on_train_batch_end(
        self, batch: int, logs: dict[str, Any] | None = None
    ) -> None:  # noqa: B027
        """Called at the end of a training batch in `fit` methods.

        Subclasses should override for any actions to run.

        Note that if the `steps_per_execution` argument to `compile` in
        `Model` is set to `N`, this method will only be called every
        `N` batches.

        Args:
            batch: Integer, index of batch within the current epoch.
            logs: Dict. Aggregated metric results up until this batch.
        """
        pass

    def on_epoch_begin(self, epoch: int, logs: dict[str, Any] | None = None) -> None:  # noqa: B027
        """Called at the start of an epoch.

        Subclasses should override for any actions to run. This function should
        only be called during TRAIN mode.

        Args:
            epoch: Integer, index of epoch.
            logs: Dict. Currently no data is passed to this argument for this
              method but that may change in the future.
        """
        pass

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:  # noqa: B027
        """Called at the end of an epoch.

        Subclasses should override for any actions to run. This function should
        only be called during TRAIN mode.

        Args:
            epoch: Integer, index of epoch.
            logs: Dict, metric results for this training epoch, and for the
              validation epoch if validation is performed. Validation result
              keys are prefixed with `val_`. For training epoch, the values of
              the `Model`'s metrics are returned. Example:
              `{'loss': 0.2, 'accuracy': 0.7}`.
        """
        pass


class CallbackList(Callback):
    def __init__(self, callbacks: list[Callback] | None):
        self.callbacks = callbacks if callbacks is not None else []

    def append(self, callback: Callback) -> None:
        self.callbacks.append(callback)

    def on_train_begin(self, logs: dict[str, Any] | None = None) -> None:
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs: dict[str, Any] | None = None) -> None:
        for callback in self.callbacks:
            callback.on_train_end(logs)

    def on_train_batch_begin(
        self, batch: int, logs: dict[str, Any] | None = None
    ) -> None:
        for callback in self.callbacks:
            callback.on_train_batch_begin(batch, logs)

    def on_train_batch_end(
        self, batch: int, logs: dict[str, Any] | None = None
    ) -> None:
        for callback in self.callbacks:
            callback.on_train_batch_end(batch, logs)

    def on_epoch_begin(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)


class ProgressBarLogger(Callback):
    def __init__(self, epochs: int, verbose: int) -> None:
        super().__init__()
        self.epochs = epochs
        self.verbose = verbose

        self._epoch_tic = 0.0

    def on_epoch_begin(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        self._epoch_tic = time.perf_counter()

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
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
    def __init__(self, log_dir: str) -> None:
        super().__init__()
        self.writer = SummaryWriter(log_dir=log_dir)

    def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
        if logs is not None:
            self.writer.add_scalars("losses", logs, epoch)

    def on_train_end(self, logs: dict[str, Any] | None = None) -> None:
        self.writer.close()
