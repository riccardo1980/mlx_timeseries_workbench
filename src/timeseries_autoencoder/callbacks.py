import logging
import time

from abc import ABC
from tensorboardX import SummaryWriter
from typing import List

logger = logging.getLogger(__name__)

class Callback(ABC):
    
    def on_train_begin(self, logs=None):
        """Called at the beginning of training.

        Subclasses should override for any actions to run.

        Args:
            logs: Dict. Currently no data is passed to this argument for this
              method but that may change in the future.
        """

    def on_train_end(self, logs=None):
        """Called at the end of training.

        Subclasses should override for any actions to run.

        Args:
            logs: Dict. Currently the output of the last call to
              `on_epoch_end()` is passed to this argument for this method but
              that may change in the future.
        """

    def on_train_batch_begin(self, batch, logs=None):
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

    def on_train_batch_end(self, batch, logs=None):
        """Called at the end of a training batch in `fit` methods.

        Subclasses should override for any actions to run.

        Note that if the `steps_per_execution` argument to `compile` in
        `Model` is set to `N`, this method will only be called every
        `N` batches.

        Args:
            batch: Integer, index of batch within the current epoch.
            logs: Dict. Aggregated metric results up until this batch.
        """

    def on_epoch_begin(self, epoch, logs=None):
        """Called at the start of an epoch.

        Subclasses should override for any actions to run. This function should
        only be called during TRAIN mode.

        Args:
            epoch: Integer, index of epoch.
            logs: Dict. Currently no data is passed to this argument for this
              method but that may change in the future.
        """

    def on_epoch_end(self, epoch, logs=None):
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

class ProgressBarLogger(Callback):

    def __init__(self, epochs, verbose: int):
        super().__init__()
        self.epochs = epochs
        self.verbose = verbose

        self._epoch_tic = 0

    def on_epoch_begin(self, epoch, logs=None):
        self._epoch_tic = time.perf_counter()
    
    def on_epoch_end(self, epoch, logs):
        toc = time.perf_counter()
        epoch_elapsed_time_sec = toc - self._epoch_tic

        if self.verbose > 0:
            msg: List[str] = [f"Epoch {epoch: 4d} /{self.epochs: 4d}"]
            for m in logs:
                msg.append(f"{m}: {logs[m]:.2e}")
            msg.append(f"time: {epoch_elapsed_time_sec:.3f}[s]")

            print(" - ".join(msg))

class TensorBoardLogger(Callback):
    def __init__(self, log_dir: str):
        super().__init__()
        self.writer = SummaryWriter(log_dir=log_dir)

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            self.writer.add_scalars('losses', logs, epoch)

    def on_train_end(self, logs=None):
        self.writer.close()