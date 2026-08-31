import logging
from collections.abc import Callable, Iterable
from functools import partial
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from mlx_timeseries_workbench.callbacks import Callback, CallbackList, ProgressBarLogger

logger = logging.getLogger(__name__)


class Trainer:
    """Trainer class for compiling and fitting MLX models."""

    def __init__(self, module: nn.Module):
        """Initialize the trainer with an MLX module.

        :param module: MLX neural network module to train.
        :type module: nn.Module
        """
        super().__init__()
        self.module: nn.Module = module

        self.state: list[dict[str, Any]]
        self.optimizer: optim.Optimizer

        # compiled function for loss function
        self._loss: Callable[[mx.array, mx.array], mx.array]

        # compiled function for train step
        self._train_step: Callable[[mx.array, mx.array], mx.array]

    def compile(
        self,
        optimizer: optim.Optimizer,
        loss: Callable[[nn.Module, mx.array, mx.array], mx.array],
    ) -> None:
        """Compile the training step and loss evaluation functions.

        :param optimizer: Optimizer to use for parameter updates.
        :type optimizer: optim.Optimizer
        :param loss: Loss function accepting (model, X, y) and returning loss scalar.
        :type loss: Callable[[nn.Module, mx.array, mx.array], mx.array]
        :return: None
        :rtype: None
        """
        self.optimizer = optimizer

        self.state = [self.module.state, self.optimizer.state]
        self._train_step = self._build_train_step(nn.value_and_grad(self.module, loss))

        self._loss = self._build_loss_fn(loss)

    def fit(
        self,
        X: mx.array,
        y: mx.array,
        batch_size: int,
        epochs: int,
        verbose: int = 1,
        callbacks: list[Callback] | None = None,
        validation_set: tuple[mx.array, mx.array] | None = None,
    ) -> None:
        """Train the model on the provided dataset.

        :param X: Input training data array.
        :type X: mx.array
        :param y: Target training data array.
        :type y: mx.array
        :param batch_size: Size of each training mini-batch.
        :type batch_size: int
        :param epochs: Number of epochs to train.
        :type epochs: int
        :param verbose: Verbosity mode (0: silent, 1: progress logging), defaults to 1.
        :type verbose: int, optional
        :param callbacks: List of training callbacks, defaults to None.
        :type callbacks: list[Callback] | None, optional
        :param validation_set: Optional tuple of (X_val, y_val) arrays for validation, defaults to None.
        :type validation_set: tuple[mx.array, mx.array] | None, optional
        :return: None
        :rtype: None
        """
        callback_list: CallbackList = CallbackList(callbacks)
        callback_list.append(ProgressBarLogger(epochs=epochs, verbose=verbose))
        callback_list.on_train_begin()

        for e in range(1, epochs + 1):
            callback_list.on_epoch_begin(e)

            total_train_loss = 0.0
            total_samples = 0

            for b, (Xb, yb) in enumerate(self._batch_iterate(batch_size, X, y)):
                callback_list.on_train_batch_begin(b)

                train_loss = self._train_step(Xb, yb)
                mx.eval(self.state)

                # accumulate batch train loss
                actual_batch_size = Xb.shape[0]
                total_train_loss += train_loss.item() * actual_batch_size
                total_samples += actual_batch_size

                callback_list.on_train_batch_end(b)

            # epoch train loss
            epoch_train_loss = total_train_loss / total_samples
            eval_loss = None
            if validation_set is not None:
                eval_loss = self._loss(validation_set[0], validation_set[1])

            logs = {"train": epoch_train_loss}
            if eval_loss is not None:
                logs["eval"] = eval_loss.item()

            callback_list.on_epoch_end(e, logs)

        callback_list.on_train_end()

    def _build_train_step(
        self,
        f: Callable[
            [nn.Module, mx.array, mx.array], tuple[mx.array, dict[str, mx.array]]
        ],
    ) -> Callable[[mx.array, mx.array], mx.array]:
        """Build a compiled version of the train step function.

        :param f: Function returning a tuple of (loss, grads).
        :type f: Callable[[nn.Module, mx.array, mx.array], tuple[mx.array, dict[str, mx.array]]]
        :return: Compiled training step function taking (X, y) and returning loss.
        :rtype: Callable[[mx.array, mx.array], mx.array]
        """

        @partial(mx.compile, inputs=self.state, outputs=self.state)
        def _f(X: mx.array, y: mx.array) -> mx.array:
            """Execute a single forward-backward-update optimization step.

            :param X: Mini-batch input features.
            :type X: mx.array
            :param y: Mini-batch target values.
            :type y: mx.array
            :return: Scalar batch loss array.
            :rtype: mx.array
            """
            loss: mx.array
            grads: dict[str, mx.array]
            loss, grads = f(self.module, X, y)
            self.optimizer.update(self.module, grads)
            return loss

        return _f

    def _build_loss_fn(
        self, f: Callable[[nn.Module, mx.array, mx.array], mx.array]
    ) -> Callable[[mx.array, mx.array], mx.array]:
        """Build a compiled version of the loss function.

        :param f: Loss function accepting (model, X, y).
        :type f: Callable[[nn.Module, mx.array, mx.array], mx.array]
        :return: Compiled evaluation loss function taking (X, y) and returning loss.
        :rtype: Callable[[mx.array, mx.array], mx.array]
        """

        @partial(mx.compile, inputs=self.state)
        def _f(X: mx.array, y: mx.array) -> mx.array:
            """Evaluate the loss on the given batch or dataset.

            :param X: Input features array.
            :type X: mx.array
            :param y: Target array.
            :type y: mx.array
            :return: Scalar loss array.
            :rtype: mx.array
            """
            return f(self.module, X, y)

        return _f

    def _batch_iterate(
        self, batch_size: int, X: mx.array, y: mx.array
    ) -> Iterable[tuple[mx.array, mx.array]]:
        """Yield mini-batches of inputs and targets.

        :param batch_size: Number of samples per mini-batch.
        :type batch_size: int
        :param X: Input data array.
        :type X: mx.array
        :param y: Target data array.
        :type y: mx.array
        :return: Generator yielding tuples of (batch_X, batch_y).
        :rtype: Iterable[tuple[mx.array, mx.array]]
        """
        for s in range(0, len(y), batch_size):
            logger.debug(f"batch iterate: {s}")
            yield X[s : s + batch_size], y[s : s + batch_size]
