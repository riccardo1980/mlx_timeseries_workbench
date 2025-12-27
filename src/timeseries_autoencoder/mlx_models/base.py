import logging
from collections.abc import Callable, Iterable
from functools import partial
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from timeseries_autoencoder.callbacks import Callback, CallbackList, ProgressBarLogger

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, module: nn.Module):
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
        callback_list: CallbackList = CallbackList(callbacks)
        callback_list.append(ProgressBarLogger(epochs=epochs, verbose=verbose))
        callback_list.on_train_begin()

        for e in range(1, epochs + 1):
            callback_list.on_epoch_begin(e)

            for b, (Xb, yb) in enumerate(self._batch_iterate(batch_size, X, y)):
                callback_list.on_train_batch_begin(b)

                train_loss = self._train_step(Xb, yb)
                mx.eval(self.module.state)

                callback_list.on_train_batch_end(b)

            eval_loss = None
            if validation_set is not None:
                eval_loss = self._loss(validation_set[0], validation_set[1])

            logs = {"train": train_loss.item()}
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
        """
        Builds a compiled version of the train step function.

        """

        @partial(mx.compile, inputs=self.state, outputs=self.state)
        def _f(X: mx.array, y: mx.array) -> mx.array:
            loss: mx.array
            grads: dict[str, mx.array]
            loss, grads = f(self.module, X, y)
            self.optimizer.update(self.module, grads)
            return loss

        return _f

    def _build_loss_fn(
        self, f: Callable[[nn.Module, mx.array, mx.array], mx.array]
    ) -> Callable[[mx.array, mx.array], mx.array]:
        """
        Builds a compiled version of the loss function.

        """

        @partial(mx.compile, inputs=self.state)
        def _f(X: mx.array, y: mx.array) -> mx.array:
            return f(self.module, X, y)

        return _f

    def _batch_iterate(
        self, batch_size: int, X: mx.array, y: mx.array
    ) -> Iterable[tuple[mx.array, mx.array]]:
        for s in range(0, len(y), batch_size):
            logger.debug(f"batch iterate: {s}")
            yield X[s : s + batch_size], y[s : s + batch_size]
