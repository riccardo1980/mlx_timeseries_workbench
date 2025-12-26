import logging
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from typing import Callable, Tuple, Optional
from functools import partial

from timeseries_autoencoder.callbacks import ProgressBarLogger, CallbackList

logger = logging.getLogger(__name__)

class Trainer():
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

        self.state = None
        self.optimizer: optim.Optimizer = None
        
        # compiled function for loss and gradient
        self.loss_and_grad_fn = None

        # compiled function for loss function
        self.loss: Callable = None

    def compile(
        self,
        optimizer: optim.Optimizer,
        loss: Callable
    ) -> None:
        self.optimizer = optimizer

        self.state = [self.module.state, self.optimizer.state]
        self._train_step = self._build_train_step(nn.value_and_grad(self.module, loss))
        
        self._loss = self._build_eval_fn(loss)

    def fit(
            self,
            X:mx.array,
            y:mx.array,
            batch_size: int,
            epochs: int,
            verbose: int = 1,
            callbacks: Optional[list] = None,
            validation_set: Optional[Tuple[mx.array, mx.array]] = None

    ):
        callbacks.append(ProgressBarLogger(epochs=epochs, verbose=verbose))
        callbacks = CallbackList(callbacks)

        callbacks.on_train_begin()
           
        for e in range(1,epochs+1):
            callbacks.on_epoch_begin(e)
            
            for b, (Xb, yb) in enumerate(Trainer._batch_iterate(batch_size, X, y)):
                callbacks.on_train_batch_begin(b)

                train_loss = self._train_step(Xb, yb)
                mx.eval(self.module.state)

                callbacks.on_train_batch_end(b)

            eval_loss = None
            if validation_set is not None:
                eval_loss = self._loss(validation_set[0], validation_set[1])

            logs = {'train': train_loss.item()}
            if eval_loss is not None:
                logs['eval'] = eval_loss.item()

            callbacks.on_epoch_end(e, logs)
            
        callbacks.on_train_end()


    def _build_train_step(self, f: Callable):
        """
        Builds a compiled version of the train step function.
        
        """        
        @partial(mx.compile, inputs=self.state, outputs=self.state)
        def _f(X, y):
            loss, grads = f(self.module, X, y)
            self.optimizer.update(self.module, grads)
            return loss

        return _f

    def _build_eval_fn(self, f: Callable):
        """
        Builds a compiled version of the eval function.
        
        """
        @partial(mx.compile, inputs=self.state)
        def _f(X, y):
            return f(self.module, X, y)

        return _f

    def _batch_iterate(batch_size, X, y):
        for s in range(0, len(y), batch_size):
            logger.debug(f'batch iterate: {s}')
            yield X[s : s + batch_size], y[s : s + batch_size]

    def train_step(self, X, y):
        loss = self._step(X, y)
        mx.eval(self.module.state)
        return loss

    def eval_fn(self, X, y):
        return self._eval_fn(X, y)

    
