import logging
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from typing import Callable
from functools import partial

logger = logging.getLogger(__name__)

class TrainHelper():
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

        self.state = None
        self.optimizer: optim.Optimizer = None
        self.loss: Callable = None
        self.loss_and_grad_fn = None

    
    def compile(
        self,
        optimizer: optim.Optimizer,
        loss: Callable,
    ) -> None:
        self.optimizer = optimizer
        self.loss = loss

        self.loss_and_grad_fn = nn.value_and_grad(self.module, self.loss)
        self.state = [self.module.state, self.optimizer.state]
        self._step = self._build_step()
        self._eval_fn = self._build_eval_fn()

    def _build_step(self):
        """
        Builds a compiled version of the step function.
        
        """        
        @partial(mx.compile, inputs=self.state, outputs=self.state)
        def _step(X, y):
            loss, grads = self.loss_and_grad_fn(self.module, X, y)
            self.optimizer.update(self.module, grads)
            return loss

        return _step

    def _build_eval_fn(self):
        """
        Builds a compiled version of the eval function.
        
        """
        @partial(mx.compile, inputs=self.state)
        def _eval_fn(X, y):
            return self.loss(self.module, X, y)

        return _eval_fn

    def step(self, X, y):
        loss = self._step(X, y)
        mx.eval(self.module.state)
        return loss

    def eval_fn(self, X, y):
        return self._eval_fn(X, y)

    
