import time
import numpy as np
import torch
from torch import nn
from d2l import torch as d2l


class HyperParameters:
    # Hyperparameters base class
    def save_hyperparameters(self, ignore=[]):
        raise NotImplementedError


class B(d2l.HyperParameters):
    def __init__(self, a, b, c):
        self.save_hyperparameters(ignore=['c'])

        print('self.a = ', self.a, 'self.b = ', self.b)
        print('There is no self.c = ', not hasattr(self, 'c'))


b = B(a=1, b=2, c=3)


class ProgressBoard(d2l.HyperParameters):  #@save
    """The board that plots data points in animation."""

    def __init__(
        self,
        xlabel=None,
        ylabel=None,
        xlim=None,
        ylim=None,
        xscale='linear',
        yscale='linear',
        ls=['-', '--', '-.', ':'],
        colors=['C0', 'C1', 'C2', 'C3'],
        fig=None,
        axes=None,
        figsize=(3.5, 2.5),
        display=True
    ):
        self.save_hyperparameters()

    def draw(self, x, y, label, every_n=1):
        raise NotImplementedError


board = d2l.ProgressBoard('x')

for x in np.arange(0, 10, 0.1):
    board.draw(x, np.sin(x), 'sin', every_n=2)
    board.draw(x, np.cos(x), 'cos', every_n=10)


class Module(nn.Module, d2l.HyperParameters):  #@save
    """The base class of models."""

    def __init__(
        self,
        plot_train_per_epoch=2,
        plot_valid_per_epoch=1
    ):
        super().__init__()
        self.save_hyperparameters()
        self.board = d2l.ProgressBoard()

    def loss(self, y_hat, y):
        raise NotImplementedError

    def forward(self, X):
        assert hasattr(self, 'net'), 'Neural network is not defined'
        return self.net(X)

    def plot(self, key, value, train):
        """Plot a point in animation."""

        assert hasattr(self, 'trainer'), 'Trainer is not initialized'

        self.board.xlabel = 'epoch'

        if train:
            x = (
                self.trainer.train_batch_idx
                / self.trainer.num_train_batches
            )

            n = (
                self.trainer.num_train_batches
                / self.plot_train_per_epoch
            )

        else:
            x = self.trainer.epoch + 1

            n = (
                self.trainer.num_val_batches
                / self.plot_valid_per_epoch
            )

        self.board.draw(
            x,
            value.to(d2l.cpu()).detach().numpy(),
            ('train_' if train else 'val_') + key,
            every_n=int(n)
        )