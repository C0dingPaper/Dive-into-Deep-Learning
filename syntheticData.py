# Generating 1000 samples which are part of R (1000x2)

# Synthetic data is useful for didatic purposes, evaluate properties of learning algorithm and confirm that the
# ML model works as intended

# y = Xw + 𝑏 + 𝝐 . 

import random
import torch
from d2l import torch as d2l

class SyntheticRegressionData(d2l.DataModule):  #@save
    """Synthetic data for linear regression."""

    def __init__(
        self,
        w,
        b,
        noise=0.01,
        num_train=1000,
        num_val=1000,
        batch_size=32
    ):
        super().__init__()
        self.save_hyperparameters()

        n = num_train + num_val

        self.X = torch.randn(n, len(w))
        noise = torch.randn(n, 1) * noise

        self.y = torch.matmul(self.X, w.reshape((-1, 1))) + b + noise