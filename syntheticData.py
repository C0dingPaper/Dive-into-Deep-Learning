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

    def get_dataloader(self, train):
        if train:
            indices = list(range(0, self.num_train))

            # The examples are read in random order
            random.shuffle(indices)

        else:
            indices = list(
                range(self.num_train, self.num_train + self.num_val)
            )

        for i in range(0, len(indices), self.batch_size):
            batch_indices = torch.tensor(
                indices[i:i + self.batch_size]
            )

            yield self.X[batch_indices], self.y[batch_indices]

# Initializes the class object 
X, y = next(iter(data.train_dataloader()))

print('X shape:', X.shape, '\ny shape:', y.shape)