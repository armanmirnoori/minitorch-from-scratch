import numpy as np
from tensor import Tensor

class Linear:
    def __init__(self, in_features, out_features):
        self.W = Tensor(
            np.random.randn(in_features, out_features) * 0.01,
            requires_grad=True
        )

        self.b = Tensor(
            np.zeros((1, out_features)),
            requires_grad=True
        )

    def forward(self, x):
        return x @ self.W + self.b