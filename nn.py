import numpy as np
from tensor import Tensor


class Linear:
    def __init__(self, in_features, out_features):
        # He initialization: good starting scale for layers followed by ReLU
        # (with a fixed 0.01 the signal gets very small in deeper networks)
        self.W = Tensor(
            np.random.randn(in_features, out_features) * np.sqrt(2 / in_features),
            requires_grad=True
        )

        self.b = Tensor(
            np.zeros((1, out_features)),
            requires_grad=True
        )

    def forward(self, x):
        return x @ self.W + self.b

    def parameters(self):
        return [self.W, self.b]


class MLP:
    # sizes = [2, 16, 16, 1] means: 2 inputs, two hidden layers of 16, 1 output
    def __init__(self, sizes):
        self.layers = []
        for i in range(len(sizes) - 1):
            self.layers.append(Linear(sizes[i], sizes[i + 1]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer.forward(x)
            if i < len(self.layers) - 1:  # no relu after the last layer
                x = x.relu()
        return x

    def parameters(self):
        params = []
        for layer in self.layers:
            params += layer.parameters()
        return params


# ---------------- loss functions ----------------

def mse_loss(pred, target):
    return ((pred - target) ** 2).mean()


def bce_loss(logits, target):
    # binary cross entropy:  -[y*log(p) + (1-y)*log(1-p)]
    # eps keeps log() away from log(0)
    eps = 1e-7
    p = logits.sigmoid()
    loss = -(target * (p + eps).log() + (1 - target) * (1 - p + eps).log())
    return loss.mean()
