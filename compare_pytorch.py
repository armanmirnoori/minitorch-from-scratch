# train the SAME network (same starting weights, same data, same lr)
# with minitorch and with PyTorch, and compare loss + gradients at every step.
# if our autograd is correct, the two runs should be practically identical.

import numpy as np
import torch

from tensor import Tensor
from nn import MLP, bce_loss
from optim import SGD
from data import make_moons

np.random.seed(0)
X, y = make_moons(n=400, noise=0.15)
steps, lr = 200, 0.5

# ---- minitorch model ----
model = MLP([2, 16, 16, 1])
optimizer = SGD(model.parameters(), lr=lr)

# ---- pytorch model with a copy of the same starting weights ----
torch_params = [torch.tensor(p.data.copy(), requires_grad=True) for p in model.parameters()]
torch_opt = torch.optim.SGD(torch_params, lr=lr)
X_t = torch.tensor(X)
y_t = torch.tensor(y)


def torch_forward(x):
    W1, b1, W2, b2, W3, b3 = torch_params
    x = torch.relu(x @ W1 + b1)
    x = torch.relu(x @ W2 + b2)
    return x @ W3 + b3


def torch_bce(logits, target):
    # same formula as nn.bce_loss (with the same eps)
    eps = 1e-7
    p = torch.sigmoid(logits)
    return -(target * torch.log(p + eps) + (1 - target) * torch.log(1 - p + eps)).mean()


max_loss_diff = 0.0
max_grad_diff = 0.0

for step in range(1, steps + 1):
    # minitorch
    loss = bce_loss(model.forward(Tensor(X)), Tensor(y))
    optimizer.zero_grad()
    loss.backward()

    # pytorch
    torch_loss = torch_bce(torch_forward(X_t), y_t)
    torch_opt.zero_grad()
    torch_loss.backward()

    # compare before updating the weights
    max_loss_diff = max(max_loss_diff, abs(loss.data.item() - torch_loss.item()))
    for p, tp in zip(model.parameters(), torch_params):
        max_grad_diff = max(max_grad_diff, np.abs(p.grad - tp.grad.numpy()).max())

    optimizer.step()
    torch_opt.step()

    if step % 50 == 0:
        print(f"step {step:3d}  minitorch loss {loss.data.item():.8f}  "
              f"pytorch loss {torch_loss.item():.8f}")

print(f"\nlargest loss difference over {steps} steps:     {max_loss_diff:.2e}")
print(f"largest gradient difference over {steps} steps: {max_grad_diff:.2e}")
