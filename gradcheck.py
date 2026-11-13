# check every backward() against a numerical gradient
#   numerical: (f(x + h) - f(x - h)) / 2h   for each element of x
# if the formulas in tensor.py are right, both numbers are (almost) equal.

import numpy as np
from tensor import Tensor
from nn import bce_loss

np.random.seed(0)
h = 1e-6


def check(name, f, *shapes):
    inputs = [Tensor(np.random.randn(*s), requires_grad=True) for s in shapes]

    # analytic gradient (our backward)
    out = f(*inputs)
    out.backward()

    worst = 0.0
    for t in inputs:
        num_grad = np.zeros_like(t.data)
        for i in np.ndindex(t.data.shape):
            old = t.data[i]
            t.data[i] = old + h
            plus = f(*inputs).data
            t.data[i] = old - h
            minus = f(*inputs).data
            t.data[i] = old
            num_grad[i] = (plus - minus) / (2 * h)

        diff = np.abs(num_grad - t.grad).max()
        worst = max(worst, diff)

    status = "OK  " if worst < 1e-5 else "FAIL"
    print(f"{status} {name:28s} max diff = {worst:.2e}")
    return worst < 1e-5


positive = lambda a: (a ** 2) + 0.5  # keep log/pow inputs positive

results = [
    check("add (with broadcasting)", lambda a, b: (a + b).sum(), (4, 3), (1, 3)),
    check("sub", lambda a, b: (a - b).sum(), (4, 3), (4, 3)),
    check("mul (with broadcasting)", lambda a, b: (a * b).sum(), (4, 3), (1, 3)),
    check("div", lambda a, b: (a / positive(b)).sum(), (4, 3), (4, 3)),
    check("pow", lambda a: (a ** 3).sum(), (4, 3)),
    check("matmul", lambda a, b: (a @ b).sum(), (4, 3), (3, 2)),
    check("relu", lambda a: (a.relu() * a).sum(), (4, 3)),
    check("sigmoid", lambda a: a.sigmoid().sum(), (4, 3)),
    check("log", lambda a: positive(a).log().sum(), (4, 3)),
    check("mean", lambda a: (a * a).mean(), (4, 3)),
    check("2-layer net + BCE loss",
          lambda x, W1, W2: bce_loss((x @ W1).relu() @ W2, Tensor((np.arange(5) % 2).reshape(5, 1))),
          (5, 3), (3, 4), (4, 1)),
]

print("\nall gradients correct" if all(results) else "\nsome gradients are wrong")
