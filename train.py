import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # save the plot to a file, no window
import matplotlib.pyplot as plt

from tensor import Tensor
from nn import MLP, bce_loss
from optim import SGD
from data import make_moons

np.random.seed(0)

# 1) data: 80% train, 20% test
X, y = make_moons(n=500, noise=0.15)
idx = np.random.permutation(len(X))
train_idx, test_idx = idx[:400], idx[400:]
X_train, y_train = Tensor(X[train_idx]), Tensor(y[train_idx])
X_test, y_test = X[test_idx], y[test_idx]

# 2) model + optimizer
model = MLP([2, 16, 16, 1])
optimizer = SGD(model.parameters(), lr=0.5)


def accuracy(X_np, y_np):
    logits = model.forward(Tensor(X_np)).data
    pred = (logits > 0).astype(float)  # logit > 0  <=>  sigmoid > 0.5
    return (pred == y_np).mean()


# 3) training loop (full batch gradient descent)
losses = []
for epoch in range(1, 1001):
    logits = model.forward(X_train)
    loss = bce_loss(logits, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.data.item())
    if epoch % 100 == 0:
        print(f"epoch {epoch:4d}  loss {loss.data.item():.4f}  "
              f"train acc {accuracy(X[train_idx], y[train_idx]):.3f}  "
              f"test acc {accuracy(X_test, y_test):.3f}")

# 4) plots: loss curve + decision boundary
os.makedirs("results", exist_ok=True)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))

ax[0].plot(losses)
ax[0].set_title("training loss (BCE)")
ax[0].set_xlabel("epoch")

xx, yy = np.meshgrid(np.linspace(-1.5, 2.5, 200), np.linspace(-1, 1.5, 200))
grid = np.stack([xx.ravel(), yy.ravel()], axis=1)
zz = model.forward(Tensor(grid)).sigmoid().data.reshape(xx.shape)
ax[1].contourf(xx, yy, zz, levels=20, cmap="RdBu", alpha=0.6)
ax[1].scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap="RdBu", edgecolors="k", s=15)
ax[1].set_title("decision boundary")

plt.tight_layout()
plt.savefig("results/moons.png", dpi=120)
print("saved results/moons.png")
