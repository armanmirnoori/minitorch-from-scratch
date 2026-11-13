import numpy as np


def make_moons(n=400, noise=0.15, seed=0):
    # two half circles that a straight line cannot separate
    # (same idea as sklearn.datasets.make_moons, written with numpy only)
    rng = np.random.default_rng(seed)
    half = n // 2

    angle = rng.uniform(0, np.pi, half)
    upper = np.stack([np.cos(angle), np.sin(angle)], axis=1)

    angle = rng.uniform(0, np.pi, n - half)
    lower = np.stack([1 - np.cos(angle), 0.5 - np.sin(angle)], axis=1)

    X = np.concatenate([upper, lower]) + rng.normal(0, noise, (n, 2))
    y = np.concatenate([np.zeros(half), np.ones(n - half)]).reshape(-1, 1)
    return X, y
