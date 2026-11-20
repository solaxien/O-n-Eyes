"""Kernel Inception Distance (unbiased MMD with a cubic polynomial kernel).

KID is the primary distribution metric for modest sample counts because its
estimator is unbiased; FID at a few hundred images is biased upwards.
"""

import numpy as np


def _mmd2_unbiased(x: np.ndarray, y: np.ndarray) -> float:
    dim = x.shape[1]
    kxx = (x @ x.T / dim + 1) ** 3
    kyy = (y @ y.T / dim + 1) ** 3
    kxy = (x @ y.T / dim + 1) ** 3
    m, n = len(x), len(y)
    return float(
        (kxx.sum() - np.trace(kxx)) / (m * (m - 1))
        + (kyy.sum() - np.trace(kyy)) / (n * (n - 1))
        - 2 * kxy.mean()
    )


def kernel_inception_distance(first: np.ndarray, second: np.ndarray, *, subsets: int = 100,
                              subset_size: int = 1000, seed: int = 0) -> tuple[float, float]:
    """Return (mean, std) of KID over random subsets, as in torch-fidelity."""
    first, second = np.asarray(first, np.float64), np.asarray(second, np.float64)
    size = min(subset_size, len(first), len(second))
    if size < 2:
        raise ValueError("KID needs at least two samples per set")
    rng = np.random.default_rng(seed)
    values = [
        _mmd2_unbiased(first[rng.choice(len(first), size, replace=False)],
                       second[rng.choice(len(second), size, replace=False)])
        for _ in range(subsets)
    ]
    return float(np.mean(values)), float(np.std(values))
