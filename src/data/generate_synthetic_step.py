import numpy as np
from scipy import special

from .data_class import SingleProxyTrainDataSet, SingleProxyTestDataSet


def cal_ate(A: float, n_steps, rng):
    U = rng.uniform(-1, 1, (1000, 1))
    steps = np.linspace(-1, 1, n_steps)
    Y = np.mean(np.sin(steps[np.digitize(U, steps) - 1] / 2 * np.pi)) + A ** 2 - 0.3
    return Y


def generate_synthetic_train_data_with_step(n_data: int, noise_level: float = 0.0,
                                            n_steps: int = 10, seed: int = 42) -> SingleProxyTrainDataSet:
    rng = np.random.default_rng(seed=seed)
    U = rng.uniform(-1, 1, (n_data, 1))
    A = special.erf(U) + rng.normal(0, 0.1, (n_data, 1))
    steps = np.linspace(-1, 1, n_steps)
    Y = np.sin(steps[np.digitize(U, steps) - 1] / 2 * np.pi) + A ** 2 - 0.3 + rng.normal(0, noise_level, (n_data, 1))
    W = np.exp(U) + rng.normal(0, 0.1, (n_data, 1))
    return SingleProxyTrainDataSet(treatment=A, outcome=Y, proxy=W)


def generate_synthetic_test_data_with_step(n_steps: int = 10) -> SingleProxyTestDataSet:
    test_A = np.linspace(-1, 1, 100)[:, np.newaxis]
    rng = np.random.default_rng(seed=42)
    truth = np.array([cal_ate(A, n_steps, rng) for A in test_A])
    return SingleProxyTestDataSet(treatment=test_A, structural=truth)
