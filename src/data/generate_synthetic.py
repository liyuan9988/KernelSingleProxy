import numpy as np
from scipy import special

from .data_class import SingleProxyTrainDataSet, SingleProxyTestDataSet


def generate_synthetic_train_data(n_data: int, noise_level: float = 0.0, freq_ratio: float = 1.0,
                                  high_freq_level: float = 0.0, seed: int = 42) -> SingleProxyTrainDataSet:
    rng = np.random.default_rng(seed=seed)
    U = rng.uniform(-1, 1, (n_data, 1))
    A = special.erf(U) + rng.normal(0, 0.1, (n_data, 1))
    Y = np.sin(np.pi * U * freq_ratio / 2.0) + high_freq_level * np.sin(10 * np.pi * U) + A ** 2 - 0.3 + rng.normal(0, noise_level, (n_data, 1))
    W = np.exp(U) + rng.normal(0, 0.1, (n_data, 1))
    return SingleProxyTrainDataSet(treatment=A, outcome=Y, proxy=W)


def generate_synthetic_test_data() -> SingleProxyTestDataSet:
    test_A = np.linspace(-1, 1, 100)[:, np.newaxis]
    truth = test_A ** 2 - 0.3
    return SingleProxyTestDataSet(treatment=test_A, structural=truth)
