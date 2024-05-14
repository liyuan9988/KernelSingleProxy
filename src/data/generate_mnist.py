from typing import Tuple, Sequence
import numpy as np
from numpy.random import default_rng
from torchvision.datasets import MNIST
from torchvision import transforms
from filelock import FileLock
from scipy import special

from .data_class import SingleProxyTrainDataSet, SingleProxyTestDataSet

def attach_image(num_array: Sequence[int], train_flg: bool, seed: int = 42) -> np.ndarray:
    """
    Randomly samples number from MNIST datasets

    Parameters
    ----------
    num_array : Array[int]
        Array of numbers that we sample for
    train_flg : bool
        Whether to sample from train/test in MNIST dataset
    seed : int
        Random seed
    Returns
    -------
    result : (len(num_array), 28*28) array
        ndarray for sampled images
    """
    rng = default_rng(seed)
    with FileLock("./data.lock"):
        mnist = MNIST("./data/", train=train_flg, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]), target_transform=None, download=True)
    img_data = mnist.data.numpy()
    img_target = mnist.targets.numpy()

    def select_one(idx):
        sub = img_data[img_target == idx]
        return sub[rng.choice(sub.shape[0])].reshape((1, -1))

    return np.concatenate([select_one(num) for num in num_array], axis=0)



def generate_mnist_train_data(n_data: int, noise_level: float = 0.0, seed: int = 42) -> SingleProxyTrainDataSet:
    rng = np.random.default_rng(seed=seed)
    U = rng.uniform(-1, 1, (n_data, 1))
    W = attach_image(np.floor(U*5 + 5).astype(int), True, seed) + rng.normal(0, 0.1, (n_data, 784))

    A = special.erf(U) + rng.normal(0, 0.1, (n_data, 1))
    Y = np.sin(np.pi * U / 2.0) + A ** 2 - 0.3 + rng.normal(0, noise_level, (n_data, 1))
    return SingleProxyTrainDataSet(treatment=A, outcome=Y, proxy=W)


def generate_mnist_test_data() -> SingleProxyTestDataSet:
    test_A = np.linspace(-1, 1, 100)[:, np.newaxis]
    truth = test_A ** 2 - 0.3
    return SingleProxyTestDataSet(treatment=test_A, structural=truth)
