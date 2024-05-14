

from .nn_structure_for_synthetic import SyntheticDistribution
from .nn_structure_for_mnist import MNISTDistribution

import logging

logger = logging.getLogger()



def build_extractor(data_name: str, hidden_dim, n_sample):
    if data_name == "synthetic":
        return SyntheticDistribution(n_hidden_dim=hidden_dim, n_learning_sample=n_sample)
    elif data_name == "mnist":
        return MNISTDistribution(n_hidden_dim=hidden_dim, n_learning_sample=n_sample)
    else:
        raise ValueError(f"data name {data_name} is not valid")
