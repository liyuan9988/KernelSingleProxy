from typing import Dict, Any, Optional

from .data_class import SingleProxyTestDataSet, SingleProxyTrainDataSet
from .generate_synthetic import generate_synthetic_train_data, generate_synthetic_test_data
from .generate_mnist import generate_mnist_test_data, generate_mnist_train_data


def generate_train_data(data_config: Dict[str, Any], rand_seed: int) -> SingleProxyTrainDataSet:
    data_name = data_config["name"]
    if data_name == "synthetic":
        return generate_synthetic_train_data(seed=rand_seed,
                                             n_data=data_config["n_data"],
                                             high_freq_level=data_config.get("high_freq_level", 0.0),
                                             noise_level=data_config["noise_level"])
    elif data_name == "mnist":
        return generate_mnist_train_data(seed=rand_seed,
                                         n_data=data_config["n_data"],
                                         noise_level=data_config["noise_level"])

    else:
        raise ValueError(f"data name {data_name} is not valid")


def generate_test_data(data_config: Dict[str, Any]) -> Optional[SingleProxyTestDataSet]:
    data_name = data_config["name"]
    if data_name == "synthetic":
        return generate_synthetic_test_data()
    elif data_name == "mnist":
        return generate_mnist_test_data()
    else:
        raise ValueError(f"data name {data_name} is not valid")
