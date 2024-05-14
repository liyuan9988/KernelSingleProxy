from typing import NamedTuple, Optional, Tuple, Dict
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class SingleProxyTrainDataSet(NamedTuple):
    proxy: np.ndarray
    treatment: np.ndarray
    outcome: np.ndarray


class SingleProxyTestDataSet(NamedTuple):
    treatment: np.ndarray
    structural: np.ndarray


class SingleProxyTrainDataSetTorch(NamedTuple):
    proxy: torch.Tensor
    treatment: torch.Tensor
    outcome: torch.Tensor

    @classmethod
    def from_numpy(cls, train_data: SingleProxyTrainDataSet):
        return SingleProxyTrainDataSetTorch(proxy=torch.tensor(train_data.proxy, dtype=torch.float32),
                                            treatment=torch.tensor(train_data.treatment, dtype=torch.float32),
                                            outcome=torch.tensor(train_data.outcome, dtype=torch.float32))


class SingleProxyTestDataSetTorch(NamedTuple):
    treatment: torch.Tensor
    structural: torch.Tensor

    @classmethod
    def from_numpy(cls, test_data: SingleProxyTestDataSet):
        return SingleProxyTestDataSetTorch(treatment=torch.tensor(test_data.treatment, dtype=torch.float32),
                                           structural=torch.tensor(test_data.structural, dtype=torch.float32))


def split_train_data(train_data: SingleProxyTrainDataSet, split_ratio=0.5):
    if split_ratio < 0.0:
        return train_data, train_data

    n_data = train_data[0].shape[0]
    idx_train_1st, idx_train_2nd = train_test_split(np.arange(n_data), train_size=split_ratio)

    def get_data(data, idx):
        return data[idx] if data is not None else None

    train_1st_data = SingleProxyTrainDataSet(*[get_data(data, idx_train_1st) for data in train_data])
    train_2nd_data = SingleProxyTrainDataSet(*[get_data(data, idx_train_2nd) for data in train_data])
    return train_1st_data, train_2nd_data


def standardise(data: SingleProxyTrainDataSet) -> Tuple[SingleProxyTrainDataSet, Dict[str, StandardScaler]]:
    proxy_scaler = StandardScaler()
    proxy_s = proxy_scaler.fit_transform(data.proxy)

    treatment_scaler = StandardScaler()
    treatment_s = treatment_scaler.fit_transform(data.treatment)

    outcome_scaler = StandardScaler()
    outcome_s = outcome_scaler.fit_transform(data.outcome)

    train_data = SingleProxyTrainDataSet(treatment=treatment_s,
                                         proxy=proxy_s,
                                         outcome=outcome_s)

    scalers = dict(proxy_scaler=proxy_scaler,
                   treatment_scaler=treatment_scaler,
                   outcome_scaler=outcome_scaler)

    return train_data, scalers
