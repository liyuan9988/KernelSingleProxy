import numpy as np
from typing import Tuple, Optional, Dict, Any
from pathlib import Path
import operator

from src.data.preprocess import get_preprocessor
from src.utils.kernel_func import GaussianKernel, AbsKernel
from src.data.data_class import SingleProxyTrainDataSet, SingleProxyTestDataSet, split_train_data
from src.data import generate_train_data, generate_test_data


def get_kernel_func(data_name: str) -> Tuple[AbsKernel, AbsKernel, AbsKernel]:
    if data_name in ("synthetic", "mnist"):
        return GaussianKernel(), GaussianKernel(), GaussianKernel()
    else:
        raise ValueError(f"data name {data_name} is not valid")


class SPMMRModel:
    treatment_kernel_func: AbsKernel
    proxy_kernel_func: AbsKernel
    outcome_kernel_func: AbsKernel

    alpha: np.ndarray
    proxy_mean_vec: np.ndarray
    train_treatment: np.ndarray
    train_outcome_proxy: np.ndarray

    def __init__(self, val_ratio: float = 0.1, lam=None, lam_max=None, lam_min=None,
                 n_lam_search=None, scale=1.0, **kwargs):
        self.lam: Optional[float] = lam
        self.lam_max: Optional[float] = lam_max
        self.lam_min: Optional[float] = lam_min
        self.n_lam_search: Optional[int] = n_lam_search
        self.val_ratio = val_ratio
        self.scale: float = scale

    def tune_lam(self, L, sqG, val_data):
        lam1_candidate_list = np.logspace(np.log10(self.lam_min), np.log10(self.lam_max), self.n_lam_search)
        grid_search = dict()
        n_data = self.train_data.outcome.shape[0]
        K = sqG @ L @ sqG
        ky = sqG @ self.train_data.outcome
        k_test = self.treatment_kernel_func.cal_kernel_mat(self.train_data.treatment, val_data.treatment) \
                 * self.proxy_kernel_func.cal_kernel_mat(self.train_data.proxy, val_data.proxy)
        W = self.treatment_kernel_func.cal_kernel_mat(val_data.treatment, val_data.treatment) \
            * self.outcome_kernel_func.cal_kernel_mat(val_data.outcome, val_data.outcome)
        for lam1_candi in lam1_candidate_list:
            alpha = sqG @ np.linalg.solve(K + lam1_candi * n_data * n_data * np.eye(n_data), ky)
            pred = k_test.T.dot(alpha)
            error = val_data.outcome - pred
            score = error.T.dot(W.dot(error)) + lam1_candi * n_data * n_data * alpha.T.dot(L.dot(alpha))
            grid_search[lam1_candi] = score

        self.lam, loo = min(grid_search.items(), key=operator.itemgetter(1))
        print(self.lam)

    def fit(self, train_data: SingleProxyTrainDataSet, data_name: str):
        kernels = get_kernel_func(data_name)
        self.treatment_kernel_func = kernels[0]
        self.proxy_kernel_func = kernels[1]
        self.outcome_kernel_func = kernels[2]

        # Set scales to be median
        self.treatment_kernel_func.fit(train_data.treatment, scale=self.scale)
        self.proxy_kernel_func.fit(train_data.proxy, scale=self.scale)
        self.outcome_kernel_func.fit(train_data.outcome, scale=self.scale)

        val_data = None
        if self.val_ratio > 0:
            val_data, train_data = split_train_data(train_data, self.val_ratio)

        self.train_data = train_data
        n_data = train_data.treatment.shape[0]

        KAA = self.treatment_kernel_func.cal_kernel_mat(train_data.treatment, train_data.treatment)
        KYY = self.outcome_kernel_func.cal_kernel_mat(train_data.outcome, train_data.outcome)
        KWW = self.proxy_kernel_func.cal_kernel_mat(train_data.proxy, train_data.proxy)

        G = KAA * KYY
        L = KAA * KWW
        D, V = np.linalg.eigh(G)
        sqG = V @ np.diag(np.sqrt(np.maximum(D, 0.0))) @ V.T

        if self.lam is None:
            assert val_data is not None
            self.tune_lam(L, sqG, val_data)

        self.alpha = sqG @ np.linalg.solve(sqG @ L @ sqG + self.lam * n_data * n_data * np.eye(n_data),
                                           sqG @ train_data.outcome)

        self.proxy_mean_vec = np.mean(KWW, axis=1, keepdims=True)

    def predict(self, treatment: np.ndarray) -> np.ndarray:

        test_kernel = self.treatment_kernel_func.cal_kernel_mat(self.train_data.treatment, treatment)
        k = test_kernel * self.proxy_mean_vec
        pred = k.T.dot(self.alpha)
        return pred

    def evaluate(self, test_data: SingleProxyTestDataSet):
        pred = self.predict(treatment=test_data.treatment)
        return np.mean((pred - test_data.structural) ** 2)


def spmmr_experiments(data_config: Dict[str, Any], model_param: Dict[str, Any],
                      one_mdl_dump_dir: Path,
                      random_seed: int = 42, verbose: int = 0):
    train_data_org = generate_train_data(data_config=data_config, rand_seed=random_seed)
    test_data_org = generate_test_data(data_config=data_config)

    preprocessor = get_preprocessor(data_config.get("preprocess", "Identity"))
    train_data = preprocessor.preprocess_for_train(train_data_org)
    test_data = preprocessor.preprocess_for_test_input(test_data_org)

    model = SPMMRModel(**model_param)
    model.fit(train_data, data_config["name"])
    pred = model.predict(test_data.treatment)
    pred = preprocessor.postprocess_for_prediction(pred)

    np.savetxt(one_mdl_dump_dir.joinpath(f"{random_seed}.pred.txt"), pred)
    if test_data.structural is not None:
        return np.mean((pred - test_data.structural) ** 2)
    else:
        return 0.0
