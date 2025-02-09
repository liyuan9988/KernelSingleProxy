import numpy as np
from typing import Tuple, Optional, Dict, Any
from pathlib import Path
import operator


from src.data.preprocess import get_preprocessor
from src.utils.kernel_func import GaussianKernel, AbsKernel
from src.data.data_class import SingleProxyTrainDataSet, SingleProxyTestDataSet, split_train_data
from src.data import generate_train_data, generate_test_data


def get_kernel_func(data_name: str) -> Tuple[AbsKernel, AbsKernel, AbsKernel]:
    if data_name in ("synthetic", "mnist", "step"):
        return GaussianKernel(), GaussianKernel(), GaussianKernel()
    else:
        raise ValueError(f"data name {data_name} is not valid")


class SKPVModel:
    treatment_kernel_func: AbsKernel
    proxy_kernel_func: AbsKernel
    outcome_kernel_func: AbsKernel

    alpha: np.ndarray
    proxy_mean_vec: np.ndarray
    train_treatment: np.ndarray
    train_outcome_proxy: np.ndarray

    def __init__(self, split_ratio: float, lam1=None, lam2=None, lam1_max=None, lam1_min=None,
                 n_lam1_search=None, lam2_max=None, lam2_min=None, n_lam2_search=None,
                 scale=1.0, use_old_form=False, **kwargs):
        self.lam1: Optional[float] = lam1
        self.lam2: Optional[float] = lam2
        self.lam1_max: Optional[float] = lam1_max
        self.lam1_min: Optional[float] = lam1_min
        self.n_lam1_search: Optional[int] = n_lam1_search
        self.lam2_max: Optional[float] = lam2_max
        self.lam2_min: Optional[float] = lam2_min
        self.n_lam2_search: Optional[int] = n_lam2_search
        self.scale: float = scale
        self.split_ratio: float = split_ratio
        self.use_old_form: float = use_old_form

    def tune_lam1(self, K1, KW1W1):
        lam1_candidate_list = np.logspace(np.log10(self.lam1_min), np.log10(self.lam1_max), self.n_lam1_search)
        grid_search = dict()
        n_data = K1.shape[0]
        for lam1_candi in lam1_candidate_list:
            H = np.eye(n_data) - np.linalg.solve(K1 + lam1_candi * n_data * np.eye(n_data), K1)
            H_tilde = np.diag(1 / np.diag(H))
            grid_search[lam1_candi] = np.trace(H_tilde @ H @ KW1W1 @ H @ H_tilde)

        self.lam1, loo = min(grid_search.items(), key=operator.itemgetter(1))


    def tune_lam2(self, M, train_y, test_k, test_y):
        n_train_2nd = M.shape[0]
        lam2_candidate_list = np.logspace(np.log10(self.lam2_min), np.log10(self.lam2_max), self.n_lam2_search)
        grid_search = dict()
        for lam2_candi in lam2_candidate_list:
            pred = train_y.T.dot(np.linalg.solve(M + n_train_2nd * lam2_candi * np.eye(n_train_2nd), test_k))
            grid_search[lam2_candi] = np.mean((pred - test_y) ** 2)

        self.lam2, loo = min(grid_search.items(), key=operator.itemgetter(1))

    def fit(self, train_data: SingleProxyTrainDataSet, data_name: str):
        train_data_1st, train_data_2nd = split_train_data(train_data, self.split_ratio)
        kernels = get_kernel_func(data_name)
        self.treatment_kernel_func = kernels[0]
        self.proxy_kernel_func = kernels[1]
        self.outcome_kernel_func = kernels[2]
        n_train_1st = train_data_1st.treatment.shape[0]
        n_train_2nd = train_data_2nd.treatment.shape[0]

        self.train_data_2nd = train_data_2nd

        # Set scales to be median
        self.treatment_kernel_func.fit(train_data.treatment, scale=self.scale)
        self.proxy_kernel_func.fit(train_data.proxy, scale=self.scale)
        self.outcome_kernel_func.fit(train_data.outcome, scale=self.scale)

        KA1A1 = self.treatment_kernel_func.cal_kernel_mat(train_data_1st.treatment, train_data_1st.treatment)
        KY1Y1 = self.outcome_kernel_func.cal_kernel_mat(train_data_1st.outcome, train_data_1st.outcome)
        KW1W1 = self.proxy_kernel_func.cal_kernel_mat(train_data_1st.proxy, train_data_1st.proxy)
        K = KA1A1 * KY1Y1

        KA1A2 = self.treatment_kernel_func.cal_kernel_mat(train_data_1st.treatment, train_data_2nd.treatment)
        KY1Y2 = self.outcome_kernel_func.cal_kernel_mat(train_data_1st.outcome, train_data_2nd.outcome)
        K2 = KA1A2 * KY1Y2

        KA2A2 = self.treatment_kernel_func.cal_kernel_mat(train_data_2nd.treatment, train_data_2nd.treatment)

        if self.lam1 is None:
            self.tune_lam1(K, KW1W1)

        B = np.linalg.solve(K + self.lam1 * n_train_1st * np.eye(n_train_1st), K2)
        B_tilde = np.linalg.solve(K + self.lam1 * n_train_1st * np.eye(n_train_1st), K)
        M = KA2A2 * (B.T @ KW1W1 @ B)
        M_tilde = KA1A2.T * (B_tilde.T @ KW1W1)
        self.w = B.T @ np.mean(KW1W1, axis=1, keepdims=True)
        if self.lam2 is None:
            self.tune_lam2(M, M_tilde, train_data_2nd.outcome, train_data_1st.outcome)


        if self.use_old_form:
            print("use_old_form")
            self.alpha = np.linalg.solve(M @ M + self.lam2 * n_train_2nd * M, M @ train_data_2nd.outcome)
        else:
            self.alpha = np.linalg.solve(M + self.lam2 * n_train_2nd * np.eye(n_train_2nd), train_data_2nd.outcome)

        print((self.lam1, self.lam2))

    def predict(self, treatment: np.ndarray) -> np.ndarray:

        test_kernel = self.treatment_kernel_func.cal_kernel_mat(self.train_data_2nd.treatment, treatment)
        k = test_kernel * self.w
        pred = k.T.dot(self.alpha)
        return pred

    def evaluate(self, test_data: SingleProxyTestDataSet):
        pred = self.predict(treatment=test_data.treatment)
        return np.mean((pred - test_data.structural) ** 2)


def skpv_experiments(data_config: Dict[str, Any], model_param: Dict[str, Any],
                     one_mdl_dump_dir: Path,
                     random_seed: int = 42, verbose: int = 0):
    train_data_org = generate_train_data(data_config=data_config, rand_seed=random_seed)
    test_data_org = generate_test_data(data_config=data_config)

    preprocessor = get_preprocessor(data_config.get("preprocess", "Identity"))
    train_data = preprocessor.preprocess_for_train(train_data_org)
    test_data = preprocessor.preprocess_for_test_input(test_data_org)

    model = SKPVModel(**model_param)
    model.fit(train_data, data_config["name"])
    pred = model.predict(test_data.treatment)
    pred = preprocessor.postprocess_for_prediction(pred)

    np.savetxt(one_mdl_dump_dir.joinpath(f"{random_seed}.pred.txt"), pred)
    if test_data.structural is not None:
        return np.mean((pred - test_data.structural) ** 2)
    else:
        return 0.0
