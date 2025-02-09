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


class RegressionModel:
    treatment_kernel_func: AbsKernel
    proxy_kernel_func: AbsKernel
    outcome_kernel_func: AbsKernel

    alpha: np.ndarray
    train_treatment: np.ndarray

    def __init__(self, lam=None, scale=1.0, **kwargs):
        self.lam: Optional[float] = lam
        self.scale = scale

    def fit(self, train_data: SingleProxyTrainDataSet, data_name: str):
        kernels = get_kernel_func(data_name)
        self.treatment_kernel_func = kernels[0]
        self.proxy_kernel_func = kernels[1]
        self.outcome_kernel_func = kernels[2]

        # Set scales to be median
        self.treatment_kernel_func.fit(train_data.treatment, scale=self.scale)
        self.proxy_kernel_func.fit(train_data.proxy, scale=self.scale)
        self.outcome_kernel_func.fit(train_data.outcome, scale=self.scale)

        n_train = train_data.proxy.shape[0]
        self.train_treatment = train_data.treatment
        KAA = self.treatment_kernel_func.cal_kernel_mat(train_data.treatment, train_data.treatment)
        self.alpha = np.linalg.solve(KAA + self.lam * n_train * np.eye(n_train), train_data.outcome)

    def predict(self, treatment: np.ndarray) -> np.ndarray:

        test_kernel = self.treatment_kernel_func.cal_kernel_mat(self.train_treatment, treatment)
        pred = test_kernel.T.dot(self.alpha)
        return pred

    def evaluate(self, test_data: SingleProxyTestDataSet):
        pred = self.predict(treatment=test_data.treatment)
        return np.mean((pred - test_data.structural) ** 2)


def regression_experiments(data_config: Dict[str, Any], model_param: Dict[str, Any],
                     one_mdl_dump_dir: Path,
                     random_seed: int = 42, verbose: int = 0):
    train_data_org = generate_train_data(data_config=data_config, rand_seed=random_seed)
    test_data_org = generate_test_data(data_config=data_config)

    preprocessor = get_preprocessor(data_config.get("preprocess", "Identity"))
    train_data = preprocessor.preprocess_for_train(train_data_org)
    test_data = preprocessor.preprocess_for_test_input(test_data_org)

    model = RegressionModel(**model_param)
    model.fit(train_data, data_config["name"])
    pred = model.predict(test_data.treatment)
    pred = preprocessor.postprocess_for_prediction(pred)

    np.savetxt(one_mdl_dump_dir.joinpath(f"{random_seed}.pred.txt"), pred)
    if test_data.structural is not None:
        return np.mean((pred - test_data.structural) ** 2)
    else:
        return 0.0
