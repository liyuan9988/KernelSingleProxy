from typing import List
import numpy as np
from scipy.spatial.distance import cdist


class AbsKernel:
    def fit(self, data: np.ndarray, **kwargs) -> None:
        raise NotImplementedError

    def cal_kernel_mat(self, data1: np.ndarray, data2: np.ndarray) -> np.ndarray:
        raise NotImplementedError

class LinearDotKernel(AbsKernel):
    def __init__(self):
        super(LinearDotKernel, self).__init__()

    def fit(self, data: np.ndarray, **kwargs) -> None:
        pass

    def cal_kernel_mat(self, data1: np.ndarray, data2: np.ndarray) -> np.ndarray:
        return data1.dot(data2.T)

class GaussianKernel(AbsKernel):
    sigma: np.float32

    def __init__(self):
        super(GaussianKernel, self).__init__()

    def fit(self, data: np.ndarray, scale=1.0, **kwargs) -> None:
        dists = cdist(data, data, 'sqeuclidean')
        self.sigma = np.median(dists) * scale

    def cal_kernel_mat(self, data1: np.ndarray, data2: np.ndarray) -> np.ndarray:
        dists = cdist(data1, data2, 'sqeuclidean')
        dists = dists / self.sigma
        return np.exp(-dists)


class ColumnWiseGaussianKernel(AbsKernel):
    kernel_list: List[GaussianKernel]

    def __init__(self):
        super(ColumnWiseGaussianKernel, self).__init__()
        self.kernel_list = []

    def fit(self, data: np.ndarray, scale=1.0, **kwargs) -> None:
        for i in range(data.shape[1]):
            self.kernel_list.append(GaussianKernel())
            self.kernel_list[-1].fit(data[:, [i]], scale=scale)

    def cal_kernel_mat(self, data1: np.ndarray, data2: np.ndarray) -> np.ndarray:
        assert self.kernel_list is not None
        res = []
        for i in range(len(self.kernel_list)):
            res.append(self.kernel_list[i].cal_kernel_mat(data1[:, [i]], data2[:, [i]])[:,:,np.newaxis])
        res_mat = np.concatenate(res, axis=2)
        return np.product(res_mat, axis=2)
