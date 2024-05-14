import numpy as np

from sklearn.preprocessing import StandardScaler

from .data_class import SingleProxyTrainDataSet, SingleProxyTestDataSet


class AbstractPreprocessor:

    def preprocess_for_train(self, train_data: SingleProxyTrainDataSet, **kwarg) -> SingleProxyTrainDataSet:
        raise NotImplementedError

    def preprocess_for_test_input(self, test_data: SingleProxyTestDataSet) -> SingleProxyTestDataSet:
        raise NotImplementedError

    def postprocess_for_prediction(self, predict: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class IdentityPreprocessor(AbstractPreprocessor):

    def __init__(self):
        super(IdentityPreprocessor, self).__init__()

    def preprocess_for_train(self, train_data: SingleProxyTrainDataSet, **kwarg) -> SingleProxyTrainDataSet:
        return train_data

    def preprocess_for_test_input(self, test_data: SingleProxyTestDataSet) -> SingleProxyTestDataSet:
        return test_data

    def postprocess_for_prediction(self, predict: np.ndarray) -> np.ndarray:
        return predict


class ScaleAllPreprocessor(AbstractPreprocessor):
    treatment_scaler: StandardScaler
    outcome_scaler: StandardScaler

    def __init__(self):
        super(ScaleAllPreprocessor, self).__init__()

    def preprocess_for_train(self, train_data: SingleProxyTrainDataSet, **kwarg) -> SingleProxyTrainDataSet:
        proxy_scaler = StandardScaler()
        proxy_s = proxy_scaler.fit_transform(train_data.proxy)

        self.treatment_scaler = StandardScaler()
        treatment_s = self.treatment_scaler.fit_transform(train_data.treatment)

        self.outcome_scaler = StandardScaler()
        outcome_s = self.outcome_scaler.fit_transform(train_data.outcome)

        return SingleProxyTrainDataSet(treatment=treatment_s,
                                       proxy=proxy_s,
                                       outcome=outcome_s,
                                       )

    def preprocess_for_test_input(self, test_data: SingleProxyTestDataSet) -> SingleProxyTestDataSet:
        return SingleProxyTestDataSet(treatment=self.treatment_scaler.transform(test_data.treatment),
                                      structural=test_data.structural)

    def postprocess_for_prediction(self, predict: np.ndarray) -> np.ndarray:
        return self.outcome_scaler.inverse_transform(predict)


def get_preprocessor(id: str) -> AbstractPreprocessor:
    if id == "ScaleAll":
        return ScaleAllPreprocessor()
    elif id == "Identity":
        return IdentityPreprocessor()
    else:
        raise KeyError(f"{id} is invalid name for preprocessing")
