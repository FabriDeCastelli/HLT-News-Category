""" Model Abstract Class. """

import os
from abc import ABC, abstractmethod
from typing import Callable, List

import numpy as np
from sklearn.metrics import classification_report

from config import config
from config.config import RESULTS_DIRECTORY
from src.main.utilities import plotting


class Model(ABC):
    """
    Abstract Class for models.
    """

    @property
    @abstractmethod
    def pipeline(self):
        raise NotImplementedError()

    @pipeline.setter
    @abstractmethod
    def pipeline(self, pipeline: List[Callable]):
        raise NotImplementedError()

    @abstractmethod
    def run_pipeline(self, data):
        raise NotImplementedError()

    def summary(self):
        raise NotImplementedError()

    @abstractmethod
    def fit(self, inputs, targets):
        raise NotImplementedError()

    @abstractmethod
    def predict(self, inputs):
        raise NotImplementedError()

    @abstractmethod
    def save_model(self):
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def load_model(cls):
        raise NotImplementedError()

    def save_results(self, x_test, y_test):
        """
        Saves the result of the model predictions in config.RESULTS_DIRECTORY/{name of the class}.
        The confusion matrix and classification report are saved.

        :param x_test: The test input data.
        :param y_test: The test target data.
        """
        if y_test.ndim == 2:
            if not isinstance(y_test, np.ndarray):
                y_test = y_test.to_numpy()
            y_test = y_test.argmax(axis=1)
            y_test = np.vectorize(config.id2label.get)(y_test)
        y_pred = self.predict(x_test)
        report = classification_report(y_test, y_pred)
        directory = RESULTS_DIRECTORY.format(repr(self))
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, "metrics.txt")
        with open(path, "w") as file:
            file.write(report)
        plotting.plot_confusion_matrix(
            y_test, y_pred, path=os.path.join(directory, "confusion_matrix.png")
        )

    def __repr__(self):
        return self.__class__.__name__
