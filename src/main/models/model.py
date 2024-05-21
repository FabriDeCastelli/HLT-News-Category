""" Model Abstract Class. """

import inspect
import os
from abc import ABC, abstractmethod
from typing import Callable, List

import pandas as pd
from sklearn.metrics import classification_report

from config.config import RESULTS_DIRECTORY
from main.utilities import plotting


class Model(ABC):
    """
    Abstract Class for models.
    """

    @property
    @abstractmethod
    def pipeline(self):
        """
        Getter of the pipeline of the model.
        """
        raise NotImplementedError()

    @pipeline.setter
    @abstractmethod
    def pipeline(self, pipeline: List[Callable]):
        """
        Setter of the pipeline.
        """
        raise NotImplementedError()

    @abstractmethod
    def run_pipeline(self, data):
        """
        Run the pipeline on the data.

        :param data: The data to run the pipeline on.
        """
        assert isinstance(data, pd.DataFrame), "Data is not a pandas DataFrame."

    def summary(self):
        """
        Print a summary of the model.
        """
        raise NotImplementedError()

    @abstractmethod
    def fit(self, inputs, targets):
        """
        Fit the model to the data.

        :param inputs: The input data.
        :param targets: The target data.
        """
        raise NotImplementedError()

    @abstractmethod
    def predict(self, inputs):
        """
        Predict the target variable for the input data.

        :param inputs: The input data.
        """
        raise NotImplementedError()

    @abstractmethod
    def save_model(self):
        """
        Save the model.
        """

    @classmethod
    @abstractmethod
    def load_model(cls):
        """
        Load the model.
        """
        raise NotImplementedError()

    def save_results(self, x_test, y_test):
        """
        Save the results.
        """
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
