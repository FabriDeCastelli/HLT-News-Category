""" Model Abstract Class. """

from abc import ABC, abstractmethod
from typing import Callable, List

import pandas as pd


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

    def __repr__(self):
        return self.__class__.__name__
