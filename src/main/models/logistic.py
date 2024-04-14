""" Multinomial Logistic Regression. """

import os
from typing import Callable, List

import pandas as pd

from src.main.models.model import Model
from src.main.pipeline.pipeline import Pipeline
from config.config import PIPELINE_DATASET_PATH
from sklearn.linear_model import LogisticRegression


class Logistic(Model):
    """
    Multinomial Logistic Regression class.
    """

    def __init__(self, **kwargs):
        """
        Constructor for the Multinomial Logistic Regression class.
        Instantiates the Logistic Regression model by creating a sklearn LogisticRegression object, see the sklearn
        documentation at
        https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

        :param kwargs: the arguments that are going to be passed to the Logistic Regression model.

        """
        super().__init__()
        self.logistic = LogisticRegression(**kwargs)
        self.pipeline = None

    def set_pipeline(self, pipeline: List[Callable]):
        """
        Set the pipeline for the model.

        :param pipeline: an array of functions that are going to be executed in the pipeline.
        :return:
        """
        self.pipeline = Pipeline(pipeline)

    def run_pipeline(self, data: pd.DataFrame):
        """
        Run the pipeline. If the pipeline for this model has already been run, then the dataset is read from the file.

        :param data: the data to run the pipeline on.
        :return: the data after the processing.
        """
        assert self.pipeline is not None, "Pipeline is not set."
        assert isinstance(data, pd.DataFrame), "Data is not a pandas DataFrame."

        path = PIPELINE_DATASET_PATH.format(repr(self))
        if os.path.exists(path):
            return pd.read_csv(path)
        return self.pipeline.execute(data, model=repr(self), save=True)

    def fit(self, inputs, targets, sample_weight=None):
        """
        Fit the model to the data.

        :param inputs: the training data, excluding the targets
        :param targets: the target values
        :param sample_weight: the weights of the samples
        """
        self.logistic.fit(inputs, targets, sample_weight)

    def predict(self, inputs):
        raise NotImplementedError()

    def __repr__(self):
        return "Logistic"
