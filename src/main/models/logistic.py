""" Multinomial Logistic Regression. """

from src.main.models.model import Model
from src.main.pipeline.pipeline import Pipeline
from config.config import MODELS_PATH, HYPERPARAMETERS_PATH
from src.main.utilities.utils import read_yaml

import os
from typing import Callable, List

import pandas as pd
import numpy as np
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV


class Logistic(Model):
    """
    Multinomial Logistic Regression class.
    """

    def __init__(self, model=None, **kwargs):
        """
        Constructor for the Multinomial Logistic Regression class.
        Instantiates the Logistic Regression model by creating a sklearn LogisticRegression object, see the sklearn
        documentation at
        https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

        :param model: the Logistic Regression model, if provided in input **kwargs are ignored
        :param kwargs: the arguments that are going to be passed to the Logistic Regression model.
        """
        if model is not None:
            self._logistic = model
        else:
            self._logistic = LogisticRegression(**kwargs)
        self._pipeline = None
        self.hyperparameters = read_yaml(HYPERPARAMETERS_PATH.format(repr(self)))

    @property
    def logistic(self):
        return self._logistic

    @logistic.setter
    def logistic(self, model):
        if not isinstance(model, LogisticRegression):
            raise ValueError(
                "The model should be an instance of sklearn.linear_model.LogisticRegression"
            )
        self._logistic = model

    @property
    def pipeline(self):
        return self._pipeline

    @pipeline.setter
    def pipeline(self, pipeline: List[Callable]):
        self._pipeline = Pipeline(pipeline)

    def run_pipeline(self, data: pd.DataFrame, save=True):
        """
        Run the pipeline. If the pipeline for this model has already been run, then the dataset is read from the file.
        Since in the pipline class the return format is universal, it has to be adapted dynamically.

        :param data: the data to run the pipeline on.
        :param save: a boolean indicating whether to save the data to a file.
        :return: the data after the processing.
        """
        assert self.pipeline is not None, "Cannot run the pipeline: it is not set."
        result = self.pipeline.execute(data, model_file=repr(self) + ".npz", save=save)
        if isinstance(result, np.ndarray) and result.shape == ():
            return result.item()
        return result

    def fit(self, inputs, targets, sample_weight=None):
        """
        Fit the model to the data.

        :param inputs: the training data, excluding the targets
        :param targets: the target values
        :param sample_weight: the weights of the samples
        """
        if os.path.isfile(os.path.join(MODELS_PATH, repr(self) + ".pkl")):
            self.load_model()
        else:
            self.logistic = self.logistic.fit(inputs, targets, sample_weight)

    def grid_search(self, x_train, y_train, n_iter=30):
        """
        Performs a random grid search. Best model is chosen based on the F1 score.

        :param x_train: the training data
        :param y_train: the target values
        :param n_iter: the number of iterations to run the Randomized Search
        :return: the cross validation results
        """

        # Randomized Search
        randomized_search = RandomizedSearchCV(
            estimator=self.logistic,
            param_distributions=self.hyperparameters,
            scoring="f1_macro",
            n_jobs=-1,
            n_iter=n_iter,
            random_state=42,
            verbose=True,
        )

        result = randomized_search.fit(x_train, y_train)
        self.logistic = result.best_estimator_
        return result

    def evaluate(self, inputs, targets):
        return self.logistic.score(inputs, targets)

    def predict(self, data):
        return self.logistic.predict(data)

    def save_model(self):
        path = os.path.join(MODELS_PATH, repr(self) + ".pkl")
        joblib.dump(self.logistic, path)

    @classmethod
    def load_model(cls):
        """
        Load the model from a file in a pkl format.
        """
        path = os.path.join(MODELS_PATH, repr(cls) + ".pkl")
        assert os.path.isfile(
            path
        ), f"Error: trying to load {repr(cls)} model at unknown path {path}"
        return cls(model=joblib.load(path))

    def __repr__(self):
        return self.__class__.__name__
