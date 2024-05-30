""" Multinomial Naive Bayes Classifier """

import numpy as np
from src.main.models.model import Model
from src.main.pipeline.pipeline import Pipeline
from config.config import MODELS_PATH

from typing import Callable, List
from sklearn.naive_bayes import MultinomialNB

import joblib
import pandas as pd
import os


class Naivebayes(Model):
    """
    Multinomial Naive Bayes class.
    """

    def __init__(self, model=None, **kwargs):
        """
        Constructor for the Multinomial Naive Bayes class.
        Instantiates the NB model by creating a sklearn MultinomialNB object, see the sklearn
        documentation at
        https://scikit-learn.org/stable/modules/naive_bayes.html#multinomial-naive-bayes

        :param model: the MultinomialNB model, if provided in input **kwargs are ignored
        :param kwargs: the arguments that are going to be passed to the MultinomialNB model.
        """
        if model is not None:
            self._naivebayes = model
        else:
            self._naivebayes = MultinomialNB(**kwargs)
        self._pipeline = None

    @property
    def naivebayes(self):
        return self._naivebayes

    @naivebayes.setter
    def naivebayes(self, model):
        if not isinstance(model, MultinomialNB):
            raise ValueError(
                "The model should be an instance of sklearn.linear_model.MultinomialNB"
            )
        self._naivebayes = model

    @property
    def pipeline(self):
        return self._pipeline

    @pipeline.setter
    def pipeline(self, pipeline: List[Callable]):
        self._pipeline = Pipeline(pipeline)

    def run_pipeline(self, data: pd.DataFrame, save=True):
        """
        Run the pipeline. If the pipeline for this model has already been run, then the dataset is read from the file.
        Since the pipeline is returned in a common format, this has to be adapted to this model.

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
            self._naivebayes = self._naivebayes.fit(inputs, targets, sample_weight)

    def evaluate(self, inputs, targets):
        return self._naivebayes.score(inputs, targets)

    def predict(self, data):
        return self._naivebayes.predict(data)

    def save_model(self):
        path = os.path.join(MODELS_PATH, repr(self) + ".pkl")
        joblib.dump(self._naivebayes, path)

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
