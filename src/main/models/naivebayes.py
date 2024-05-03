""" Multinomial Naive Bayes Classifier """

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
        :param kwargs: the arguments that are going to be passed to the Logistic Regression model.
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
        """
        Set the pipeline for the model.

        :param pipeline: an array of functions that are going to be executed in the pipeline.
        """
        self._pipeline = Pipeline(pipeline)

    def run_pipeline(self, data: pd.DataFrame, save=True):
        """
        Run the pipeline. If the pipeline for this model has already been run, then the dataset is read from the file.

        :param data: the data to run the pipeline on.
        :param save: a boolean indicating whether to save the data to a file.
        :return: the data after the processing.
        """
        assert self.pipeline is not None, "Cannot run the pipeline: it is not set."
        return self.pipeline.execute(data, model_file=repr(self) + ".npz", save=save)

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
        """
        Evaluate the model.

        :param inputs: the data to evaluate the model on
        :param targets: the target values
        :return: the score of the model
        """
        return self._naivebayes.score(inputs, targets)

    def predict(self, data):
        """
        Make prediction over data.

        :param data: the data to predict
        :return: the predicted values
        """
        return self._naivebayes.predict(data)

    def save_model(self):
        """
        Save the model to a file.
        """
        path = os.path.join(MODELS_PATH, repr(self) + ".pkl")
        os.mkdir(path)
        joblib.dump(self._naivebayes, path)

    @classmethod
    def load_model(cls):
        """
        Load the model from a file.
        """
        path = os.path.join(MODELS_PATH, repr(cls) + ".pkl")
        assert os.path.isfile(
            path
        ), f"Error: trying to load {repr(cls)} model at unknown path {path}"
        return cls(model=joblib.load(path))

    def __repr__(self):
        return self.__class__.__name__
