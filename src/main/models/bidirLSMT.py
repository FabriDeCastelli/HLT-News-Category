""" Bidirectional LSTM. """

import keras as K

from config import config
from src.main.models.model import Model
from src.main.pipeline.pipeline import Pipeline
from typing import Callable, List

import os
import pandas as pd


class BidirectionalLSTM(Model):
    """
    Bidirectional LSTM class.
    """

    def __init__(self, model: K.models.Sequential = None, **kwargs):
        """
        Constructor for the Bidirectional LSTM class.

        :param model: the K.models.Sequential model, if provided in input **kwargs are ignored
        :param kwargs: the arguments that are going to be passed to the Bidirectional LSTM model.
        """
        self._pipeline = None
        if model is not None:
            self._bidirLSTM = model
            return

        # Get the parameters from the kwargs
        vocab_size = kwargs.get("vocab_size", 10000)
        embedding_dim = kwargs.get("embedding_dim", 100)
        max_sequence_length = kwargs.get("max_sequence_length", 100)
        lstm_units = kwargs.get("lstm_units", 128)

        # Define the model architecture
        bidirLSTM = K.models.Sequential()
        # Add an input layer
        bidirLSTM.add(K.layers.Input(shape=(None,), dtype="int32"))
        # Add an embedding layer to convert input sequences to dense vectors
        bidirLSTM.add(
            K.layers.Embedding(
                input_dim=vocab_size,
                output_dim=embedding_dim,
                input_length=max_sequence_length,
            )
        )
        # Add a Bidirectional LSTM layer
        bidirLSTM.add(
            K.layers.Bidirectional(
                K.layers.LSTM(units=lstm_units, return_sequences=True)
            )
        )
        bidirLSTM.add(K.layers.Bidirectional(K.layers.LSTM(units=lstm_units)))
        # Add a dense output layer
        bidirLSTM.add(K.layers.Dense(units=5, activation="softmax"))
        # Compile the model
        bidirLSTM.compile(
            loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
        )

        self._bidirLSTM = bidirLSTM

    @property
    def bidirLSTM(self):
        return self._bidirLSTM

    @bidirLSTM.setter
    def bidirLSTM(self, model: K.models.Sequential):
        if not isinstance(model, K.models.Sequential):
            raise ValueError(
                "The model should be an instance of keras.models.Sequential"
            )
        self._bidirLSTM = model

    @property
    def pipeline(self):
        return self._pipeline

    @pipeline.setter
    def pipeline(self, stages: List[Callable]):
        """
        Set the pipeline for the model.

        :param stages: an array of functions that are going to be executed in the pipeline.
        :return:
        """
        self._pipeline = Pipeline(stages)

    def run_pipeline(self, data: pd.DataFrame, save=True):
        """
        Run the pipeline. If the pipeline for this model has already been run, then the dataset is read from the file.

        :param data: the data to run the pipeline on as a pandas dataframe.
        :param save: a boolean indicating whether to save the data to a file.
        :return: the data after the processing.
        """
        super().run_pipeline(data)
        assert self.pipeline is not None, "Cannot run the pipeline: it is not set."
        return self.pipeline.execute(data, model_file=repr(self) + ".json", save=save)

    def fit(self, inputs, targets, **kwargs) -> K.callbacks.History:
        """
        Fit the model to the data.

        :param inputs: the training data, excluding the targets
        :param targets: the target values
        :return: the training history
        """
        tensorboard = [
            K.callbacks.TensorBoard(config.TENSORBOARD_LOGS.format(repr(self)))
        ]
        callbacks = kwargs.get("callbacks", [])
        callbacks = callbacks + tensorboard
        return self._bidirLSTM.fit(inputs, targets, callbacks=callbacks, **kwargs)

    def grid_search(self, x_train, y_train, n_iter=30):
        """
        Cross validate the model.

        :param x_train: the training data
        :param y_train: the target values
        :param n_iter: the number of iterations to run the Randomized Search
        :return: the cross validation results
        """
        pass
        # TODO

    def evaluate(self, inputs, targets):
        """
        Evaluate the model.

        :param inputs: the data to evaluate the model on
        :param targets: the target values
        :return: the score of the model
        """
        return self._bidirLSTM.evaluate(inputs, targets)

    def predict(self, data):
        """
        Make prediction over data.

        :param data: the data to predict
        :return: the predicted values
        """
        return self._bidirLSTM.predict(data)

    def save_model(self):
        """
        Save the model to a file.
        """
        path = os.path.join(config.MODELS_PATH, repr(self) + ".keras")
        os.mkdir(path)

    @classmethod
    def load_model(cls):
        """
        Load the model from a file.
        """
        path = os.path.join(config.MODELS_PATH, repr(cls) + ".keras")
        assert os.path.isfile(
            path
        ), f"Error: trying to load {repr(cls)} model at unknown path {path}"
        return cls(model=K.saving.load_model(path))

    def summary(self):
        """
        Print the summary of the model.
        """
        print(self._bidirLSTM.summary())

    def __repr__(self):
        return self.__class__.__name__
