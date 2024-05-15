""" Bidirectional LSTM. """

import keras as K
import keras_tuner as kt

from config import config
from keras_tuner import HyperModel
from src.main.models.model import Model
from src.main.pipeline.pipeline import Pipeline
from config.config import HYPERPARAMETERS_PATH, RESULTS_DIRECTORY
from src.main.utilities.utils import read_yaml
from typing import Callable, List
from keras import optimizers

import os
import pandas as pd


class BidirectionalLSTM(Model, HyperModel):
    """
    Bidirectional LSTM class.
    """

    def __init__(self, model: K.models.Sequential = None, **kwargs):
        """
        Constructor for the Bidirectional LSTM class.

        :param model: the K.models.Sequential model, if provided in input **kwargs are ignored
        """
        super().__init__()
        self._pipeline = None
        if model is not None:
            self.bidirLSTM = model
            return

        self._bidirLSTM = self.build(None, **kwargs)

        self.hyperparameters = read_yaml(HYPERPARAMETERS_PATH.format("LSTM"))

    def build(self, hp=None, **kwargs):

        if hp is None:
            return self._build_fixed(**kwargs)
        return self._build_hp(hp)

    def _build_hp(self, hp):
        vocab_size = hp.Choice("vocab_size", self.hyperparameters["vocab_size"])
        embedding_dim = hp.Choice(
            "embedding_dim", self.hyperparameters["embedding_dim"]
        )
        max_sequence_length = hp.Choice(
            "max_sequence_length", self.hyperparameters["max_sequence_length"]
        )
        lstm_units_1 = hp.Choice("lstm_units_1", self.hyperparameters["lstm_units_1"])
        lstm_units_2 = hp.Choice("lstm_units_2", self.hyperparameters["lstm_units_2"])

        bidirLSTM = K.models.Sequential()
        bidirLSTM.add(K.layers.InputLayer(shape=(None,)))
        bidirLSTM.add(
            K.layers.Embedding(
                input_dim=vocab_size,
                output_dim=embedding_dim,
                input_length=max_sequence_length,
            )
        )
        bidirLSTM.add(
            K.layers.Bidirectional(
                K.layers.LSTM(units=lstm_units_1, return_sequences=True)
            )
        )
        bidirLSTM.add(K.layers.Bidirectional(K.layers.LSTM(units=lstm_units_2)))
        bidirLSTM.add(K.layers.Dense(units=5, activation="softmax"))
        optimizer = optimizers.Adam(
            learning_rate=hp.Choice(
                "learning_rate", self.hyperparameters["learning_rate"]
            )
        )
        bidirLSTM.compile(
            loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"]
        )

        return bidirLSTM

    def _build_fixed(self, **kwargs):

        vocab_size = kwargs.get("vocab_size", 10000)
        embedding_dim = kwargs.get("embedding_dim", 16)
        max_sequence_length = kwargs.get("max_sequence_length", 100)
        lstm_units = kwargs.get("lstm_units", 32)

        bidirLSTM = K.models.Sequential()
        bidirLSTM.add(K.layers.InputLayer(shape=(None,)))
        bidirLSTM.add(
            K.layers.Embedding(
                input_dim=vocab_size,
                output_dim=embedding_dim,
                input_length=max_sequence_length,
            )
        )
        bidirLSTM.add(
            K.layers.Bidirectional(
                K.layers.LSTM(units=lstm_units, return_sequences=True)
            )
        )
        bidirLSTM.add(K.layers.Bidirectional(K.layers.LSTM(units=lstm_units)))
        bidirLSTM.add(K.layers.Dense(units=5, activation="softmax"))
        bidirLSTM.compile(
            loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
        )

        self.bidirLSTM = bidirLSTM
        return bidirLSTM

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
        return self.pipeline.execute(data, model_file=repr(self) + ".npy", save=save)

    def fit(self, hp=None, model=None, *args, **kwargs) -> K.callbacks.History:
        """
        Fit the model to the data.

        :param hp: the hyperparameters, if None the model is trained with the fixed hyperparameters
        :param model: the model, needed for keras-tuner
        :return: the training history
        """
        if hp is None or model is None:
            return self.bidirLSTM.fit(*args, **kwargs)
        epochs = hp.Choice("epochs", self.hyperparameters["epochs"])
        batch_size = hp.Choice("batch_size", self.hyperparameters["batch_size"])
        return model.fit(*args, epochs=epochs, batch_size=batch_size, **kwargs)

    def grid_search(
        self,
        x_train,
        y_train,
        callbacks=None,
        validation_split=0.2,
        n_iter=30,
        refit=True,
    ):
        """
        Performs some hyperparameters' optimization, using kera-tuner's RandomSearch.

        :param x_train: the training data
        :param y_train: the target values
        :param callbacks: the callbacks to use during training
        :param validation_split: the validation split
        :param n_iter: the number of iterations to run the Randomized Search
        :param refit: whether to refit the model on the best hyperparameters
        :return: the list of best hyperparameters in Hyperparameters format
        """
        assert isinstance(
            callbacks, List
        ), "The callbacks should be a list of K.callbacks.Callback."
        assert all(
            isinstance(callback, K.callbacks.Callback) for callback in callbacks
        ), "Grid search error: found a non callback item."

        tuner = kt.RandomSearch(
            self,
            objective="val_accuracy",
            max_trials=n_iter,
            executions_per_trial=1,
            directory=RESULTS_DIRECTORY.format(repr(self)),
            project_name=repr(self),
        )
        tuner.search(
            x_train, y_train, callbacks=callbacks, validation_split=validation_split
        )
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        self.bidirLSTM = tuner.hypermodel.build(best_hps)

        if refit:
            self.fit(
                x=x_train,
                y=y_train,
                callbacks=callbacks,
                validation_split=validation_split,
            )

        return best_hps

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
        if not os.path.exists(config.MODELS_PATH):
            os.mkdir(config.MODELS_PATH)
        self._bidirLSTM.save(path)

    @classmethod
    def load_model(cls):
        """
        Load the model from a file.
        """
        path = os.path.join(config.MODELS_PATH, cls.__name__ + ".keras")
        assert os.path.isfile(
            path
        ), f"Error: trying to load {cls.__name__} model at unknown path {path}"
        return cls(model=K.saving.load_model(path))

    def summary(self):
        """
        Print the summary of the model.
        """
        print(self._bidirLSTM.summary())

    def __repr__(self):
        return self.__class__.__name__
