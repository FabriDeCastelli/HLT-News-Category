""" Bidirectional LSTM. """

import json

import keras as K
import keras_tuner as kt
import numpy as np

from config import config
from keras_tuner import HyperModel

from src.main.utilities import utils
from src.main.models.model import Model
from src.main.pipeline.pipeline import Pipeline
from src.main.utilities.utils import read_yaml, f1_macro, precision_macro, recall_macro
from typing import Callable, List
from keras import optimizers

import os
import pandas as pd


class BidirectionalLSTM(Model, HyperModel):
    """
    Bidirectional LSTM class.
    """

    def __init__(
        self, model: K.models.Sequential = None, pretrained_embeddings=None, **kwargs
    ):
        """
        Constructor for the Bidirectional LSTM class.

        :param model: the K.models.Sequential model, if provided in input **kwargs are ignored
        """
        super().__init__()
        self.embedding_matrix = None
        self._pipeline = None
        if model is not None:
            self.bidirLSTM = model
            return

        self.metrics = [
            "accuracy",
            f1_macro,
            precision_macro,
            recall_macro,
        ]
        self.pretrained_embeddings = pretrained_embeddings
        self._bidirLSTM = self.build(None, **kwargs)
        self.hyperparameters = read_yaml(config.HYPERPARAMETERS_PATH.format(repr(self)))

    def build(self, hp=None, **kwargs):

        if hp is None:
            return self._build_fixed(**kwargs)
        return self._build_hp(hp)

    def _build_hp(self, hp):
        lstm_units_1 = hp.Choice("lstm_units_1", self.hyperparameters["lstm_units_1"])
        lstm_units_2 = hp.Choice("lstm_units_2", self.hyperparameters["lstm_units_2"])

        bidirLSTM = K.models.Sequential()
        bidirLSTM.add(K.layers.InputLayer(shape=(None,)))

        if self.embedding_matrix is not None:
            bidirLSTM.add(
                K.layers.Embedding(
                    input_dim=config.num_words,
                    output_dim=config.EMBEDDING_DIM,
                    input_length=config.MAX_SEQ_LENGTH,
                    weights=[self.embedding_matrix],
                    trainable=False,
                )
            )
        else:
            bidirLSTM.add(
                K.layers.Embedding(
                    input_dim=config.num_words,
                    output_dim=config.EMBEDDING_DIM,
                    input_length=config.MAX_SEQ_LENGTH,
                )
            )

        bidirLSTM.add(
            K.layers.Bidirectional(
                K.layers.LSTM(
                    units=lstm_units_1,
                    dropout=hp.Choice("dropout1", self.hyperparameters["dropout1"]),
                    return_sequences=True,
                )
            )
        )
        bidirLSTM.add(
            K.layers.Bidirectional(
                K.layers.LSTM(
                    units=lstm_units_2,
                    dropout=hp.Choice("dropout2", self.hyperparameters["dropout2"]),
                )
            )
        )
        bidirLSTM.add(
            K.layers.Dense(
                hp.Choice("dense1", self.hyperparameters["dense1"]), activation="relu"
            )
        )
        bidirLSTM.add(K.layers.Dropout(0.3, seed=10))
        bidirLSTM.add(
            K.layers.Dense(
                hp.Choice("dense2", self.hyperparameters["dense2"]), activation="relu"
            )
        )
        bidirLSTM.add(K.layers.Dense(units=5, activation="softmax"))

        optimizer = optimizers.Adam(
            learning_rate=hp.Choice(
                "learning_rate", self.hyperparameters["learning_rate"]
            )
        )

        bidirLSTM.compile(optimizer, "categorical_crossentropy", metrics=self.metrics)

        return bidirLSTM

    def _build_fixed(self, **kwargs):

        lstm_units_1 = kwargs.get("lstm_units_1", 105)
        lstm_units_2 = kwargs.get("lstm_units_2", 42)
        dropout1 = kwargs.get("dropout1", 0.3)
        dropout2 = kwargs.get("dropout2", 0.15)
        dense_1 = kwargs.get("dense1", 50)
        dense_2 = kwargs.get("dense2", 17)

        bidirLSTM = K.models.Sequential()
        bidirLSTM.add(K.layers.InputLayer(shape=(None,)))
        if self.embedding_matrix is not None:
            bidirLSTM.add(
                K.layers.Embedding(
                    input_dim=config.num_words,
                    output_dim=config.EMBEDDING_DIM,
                    input_length=config.MAX_SEQ_LENGTH,
                    weights=[self.embedding_matrix],
                    trainable=False,
                )
            )
        else:
            bidirLSTM.add(
                K.layers.Embedding(
                    input_dim=config.num_words,
                    output_dim=config.EMBEDDING_DIM,
                    input_length=config.MAX_SEQ_LENGTH,
                )
            )
        bidirLSTM.add(
            K.layers.Bidirectional(
                K.layers.LSTM(
                    units=lstm_units_1, return_sequences=True, dropout=dropout1
                )
            )
        )
        bidirLSTM.add(
            K.layers.Bidirectional(K.layers.LSTM(units=lstm_units_2, dropout=dropout2))
        )

        bidirLSTM.add(K.layers.Dense(dense_1, activation="relu"))
        bidirLSTM.add(K.layers.Dense(dense_2, activation="relu"))
        bidirLSTM.add(K.layers.Dense(units=5, activation="softmax"))

        optimizer = optimizers.Adam(learning_rate=0.01)

        bidirLSTM.compile(optimizer, "categorical_crossentropy", metrics=self.metrics)

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
        assert self.pipeline is not None, "Cannot run the pipeline: it is not set."
        result = self.pipeline.execute(data, model_file=repr(self) + ".npy", save=save)
        path = os.path.join(config.PIPELINE_DATASET_PATH, "word_index.pkl")
        if self.pretrained_embeddings is not None and os.path.isfile(path):
            print("Creating embedding matrix...")
            self.embedding_matrix = utils.create_embedding_matrix(
                utils.load_pretrained_embeddings(self.pretrained_embeddings)
            )
        return result

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
        x_val,
        y_val,
        callbacks=None,
        n_iter=30,
    ):
        """
        Performs some hyperparameters' optimization, using kera-tuner's RandomSearch.

        :param x_train: the training data
        :param y_train: the target values
        :param x_val: the validation data
        :param y_val: the validation target values
        :param callbacks: the callbacks to use during training
        :param n_iter: the number of iterations to run the Randomized Search
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
            objective="val_f1_score",
            max_trials=n_iter,
            executions_per_trial=1,
            directory=config.RESULTS_DIRECTORY.format(repr(self)),
            project_name=repr(self),
        )
        tuner.search(
            x_train, y_train, callbacks=callbacks, validation_data=(x_val, y_val)
        )
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        self.bidirLSTM = tuner.get_best_models(num_models=1)[0]
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
        y_pred = self._bidirLSTM.predict(data)
        classes = np.argmax(y_pred, axis=1)
        return np.vectorize(config.id2label.get)(classes).astype(object)

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
        Load the model from a file. I path is None it loads the weights from the default model location.

        :return: the model
        """
        path = os.path.join(config.MODELS_PATH, cls.__name__ + ".keras")
        assert os.path.isfile(
            path
        ), f"Error: trying to load {cls.__name__} model at unknown path {path}"
        return cls(model=K.saving.load_model(path))

    @classmethod
    def from_experiment(cls, experiment_name):
        """
        Create a BidirectionalLSTM model from an already run experiment.

        :param experiment_name: the name of an already run tuner experiment
        :return: the BidirectionalLSTM model
        """
        experiment_path = os.path.join(
            config.RESULTS_DIRECTORY.format(cls.__name__), experiment_name
        )

        trials_dir = os.path.join(experiment_path)
        trials = os.listdir(trials_dir)

        best_trial = None
        best_hyperparameters = None
        best_score = 0

        for trial in trials:

            if os.path.isfile(os.path.join(trials_dir, trial)):
                continue

            trial_path = os.path.join(trials_dir, trial, "trial.json")

            with open(trial_path, "r") as f:
                trial_data = json.load(f)

            hyperparameters = trial_data["hyperparameters"]["values"]
            score = trial_data["score"]

            # save the best score
            if score > best_score:
                best_score = score
                best_hyperparameters = hyperparameters
                best_trial = trial

        bidirLSTM = cls(**best_hyperparameters)
        weights_path = os.path.join(trials_dir, best_trial, "checkpoint.weights.h5")
        bidirLSTM.bidirLSTM.load_weights(weights_path)
        return bidirLSTM

    @classmethod
    def get_top_experiments(cls, experiment_name, n=5):
        """
        Get the top n trials from a given experiment, in a keras tuner folder.

        :param experiment_name: the name of the experiment
        :param n: the number of top trials to return
        :return: the top n trials, according to the metric for which the model selection was run
        """
        experiment_path = os.path.join(
            config.RESULTS_DIRECTORY.format(cls.__name__), experiment_name
        )

        trials_dir = os.path.join(experiment_path)
        trials = os.listdir(trials_dir)

        top_5 = []

        for trial in trials:

            if os.path.isfile(os.path.join(trials_dir, trial)):
                continue

            trial_path = os.path.join(trials_dir, trial, "trial.json")

            with open(trial_path, "r") as f:
                trial_data = json.load(f)

            hyperparameters = trial_data["hyperparameters"]["values"]
            score = trial_data["score"]

            if score is not None:
                top_5.append((hyperparameters, score))

        return sorted(top_5, key=lambda x: x[1], reverse=True)[:n]

    def summary(self):
        """
        Print the summary of the model.
        """
        print(self._bidirLSTM.summary())

    def __repr__(self):
        return self.__class__.__name__
