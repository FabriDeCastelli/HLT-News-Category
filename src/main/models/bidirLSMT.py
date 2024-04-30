""" Bidirectional LSTM. """

from src.main.models.model import Model
from src.main.pipeline.pipeline import Pipeline
from config.config import MODELS_PATH

import os
from typing import Callable, List

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib

from keras.models import Sequential
from keras.layers import Input, Embedding, Bidirectional, LSTM, Dense

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix,
)


class BidirectionalLSTM(Model):
    """
    Bidirectional LSTM class.
    """

    def __init__(self, **kwargs):
        """
        Constructor for the Bidirectional LSTM class.

        :param kwargs: the arguments that are going to be passed to the Bidirectional LSTM model.
        """
        super().__init__()

        # Get the parameters from the kwargs
        vocab_size = kwargs.get("vocab_size", 10000)
        embedding_dim = kwargs.get("embedding_dim", 100)
        max_sequence_length = kwargs.get("max_sequence_length", 100)
        lstm_units = kwargs.get("lstm_units", 128)

        # Define the model architecture
        bidirLSTM = Sequential()
        # Add an input layer
        bidirLSTM.add(Input(shape=(None,), dtype="int32"))
        # Add an embedding layer to convert input sequences to dense vectors
        bidirLSTM.add(
            Embedding(
                input_dim=vocab_size,
                output_dim=embedding_dim,
                input_length=max_sequence_length,
            )
        )
        # Add a Bidirectional LSTM layer
        bidirLSTM.add(Bidirectional(LSTM(units=lstm_units, return_sequences=True)))
        bidirLSTM.add(Bidirectional(LSTM(units=lstm_units)))
        # Add a dense output layer
        bidirLSTM.add(Dense(units=5, activation="softmax"))
        # Compile the model
        bidirLSTM.compile(
            loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
        )

        self.bidirLSTM = bidirLSTM
        self.pipeline = None

    def set_model(self, model):
        """
        Set the Bidirectional LSTM.
        """
        self.bidirLSTM = model

    def set_pipeline(self, pipeline: List[Callable]):
        """
        Set the pipeline for the model.

        :param pipeline: an array of functions that are going to be executed in the pipeline.
        :return:
        """
        self.pipeline = Pipeline(pipeline)

    def run_pipeline(self, data: pd.DataFrame, save=True):
        """
        Run the pipeline. If the pipeline for this model has already been run, then the dataset is read from the file.

        :param data: the data to run the pipeline on.
        :param save: a boolean indicating whether to save the data to a file.
        :return: the data after the processing.
        """
        assert self.pipeline is not None, "Pipeline is not set."
        assert isinstance(data, pd.DataFrame), "Data is not a pandas DataFrame."
        return self.pipeline.execute(data, model_file=repr(self) + ".json", save=save)

    def fit(self, inputs, targets, sample_weight=None):
        """
        Fit the model to the data.

        :param inputs: the training data, excluding the targets
        :param targets: the target values
        :param sample_weight: the weights of the samples
        """
        if os.path.isfile(os.path.join(MODELS_PATH, repr(self) + ".pkl")):
            self.upload_model()
        else:
            self.bidirLSTM = self.bidirLSTM.fit(
                inputs, targets, sample_weight, verbose=0
            )

    def grid_search(self, x_train, y_train, n_iter=30):
        """
        Cross validate the model.

        :param x_train: the training data
        :param y_train: the target values
        :param n_iter: the number of iterations to run the Randomized Search
        :return: the cross validation results
        """

        # TODO

    def evaluate(self, inputs, targets):
        """
        Evaluate the model.

        :param inputs: the data to evaluate the model on
        :param targets: the target values
        :return: the score of the model
        """
        return self.bidirLSTM.evaluate(inputs, targets)

    def predict(self, data):
        """
        Make prediction over data.

        :param data: the data to predict
        :return: the predicted values
        """
        return self.bidirLSTM.predict(data)

    def plot_confusion_matrix(self, y_test, y_pred):
        """
        plot the confusion matrix.

        :param y_test: real target values
        :param y_pred: the predicted values
        """
        cm = confusion_matrix(y_test, y_pred)

        # Category names in order
        categories = ["Entertainment", "Life", "Politics", "Sport", "Voices"]

        # confusion matrix plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=categories,
            yticklabels=categories,
        )
        plt.xlabel("Predicted labels")
        plt.ylabel("True labels")
        plt.title("Confusion Matrix")
        plt.show()
        return

    def compute_performance(self, y_test, y_pred):
        """
        Compute the performance of the model.

        :param y_test: real target values
        :param y_pred: predicted values
        :return: dictionary with performance matrics
        """
        res = {}
        res["Accuracy"] = accuracy_score(y_test, y_pred)
        res["f1-macro"] = f1_score(y_test, y_pred, average="macro")
        res["Precision"] = precision_score(y_test, y_pred, average="macro")
        res["Recall"] = recall_score(y_test, y_pred, average="macro")

        return res

    def print_performance(self, results):
        """
        Plot the performance of the model.

        :param results: the results of the model
        """
        # Plot the performance
        for key, value in results.items():
            print(key, ": ", value)
        return

    def plot_performance(self, results):
        """
        Plot the performance of the model.

        :param results: the results of the model
        """
        # Plot the performance
        # Assuming you already have metric_names and performances defined
        metric_names = list(results.keys())
        performances = list(results.values())
        # Create a bar plot using seaborn
        sns.set(style="whitegrid")
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x=metric_names, y=performances, palette="viridis")
        ax.set_ylabel("Performance")
        ax.set_xlabel("Metric Name")
        ax.set_title("Model Performances on Different Metrics")

        # Rotate x-axis labels for better readability (optional)
        plt.xticks(rotation=45, ha="right")

        # Set y-axis ticks from 0 to 1.0 with increments of 0.05
        plt.yticks(np.arange(0, 1.05, 0.1))

        # Show the plot
        plt.show()
        return

    def save_model(self):
        """
        Save the model to a file.

        :param path: the path to save the model to
        """

        joblib.dump(self.bidirLSTM, MODELS_PATH + "bidirLSTM.pkl")
        return

    def upload_model(self):
        """
        Load the model from a file.
        """
        self.bidirLSTM = joblib.load(MODELS_PATH + "bidirLSTM.pkl")
        return

    def summary(self):
        """
        Print the summary of the model.
        """
        print(self.bidirLSTM.summary())

    def __repr__(self):
        return "bidirLSTM"
