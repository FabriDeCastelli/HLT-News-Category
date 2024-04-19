""" Multinomial Logistic Regression. """
from src.main.models.model import Model
from src.main.pipeline.pipeline import Pipeline
from config.config import PIPELINE_DATASET_PATH

import os
from typing import Callable, List

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import load_npz
from scipy.stats import uniform
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import get_scorer, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix


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

    def run_pipeline(self, data: pd.DataFrame, save=True):
        """
        Run the pipeline. If the pipeline for this model has already been run, then the dataset is read from the file.

        :param data: the data to run the pipeline on.
        :param save: a boolean indicating whether to save the data to a file.
        :return: the data after the processing.
        """
        assert self.pipeline is not None, "Pipeline is not set."
        assert isinstance(data, pd.DataFrame), "Data is not a pandas DataFrame."

        path = PIPELINE_DATASET_PATH.format(repr(self) + ".npz")
        if os.path.exists(path):
            return load_npz(path)
        return self.pipeline.execute(data, model=repr(self), save=True)

    def fit(self, inputs, targets, sample_weight=None):
        """
        Fit the model to the data.

        :param inputs: the training data, excluding the targets
        :param targets: the target values
        :param sample_weight: the weights of the samples
        """
        self.logistic = self.logistic.fit(inputs, targets, sample_weight)


    def grid_search(self, x_train, y_train, n_iter=30):
        """
        Cross validate the model.

        :param x_train: the training data
        :param y_train: the target values
        :param n_iter: the number of iterations to run the Randomized Search
        :return: the cross validation results
        """

        # Parameter grid for Logistic Regressor
        params = {
            "penalty": ["l2", "none"],
            "C": uniform(loc=0.001, scale=5 - 0.001),
            "solver": ["newton-cg", "lbfgs", "sag", "saga"],
            "class_weight": ["balanced", None]
        }
        # Randomized Search
        rscv = RandomizedSearchCV(
            estimator=self.logistic,
            param_distributions=params,
            refit=get_scorer("accuracy"),
            n_jobs=-1,
            n_iter=n_iter,
            random_state=42
        )
        return rscv.fit(x_train, y_train)


    def evaluate(self, inputs, targets):
        """
        Evaluate the model.

        :param inputs: the data to evaluate the model on
        :param targets: the target values
        :return: the score of the model
        """
        return self.logistic.score(inputs, targets)
    
    def predict(self, data):
        """
        Make prediction over data.

        :param data: the data to predict
        :return: the predicted values
        """
        return self.logistic.predict(data)

    def plot_confusion_matrix(self, y_test, y_pred):
        """
        plot the confusion matrix.

        :param y_test: real target values
        :param y_pred: the predicted values
        """
        cm = confusion_matrix(y_test, y_pred)

        # Category names in order
        categories = ['Entertainment', 'Life', 'Politics', 'Sport', 'Voices']

        # confusion matrix plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=categories, yticklabels=categories)
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title('Confusion Matrix')
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
        res["f1-macro"] = f1_score(y_test, y_pred, average='macro')
        res["Precision"] = precision_score(y_test, y_pred, average='macro')
        res["Recall"] = recall_score(y_test, y_pred, average='macro')

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
        ax.set_ylabel('Performance')
        ax.set_xlabel('Metric Name')
        ax.set_title('Model Performances on Different Metrics')

        # Rotate x-axis labels for better readability (optional)
        plt.xticks(rotation=45, ha='right')

        # Set y-axis ticks from 0 to 1.0 with increments of 0.05
        plt.yticks(np.arange(0, 1.05, 0.1))

        # Show the plot
        plt.show()
        return

    def __repr__(self):
        return "Logistic"

