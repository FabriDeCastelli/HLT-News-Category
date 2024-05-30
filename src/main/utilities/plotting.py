""" Utilities to plot and save data."""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from config import config


def plot_confusion_matrix(y_test, y_pred, path=None):
    """
    Plots the confusion matrix. If the path is provided the plot is saved, otherwise it is only shown.

    :param y_test: real target values
    :param y_pred: the predicted values
    :param path: where to save the plot
    """

    labels = list(config.label2id.keys())
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    # confusion matrix plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.title("Confusion Matrix")
    if path is not None:
        plt.savefig(path)
    plt.show()
