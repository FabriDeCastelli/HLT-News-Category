import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix,
)


def plot_confusion_matrix(y_test, y_pred, path=None):
    """
    Plots the confusion matrix. If path is None the plot is shown, otherwise it is saved in the path.

    :param y_test: real target values
    :param y_pred: the predicted values
    :param path: the path to save the plot
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
    if path is not None:
        plt.savefig(path)
    plt.show()


# Deprecated? -------


def compute_performance(y_test, y_pred):
    """
    Compute the performance of the model.

    :param y_test: real target values
    :param y_pred: predicted values
    :return: a dictionary with performance metrics
    """
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "F1 (Macro)": f1_score(y_test, y_pred, average="macro"),
        "Precision (Macro)": precision_score(y_test, y_pred, average="macro"),
        "Recall (Macro)": recall_score(y_test, y_pred, average="macro"),
    }


def print_performance(results):
    """
    Plot the performance of the model.

    :param results: the results of the model
    """
    # Plot the performance
    for key, value in results.items():
        print(key, ": ", value)
    return


def plot_performance(results):
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
