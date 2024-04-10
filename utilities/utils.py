""" This module contains utility functions for the project. """

import pandas as pd
from config.config import DATASET_PATH


def get_dataset(filepath=DATASET_PATH):
    """
    Read the dataset from the given filepath and return the dataframe.

    :param filepath: The path to the dataset file.
    :return: The dataframe containing the dataset.
    """
    return pd.read_json(filepath, lines=True)
