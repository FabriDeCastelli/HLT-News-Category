""" This module contains utility functions for the project. """

import pandas as pd
from config.config import *


def label_ranaming(df):
    di = {}
    for new,orig in zip(new_names, [life, entertainment, voices, sports, politics]):
        for label in orig:
            di[label] = new
    df = df.replace({"category": di})    
    
    return df[df['category'].isin(new_names)]
        
def get_dataset(filepath=DATASET_PATH):
    """
    Read the dataset from the given filepath and return the dataframe.

    :param filepath: The path to the dataset file.
    :return: The dataframe containing the dataset.
    """
    df = pd.read_json(filepath, lines=True)
    df = label_ranaming(df) 
    df.drop(labels=drop_column, inplace=True, axis=1)
    return df
