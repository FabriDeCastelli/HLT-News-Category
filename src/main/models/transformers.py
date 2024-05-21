"""General transformers model"""
from src.main.models.model import Model
from src.main.pipeline.pipeline import Pipeline
from typing import Callable, List
import pandas as pd


class Transformer():
    """
    General transformers class.
    """

    def __init__(self):
        """
        Constructor for the general transformers class.
        """
        self._pipeline = None

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
        return self.pipeline.execute(data, model_file=repr(self) + ".npy", save=save)
    
    def __repr__(self):
            return self.__class__.__name__