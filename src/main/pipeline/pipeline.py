""" Pipeline module. """

from config.config import PIPELINE_DATASET_PATH
from datetime import datetime
from functools import reduce
from joblib import Parallel, delayed
from typing import Callable, List

import os
import pandas as pd


class Pipeline:
    """
    Pipeline class. Contains a list of steps to be executed in parallel on a dataset.
    """

    steps: List[Callable] = []

    def __init__(self, steps=None):
        if steps is None:
            steps = []
        self.steps = steps

    def add_step(self, step: Callable):
        """
        Add a step to the pipeline.
        """
        self.steps.append(step)

    def execute(self, data: pd.DataFrame, model=None, save=False):
        """
        Run the pipeline in parallel.
        The pipeline is run in parallel by splitting the data into chunks and running
        the pipeline on each chunk in parallel.
        The number of chunks is equal to the number of cpu's available.
        The results are flattened and returned.

        :param data: The data to run the pipeline.
        :param model: A string representing the caller model.
        :param save: A boolean indicating whether to save the data to a file.
        :return: The data after the processing.
        """

        assert len(self.steps) > 0, "Pipeline has no steps."
        assert isinstance(data, pd.DataFrame), "Data is not a pandas DataFrame."
        assert not save or model is not None, "Model is not provided."

        def execute_chuck(chunk):
            """
            Runs the pipeline.

            :param chunk: The chunk of data to run the pipeline on.
            :return: The chunk after the processing.
            """
            return list(
                map(
                    lambda item: reduce(
                        lambda previous, step: step(previous), self.steps, item
                    ),
                    chunk,
                )
            )

        cpu_count = os.cpu_count()
        results = {}

        print("Running pipeline in parallel: number of cpu's: ", cpu_count)
        for column in data.columns:
            chunks = [data[column][i::cpu_count] for i in range(cpu_count)]
            now = datetime.now()
            results[column] = Parallel(n_jobs=cpu_count)(
                delayed(execute_chuck)(chunk) for chunk in chunks
            )
            results[column] = [result for chunk in results[column] for result in chunk]
            print("Time taken: ", datetime.now() - now)

        df = pd.DataFrame(results)
        if model is not None:
            df.to_csv(PIPELINE_DATASET_PATH.format(model), index=False)
        return df
