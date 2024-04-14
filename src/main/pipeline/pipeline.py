""" Pipeline module. """

from config.config import LOGISTIC_PIPELINE_DATASET_PATH
from datetime import datetime
from functools import reduce
from joblib import Parallel, delayed
from typing import Callable, List

import os
import pandas as pd


class Pipeline:
    """
    Interface for pipelines.
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

    def execute(self, data: pd.DataFrame, save=True):
        """
        Run the pipeline in parallel.
        The pipeline is run in parallel by splitting the data into chunks and running
        the pipeline on each chunk in parallel.
        The number of chunks is equal to the number of cpu's available.
        The results are flattened and returned.

        :param data: The data to run the pipeline.
        :return: The data after the processing.
        """

        def execute_chuck(data):
            """
            Run the pipeline.
            """
            return list(
                map(
                    lambda item: reduce(
                        lambda previous, step: step(previous), self.steps, item
                    ),
                    data,
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

        print(results)
        df = pd.DataFrame(results)
        if save:
            df.to_csv(LOGISTIC_PIPELINE_DATASET_PATH, index=False)
        return df
