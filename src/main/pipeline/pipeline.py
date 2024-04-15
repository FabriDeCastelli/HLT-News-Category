""" Pipeline module. """

from config.config import PIPELINE_DATASET_PATH
from datetime import datetime
from itertools import groupby
from functools import reduce
from joblib import Parallel, delayed
from typing import Callable, List

import inspect
import os
import pandas as pd


class Pipeline:
    """
    Pipeline class. Contains a list of steps to be executed in parallel on a dataset.
    """

    steps: List[Callable] = []

    def __init__(self, steps=None):
        if steps is None or steps == []:
            steps = []

        def is_parallelizable(f):
            return inspect.signature(f).parameters.get("parallel_mode").default

        self.parallel_execution_mode = is_parallelizable(steps[0]) if steps else False

        self.steps = [list(group) for _, group in groupby(steps, is_parallelizable)]

    def switch_parallel_execution_mode(self):
        self.parallel_execution_mode = not self.parallel_execution_mode

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

        def run_chunk(functions_chunk, data_chunk):
            """
            Runs the pipeline.

            :param functions_chunk: The functions to run on the data.
            :param data_chunk: The data to run the functions on.
            :return: The chunk after the processing.
            """
            return list(
                map(
                    lambda item: reduce(
                        lambda previous, step: step(previous), functions_chunk, item
                    ),
                    data_chunk,
                )
            )

        cpu_count = os.cpu_count()
        results = {}

        now = datetime.now()

        for column in data.columns:

            for function_chunk in self.steps:
                if self.parallel_execution_mode:
                    chunks = [data[column][i::cpu_count] for i in range(cpu_count)]
                    results[column] = Parallel(n_jobs=cpu_count)(
                        delayed(run_chunk)(function_chunk, chunk) for chunk in chunks
                    )
                    results[column] = [
                        result for chunk in results[column] for result in chunk
                    ]
                else:
                    results[column] = run_chunk(function_chunk, data[column])

                self.switch_parallel_execution_mode()

        print("Pipeline execution time: ", datetime.now() - now)
        df = pd.DataFrame(results)
        if model is not None:
            df.to_csv(PIPELINE_DATASET_PATH.format(model), index=False)
        return df
