""" Pipeline module. """

import numpy as np

from config.config import PIPELINE_DATASET_PATH
from datetime import datetime
from itertools import groupby
from functools import reduce
from joblib import Parallel, delayed
from src.main.utilities import utils

from typing import Callable, List

import inspect
import os


class Pipeline:
    """
    Pipeline class. Contains a list of steps to be executed in parallel on a dataset.
    """

    @staticmethod
    def is_parallelizable(f):
        return inspect.signature(f).parameters.get("parallel_mode").default

    def __init__(self, stages: List[Callable] = None):
        assert isinstance(stages, List), "The pipeline should be a list of stages."
        assert all(
            isinstance(stage, Callable) for stage in stages
        ), "Pipeline error: found a non callable item"

        self.parallel_execution_mode = (
            Pipeline.is_parallelizable(stages[0]) if stages else False
        )
        self.stages = stages

    @property
    def stages(self):
        return self._stages

    @stages.setter
    def stages(self, stages):
        self._stages = [
            list(group) for _, group in groupby(stages, Pipeline.is_parallelizable)
        ]

    def switch_parallel_execution_mode(self):
        self.parallel_execution_mode = not self.parallel_execution_mode

    def execute(self, data: np.ndarray, model_file=None, save=False):
        """
        The pipeline is run in parallel by splitting the data into chunks and executing the functions on each chunk.
        The number of chunks is equal to the number of cpu's available.
        The results are flattened and returned.

        :param data: The data to run the pipeline.
        :param model_file: A string representing the caller model.
        :param save: A boolean indicating whether to save the data to a file.
        :return: The data after the processing.
        """

        assert len(self.stages) > 0, "Pipeline has no steps."
        assert isinstance(data, np.ndarray), "Data is not a numpy array."
        assert not save or model_file is not None, "Model is not provided."

        def run_chunk(functions_chunk, data_chunk):
            """
            Runs the pipeline.

            :param functions_chunk: The functions to run on the data.
            :param data_chunk: The data to run the functions on.
            :return: The chunk after the processing.
            """
            return np.vectorize(
                lambda item: reduce(
                    lambda previous, step: step(previous), functions_chunk, item
                )
            )(data_chunk)

        def run_whole(functions_chunk, full_data):
            """
            Runs the pipeline.

            :param functions_chunk: The functions to run on the data.
            :param full_data: The data to run the functions on.
            :return: The chunk after the processing.
            """
            return reduce(
                lambda previous, step: step(previous), functions_chunk, full_data
            )

        # Load the preprocessing result if it already exists
        if model_file is not None and os.path.exists(
            os.path.join(PIPELINE_DATASET_PATH, model_file)
        ):
            return utils.load_preprocessing(model_file)

        cpu_count = os.cpu_count()
        results = data

        now = datetime.now()
        print(f"Pipeline started")

        for function_chunk in self.stages:

            if self.parallel_execution_mode:
                chunks = [chunk for chunk in np.array_split(results, cpu_count)]
                results = Parallel(n_jobs=cpu_count)(
                    delayed(run_chunk)(function_chunk, chunk) for chunk in chunks
                )
                results = np.concatenate(results, axis=0)
            else:
                results = run_whole(function_chunk, results.flatten().tolist())

            self.switch_parallel_execution_mode()

        print(f"Pipeline execution time: {datetime.now() - now}")

        if save and model_file is not None:
            utils.save_preprocessing(results, model_file)

        return np.array(results)
