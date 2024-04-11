""" Pipeline module. """

from functools import reduce
from typing import Callable, List


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

    def execute(self, data):
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
