import unittest

import numpy as np

from src.main.pipeline.pipeline import Pipeline
import src.main.utilities.utils as utils
import src.main.pipeline.functions as functions


class TestPipeline(unittest.TestCase):
    def test_result_order(self):
        def f(x, parallel_mode=True):
            return x + 1

        data = np.arange(100)
        pipeline = Pipeline([f])
        result = pipeline.execute(data)
        self.assertTrue((result == np.arange(1, 101)).all())

    def test_non_parallel_function_data_consistency(self):
        def f(x, parallel_mode=True):
            return x + 1

        def g(x, parallel_mode=False):
            return x

        data = np.arange(100)
        pipeline = Pipeline([f, g])
        result = pipeline.execute(data)
        self.assertTrue((result == np.arange(1, 101)).all())

    def test_dataset(self):
        def g(x, parallel_mode=False):
            return x

        data = utils.get_dataset()[0][:10]
        pipeline = Pipeline([functions.clean_text, g])
        result = pipeline.execute(data)
        without_g = Pipeline([functions.clean_text])
        result_no_g = without_g.execute(data).reshape(-1)
        self.assertTrue((result == result_no_g).all())

    def test_alternating_execution_mode(self):
        def f(x, parallel_mode=True):
            return x + 1

        def g(x, parallel_mode=False):
            return x

        data = np.arange(100)
        pipeline = Pipeline([f, g, f])
        result = pipeline.execute(data)
        self.assertTrue((result == np.arange(2, 102)).all())

    def test_parallel_sequential(self):
        def f(x, parallel_mode=True):
            return x + 1

        def g(x, parallel_mode=False):
            return [i + 1 for i in x]

        data = np.arange(100)
        pipeline = Pipeline([g, f, g, f])
        result = pipeline.execute(data)
        self.assertTrue((result == np.arange(4, 104)).all())


if __name__ == "__main__":
    unittest.main()
