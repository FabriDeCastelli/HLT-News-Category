import unittest
from src.main.pipeline.pipeline import Pipeline
import pandas as pd


class TestPipeline(unittest.TestCase):
    def test_result_order(self):
        def f(x, parallel_mode=True):
            return x + 1

        data = pd.DataFrame({"full_article": range(100)})
        pipeline = Pipeline([f])
        result = pipeline.execute(data)
        self.assertEqual(result, list(range(1, 101)))

    def test_non_parallel_function_data_consistency(self):
        def f(x, parallel_mode=True):
            return x + 1

        def g(x, parallel_mode=False):
            return x

        data = pd.DataFrame({"full_article": range(100)})
        pipeline = Pipeline([f, g])
        result = pipeline.execute(data)
        self.assertEqual(result, list(range(1, 101)))

    def test_dataset(self):
        import src.main.utilities.utils as utils
        import src.main.pipeline.functions as functions

        def g(x, parallel_mode=False):
            return x

        data = utils.get_dataset()[0][:10]
        pipeline = Pipeline([functions.clean_text, g])
        result = pipeline.execute(data)
        without_g = Pipeline([functions.clean_text])
        result_no_g = without_g.execute(data)
        self.assertEqual(result, result_no_g)

    def test_alternating_execution_mode(self):
        def f(x, parallel_mode=True):
            return x + 1

        def g(x, parallel_mode=False):
            return x

        data = pd.DataFrame({"full_article": range(100)})
        pipeline = Pipeline([f, g, f])
        result = pipeline.execute(data)
        self.assertEqual(result, list(range(2, 102)))

    def test_parallel_sequential(self):
        def f(x, parallel_mode=True):
            return x + 1

        def g(x, parallel_mode=False):
            return [i + 1 for i in x]

        data = pd.DataFrame({"full_article": range(100)})
        pipeline = Pipeline([g, f, g, f])
        result = pipeline.execute(data)
        self.assertEqual(result, list(range(4, 104)))


if __name__ == "__main__":
    unittest.main()
