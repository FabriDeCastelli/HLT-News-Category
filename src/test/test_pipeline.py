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


if __name__ == "__main__":
    unittest.main()
