""" Multinomial Logistic Regression. """

from src.main.models.model import Model
from src.main.pipeline.pipeline import Pipeline


class Logistic(Model):
    """
    Multinomial Logistic Regression class.
    """

    def __init__(self):
        super().__init__()
        self.pipeline = Pipeline()

    def set_pipeline(self, pipeline):
        self.pipeline = Pipeline(pipeline)

    def run_pipeline(self, data):
        return self.pipeline.execute(data)

    def fit(self, inputs, targets):
        raise NotImplementedError()

    def predict(self, inputs):
        raise NotImplementedError()
