""" Multinomial Logistic Regression. """

from src.main.models.model import Model
from src.main.pipeline.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


class Logistic(Model):
    """
    Multinomial Logistic Regression class.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.logistic = LogisticRegression(**kwargs)
        self.pipeline = None

    def set_pipeline(self, pipeline):
        self.pipeline = Pipeline(pipeline)

    def run_pipeline(self, data):
        return self.pipeline.execute(data)

    def fit(self, inputs, targets, sample_weight=None):
        self.logistic.fit(inputs, targets, sample_weight)

    def predict(self, inputs):
        raise NotImplementedError()
