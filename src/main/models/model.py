""" Model interface. """

from src.main.pipeline.pipeline import Pipeline


class Model:
    """
    Interface for models.
    """

    pipeline: Pipeline = None

    def __init__(self):
        pass

    def set_pipeline(self, pipeline):
        """
        Set the pipeline for the model.
        """
        raise NotImplementedError()

    def run_pipeline(self, data):
        """
        Run the pipeline on the data.

        :param data: The data to run the pipeline on.
        """
        raise NotImplementedError()

    def fit(self, inputs, targets):
        """
        Fit the model to the data.

        :param inputs: The input data.
        :param targets: The target data.
        """
        raise NotImplementedError()

    def predict(self, inputs):
        """
        Predict the target variable for the input data.

        :param inputs: The input data.
        """
        raise NotImplementedError()
