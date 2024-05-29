"""General transformers model"""

import os

import numpy as np
from sklearn.metrics import classification_report

from src.main.utilities import plotting
from src.main.models.model import Model
from src.main.pipeline.pipeline import Pipeline
from typing import Callable, List
import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)
from datasets import Dataset
from config import config
from datasets import load_metric


class Transformer(Model):
    """
    General transformers class.
    """

    def __init__(self, checkpoint="distilbert-base-uncased", **training_args):
        """
        Constructor for the general transformers class.
        """

        self.checkpoint = checkpoint
        self._pipeline = None
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint,
            num_labels=5,
            label2id=config.label2id,
            id2label=config.id2label,
            ignore_mismatched_sizes=True,
        )
        self.metrics = {
            "accuracy": load_metric("accuracy"),
            "precision": load_metric("precision"),
            "recall": load_metric("recall"),
            "f1": load_metric("f1"),
        }
        self.trainer = None
        self.parameters = training_args

    @property
    def pipeline(self):
        return self._pipeline

    @pipeline.setter
    def pipeline(self, stages: List[Callable]):
        """
        Set the pipeline for the model.

        :param stages: an array of functions that are going to be executed in the pipeline.
        :return:
        """
        self._pipeline = Pipeline(stages)

    def run_pipeline(self, data: pd.DataFrame, save=True):
        """
        Run the pipeline. If the pipeline for this model has already been run, then the dataset is read from the file.

        :param data: the data to run the pipeline on as a pandas dataframe.
        :param save: a boolean indicating whether to save the data to a file.
        :return: the data after the processing.
        """
        assert self.pipeline is not None, "Cannot run the pipeline: it is not set."
        return self.pipeline.execute(
            data, model_file=repr(self) + ".npy", save=save
        ).flatten()

    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        accuracy = self.metrics["accuracy"].compute(
            predictions=predictions, references=labels
        )
        precision = self.metrics["precision"].compute(
            predictions=predictions, references=labels, average="macro"
        )
        recall = self.metrics["recall"].compute(
            predictions=predictions, references=labels, average="macro"
        )
        f1 = self.metrics["f1"].compute(
            predictions=predictions, references=labels, average="macro"
        )
        return {
            "accuracy": accuracy["accuracy"],
            "precision": precision["precision"],
            "recall": recall["recall"],
            "f1": f1["f1"],
        }

    def prepare_dataset(self, x_train, y_train, x_val, y_val, x_test, y_test):

        train_data = {"text": x_train, "label": y_train}
        val_data = {"text": x_val, "label": y_val}
        test_data = {"text": x_test, "label": y_test}

        train_dataset = Dataset.from_dict(train_data)
        val_dataset = Dataset.from_dict(val_data)
        test_dataset = Dataset.from_dict(test_data)

        train_dataset = train_dataset.map(self.tokenize_function, batched=True)
        val_dataset = val_dataset.map(self.tokenize_function, batched=True)
        test_dataset = test_dataset.map(self.tokenize_function, batched=True)

        train_dataset.set_format(
            type="torch", columns=["input_ids", "attention_mask", "label"]
        )
        val_dataset.set_format(
            type="torch", columns=["input_ids", "attention_mask", "label"]
        )
        test_dataset.set_format(
            type="torch", columns=["input_ids", "attention_mask", "label"]
        )

        return train_dataset, val_dataset, test_dataset

    def tokenize_function(self, batch):
        return self.tokenizer(batch["text"], truncation=True, max_length=512)

    def fit(self, train, val):
        training_args = TrainingArguments(**self.parameters)
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train,
            eval_dataset=val,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
        )
        self.trainer.train()

    def predict(self, test_dataset):
        return self.trainer.predict(test_dataset)

    def save_model(self):
        self.trainer.save_model(config.MODELS_PATH.format(repr(self)))

    @classmethod
    def load_model(cls):
        # TODO: Implement loading the model
        pass

    def save_results(self, test_data, **kwargs):
        result = self.predict(test_data)
        predictions = np.argmax(result[0], axis=1)
        targets = result[1]
        report = classification_report(targets, predictions)
        directory = config.RESULTS_DIRECTORY.format(repr(self))
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, "metrics.txt")
        predictions = np.vectorize(config.id2label.get)(predictions)    
        targets = np.vectorize(config.id2label.get)(targets)
        with open(path, "w") as file:
            file.write(report)
        plotting.plot_confusion_matrix(
            targets, predictions, path=os.path.join(directory, "confusion_matrix.png")
        )

    def __repr__(self):
        return self.checkpoint.split("/")[-1]
