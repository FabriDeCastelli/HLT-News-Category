""" This module contains utility functions for the project. """

import os
import pickle
import numpy as np
import pandas as pd
import yaml
from scipy.sparse import save_npz, load_npz
from gensim.models import KeyedVectors
from config import config
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
import keras


def get_dataset(
    filepath=config.DATASET_PATH, one_hot=False
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reads the whole dataset from the given filepath, performs some kind of cleaning and returns the inputs and targets
    in a numpy format. The cleaning is performed by the 'clean' function and the 'label_renaming' function.

    :param one_hot: A boolean indicating whether to one hot encode the labels.
    :param filepath: The path to the dataset file.
    :return: Inputs and targets
    """
    df = pd.read_json(filepath, lines=True)
    df = label_renaming(df)
    df = df[df["category"].isin(config.new_names)]
    df = clean(df)
    # filter out articles with less than 10 words
    df = df[df["full_article"].apply(lambda x: len(x.split()) > 10)]
    targets = df["category"]
    if one_hot:
        targets = targets.map(config.label2id)
        targets = targets.apply(lambda x: np.eye(5)[x])
        targets = np.array(targets.to_list())
    else:
        targets = targets.to_numpy()
    return df.drop(columns=["category"]).to_numpy(), targets


def label_renaming(df):
    """
    Rename the labels in the dataset to the new names, according to names defined in a configuration file.

    :param df: The dataframe containing the dataset.
    :return: The dataframe with the labels renamed.
    """
    di = {}
    for new, orig in zip(config.new_names, config.merged_categories):
        for label in orig:
            di[label] = new
    return df.replace({"category": di})


def clean(df, merge=True):
    """
    Clean the dataset by dropping columns and removing duplicates.

    :param df: The dataframe containing the dataset.
    :param merge: A boolean indicating whether to merge short_description and headline.
    :return: The cleaned dataframe.
    """
    df.drop(labels=config.drop_column, inplace=True, axis=1)
    df = df[df["short_description"] != ""]
    df = df.drop_duplicates(subset="short_description")
    df = df.drop_duplicates(subset="headline")
    df = df[~df["short_description"].str.contains("https")]
    if merge:
        df["full_article"] = df["headline"] + " " + df["short_description"]
        df = df.drop(columns=["headline", "short_description"])
    return df


def save_preprocessing(results, model_file):
    """
    Save the preprocessing results to a file.

    :param results: The results to save.
    :param model_file: The model to save the results for.
    """
    assert model_file is not None, "Model (filepath) is not provided."

    filepath = os.path.join(config.PIPELINE_DATASET_PATH, model_file)

    if not os.path.exists(config.PIPELINE_DATASET_PATH):
        os.makedirs(config.PIPELINE_DATASET_PATH)

    if ".npz" in filepath:
        save_npz(filepath, results)
    elif ".npy" in filepath:
        np.save(filepath, results, allow_pickle=True)
    else:
        raise ValueError(f"File extension of {filepath} is not supported for saving.")


def load_preprocessing(model_file):
    """
    Load the preprocessing results from a file.

    :param model_file: the file path that includes the model name.
    :return: The results from the file.
    """
    assert model_file is not None, "Model (filepath) is not provided."
    filepath = os.path.join(config.PIPELINE_DATASET_PATH, model_file)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"The file {filepath} does not exist.")
    if ".npz" in filepath:
        return load_npz(filepath)
    if ".npy" in filepath:
        return np.load(filepath, allow_pickle=True)
    else:
        raise ValueError(f"The given path {filepath} is not a valid path.")


def read_yaml(path):
    """
    Reads a file in .yaml format. Used to read the hyperparameters to tune models.

    :param path: the path of the file to read
    :return: the dictionary representation of the file
    """
    with open(path, "r") as file:
        dictionary = yaml.load(file, Loader=yaml.FullLoader)

    return dictionary


def split_train_val_test(
    inputs, targets, validation_size=0.2, test_size=0.1, random_state=42
):
    """
    Split the dataset into training, validation, and test sets. First, it splits the dataset into training and test
    sets, then it splits the training set into training and validation sets. The splits that are passed as
    hyperparameters are coherent with the final actual splits sizes.

    :param inputs: The input data (features).
    :param targets: The target data (labels).
    :param validation_size: The size of the validation set.
    :param test_size: The size of the test set.
    :param random_state: The random state for reproducibility.
    :return: 6 numpy arrays: x_train, x_val, x_test, y_train, y_val, y_test
    """

    if validation_size + test_size >= 1:
        raise ValueError(
            "The sum of validation_size and test_size must be less than 1."
        )

    # Split the dataset into training and test sets
    x_train, x_test, y_train, y_test = train_test_split(
        inputs,
        targets,
        test_size=test_size,
        random_state=random_state,
        stratify=targets,
    )
    # Adjust the validation size to be relative to the training size
    validation_size = validation_size / (1 - test_size)
    # Split the training set into training and validation sets
    x_train, x_val, y_train, y_val = train_test_split(
        x_train,
        y_train,
        test_size=validation_size,
        random_state=random_state,
        stratify=y_train,
    )

    return x_train, x_val, x_test, y_train, y_val, y_test


def load_pretrained_embeddings(embedding) -> dict | KeyedVectors:
    """
    Load the pretrained embeddings from the given path.
    If the embeddings name is not found, it raises a ValueError.
    If it is "glove" the result is a dictionary, otherwise it is a KeyedVectors object.

    :param embedding: The type of the embeddings.
    :return: The pretrained embeddings in a dictionary format.
    """
    if embedding == "google":
        return KeyedVectors.load_word2vec_format(config.GOOGLE_PATH, binary=True)
    elif embedding == "fastText":
        return KeyedVectors.load_word2vec_format(config.FASTTEXT_PATH)
    elif embedding == "glove":
        glove_embeddings = {}
        with open(config.GLOVE_PATH) as f:
            for line in f:
                word, coefficients = line.split(maxsplit=1)
                coefficients = np.fromstring(coefficients, "f", sep=" ")
                glove_embeddings[word] = coefficients
        return glove_embeddings
    raise ValueError(f"Embedding {embedding} is not supported.")


def create_embedding_matrix(pretrained_embeddings: dict | KeyedVectors) -> np.ndarray:
    """
    Creates the embedding matrix from the word embeddings. The returned embedding matrix is a numpy ndarray of shape
    (NUM_WORDS, EMBEDDING_DIM), where NUM_WORDS is the number of words in the dataset and EMBEDDING_DIM is the
    dimension of the embeddings. It is built using a word index that is saved in the pipeline dataset folder: if not
    provided it raises a FileNotFoundError.

    :param pretrained_embeddings: can be obtained with the load_pretrained_embeddings function.
    :return: The embedding matrix in a numpy ndarray format, of shape (NUM_WORDS, EMBEDDING_DIM).
    """

    path = os.path.join(config.PIPELINE_DATASET_PATH, "word_index.pkl")
    word_index = pickle.load(open(path, "rb"))

    embedding_matrix = np.zeros((config.NUM_WORDS, config.EMBEDDING_DIM))
    for word, i in word_index.items():
        if word in pretrained_embeddings:
            embedding_matrix[i] = pretrained_embeddings[word]
        elif word == "[UNK]":
            embedding_matrix[i] = np.zeros(config.EMBEDDING_DIM)
        elif word == "[NUM]":
            embedding_matrix[i] = np.ones(config.EMBEDDING_DIM)
    return embedding_matrix


def embedding_matrix_statistics(pretrained_embeddings):
    """
    Gets the percentage of words found in the pretrained embeddings. and returns all unmatched words.

    :param pretrained_embeddings: The pretrained embeddings.
    :return: The statistics of the embedding matrix.
    """
    found = 0
    not_found = 0
    unmatched_words = []
    path = os.path.join(config.PIPELINE_DATASET_PATH, "word_index.pkl")
    word_index = pickle.load(open(path, "rb"))
    for word, i in word_index.items():
        if word in pretrained_embeddings:
            if i <= 1e9:
                found += 1
        else:
            if i <= 1e9:
                not_found += 1
            if word not in unmatched_words:
                unmatched_words.append(word)
    return found / (found + not_found), unmatched_words


# All the functions below are used to calculate the macro average of precision, recall and f1 score, which are not
# available in the keras library. They are registered as keras serializable metrics to be used in the model evaluation.


# region keras metrics
@keras.saving.register_keras_serializable()
def precision_macro(y_true, y_pred):
    y_pred = K.round(y_pred)
    y_true = K.cast(y_true, "float")
    y_pred = K.cast(y_pred, "float")
    tp = K.sum(K.cast(y_true * y_pred, "float"), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, "float"), axis=0)

    precision = tp / (tp + fp + K.epsilon())
    return K.mean(precision)


@keras.saving.register_keras_serializable()
def recall_macro(y_true, y_pred):
    y_pred = K.round(y_pred)
    y_true = K.cast(y_true, "float")
    y_pred = K.cast(y_pred, "float")
    tp = K.sum(K.cast(y_true * y_pred, "float"), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), "float"), axis=0)

    recall = tp / (tp + fn + K.epsilon())
    return K.mean(recall)


@keras.saving.register_keras_serializable()
def f1_macro(y_true, y_pred):
    precision = precision_macro(y_true, y_pred)
    recall = recall_macro(y_true, y_pred)

    f1 = 2 * precision * recall / (precision + recall + K.epsilon())
    return f1


# endregion
