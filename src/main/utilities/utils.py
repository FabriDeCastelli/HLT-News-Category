""" This module contains utility functions for the project. """

import os.path
import numpy as np
import pandas as pd
import yaml
from scipy.sparse import save_npz, load_npz
from gensim.models import KeyedVectors
from config import config
from sklearn.model_selection import train_test_split

def save_preprocessing(results, model_file):
    """
    Save the preprocessing results to a file.

    :param results: The results to save.
    :param model_file: The model to save the results for.
    """
    assert model_file is not None, "Model (filepath) is not provided."
    assert isinstance(results, dict), "Results is not a dictionary."

    filepath = os.path.join(config.PIPELINE_DATASET_PATH, model_file)

    if not os.path.exists(config.PIPELINE_DATASET_PATH):
        os.makedirs(config.PIPELINE_DATASET_PATH)

    if ".npz" in filepath:
        save_npz(filepath, results["full_article"])
    elif ".npy" in filepath:
        np.save(filepath, results["full_article"], allow_pickle=True)
    else:
        raise ValueError(f"File extension of {filepath} is not supported for saving.")


def label_renaming(df):
    """
    Rename the labels in the dataset to the new names.

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


def get_dataset(filepath=config.DATASET_PATH, one_hot=False):
    """
    Read the dataset from the given filepath and return the dataframe.

    :param one_hot: A boolean indicating whether to one hot encode the labels.
    :param filepath: The path to the dataset file.
    :return: The dataframe containing the dataset.
    """
    df = pd.read_json(filepath, lines=True)
    df = label_renaming(df)
    df = df[df["category"].isin(config.new_names)]
    df = clean(df)
    targets = df["category"]
    if one_hot:
        targets = pd.get_dummies(targets)
    return df.drop(columns=["category"]), targets


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
    Reads a file in .yaml format.

    :param path: the path of the file to read
    :return: the dictionary contained in the file
    """
    with open(path, "r") as file:
        dictionary = yaml.load(file, Loader=yaml.FullLoader)

    return dictionary

def split_train_val_test(inputs, targets, validation_size=0.2, test_size=0.1, random_state=42):
    """
    Split the dataset into training, validation, and test sets.

    :param inputs: The input data.
    :param targets: The target data.
    :param validation_size: The size of the validation set.
    :param test_size: The size of the test set.
    :return: The split dataset.
    """

    if validation_size + test_size >= 1:
        raise ValueError("The sum of validation_size and test_size must be less than 1.")

    # Split the dataset into training and test sets
    x_train, x_test, y_train, y_test = train_test_split(inputs, targets, test_size=test_size, random_state=random_state, stratify=targets)
    # Adjust the validation size to be relative to the training size
    validation_size = validation_size / (1 - test_size)
    # Split the training set into training and validation sets
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=validation_size, random_state=random_state, stratify=y_train)

    return x_train, x_val, x_test, y_train, y_val, y_test

def load_pretrained_embedddings(embedding_path, embedding):
    """
    Load the pretrained embeddings from the given path.

    :param embedding_path: The path to the pretrained embeddings.
    :param embedding: The type of the embedding.
    :return: The pretrained embeddings.
    """
    if(embedding =="google"):
        return KeyedVectors.load_word2vec_format(embedding_path, binary=True)
    elif(embedding =="fastText"):
        return KeyedVectors.load_word2vec_format(embedding_path)
    elif(embedding =="glove"):
        glove_embeddings = {}
        with open(embedding_path) as f:
            for line in f:
                word, coefs = line.split(maxsplit=1)
                coefs = np.fromstring(coefs, "f", sep=" ")
                glove_embeddings[word] = coefs
        return glove_embeddings

def create_embedding_matrix(pretrained_embeddings):
    """
    Create the embedding matrix from the word embeddings.
    :param word_index: The word index.
    :param pretrained_embeddings: The pretrained embeddings.
    :return: The embedding matrix.
    """
    find = 0
    not_find = 0
    unmached_words = []
    for word, i in config.word_index.items():
        if word in pretrained_embeddings:
            embedding_vector = pretrained_embeddings[word]
            config.embedding_matrix[i] = embedding_vector
            find += 1
        else:
            config.embedding_matrix[i] = np.random.normal(0, 1, config.EMBEDDING_DIM)
            not_find += 1
            if word not in unmached_words:
                unmached_words.append(word)
    return find/(find+not_find), unmached_words
