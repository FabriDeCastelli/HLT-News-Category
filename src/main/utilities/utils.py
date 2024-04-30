""" This module contains utility functions for the project. """

import os.path

import nltk
import pandas as pd
import re
import string

from scipy.sparse import isspmatrix_csr, save_npz, load_npz
from config import config
from nltk.tokenize import casual_tokenize
from unicodedata import normalize


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
    elif ".json" in filepath:
        pd.DataFrame(results).to_json(filepath)
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


def clean_text(corpus, parallel_mode=True) -> str:
    """
    Clean the text in the dataframe.

    :param corpus: The text to clean.
    :param parallel_mode: A boolean indicating whether to run the function in parallel.
    :return: The dataframe with the text cleaned.
    """
    text = corpus.lower()
    # remove square brackets
    text = re.sub(r"\[|\]", "", text)
    # remove special characters
    text = re.sub(r"[^\w\s]", "", text)
    # remove all punctuations
    text = re.sub("[%s]" % re.escape(string.punctuation), "", text)
    # remove angle brackets
    text = re.sub(r"<(.*?)>", r"\1", text)
    # remove newlines
    text = re.sub("\n", "", text)
    # remove special characters (example \u2014)
    text = normalize("NFKD", text).encode("ascii", "ignore")
    return text


def stop_words_removal(corpus, parallel_mode=True) -> str:
    """
    Remove the stop words from the text.

    :param corpus: The text to remove the stop words from.
    :param parallel_mode: A boolean indicating whether to run the function in parallel.
    :return: The text with the stop words removed, as a string.
    """
    stop_words = config.stop_words
    if isinstance(corpus, bytes):
        corpus = corpus.decode("utf-8")
    words = corpus.split(" ")
    words = [word for word in words if word not in stop_words]
    return " ".join(words)


def lemmatization(corpus, parallel_mode=True) -> str:
    """
    Lemmatize the text.

    :param corpus: The text to lemmatize.
    :param parallel_mode: A boolean indicating whether to run the function in parallel.
    :return: The lemmatized text, as a string.
    """

    doc = config.nlp(corpus)
    return " ".join([token.lemma_ for token in doc])


def stemming(corpus, parallel_mode=True):
    """
    Stem the text.
    """
    res = " ".join(config.stemmer.stem(word) for word in corpus.split(" "))
    return res


def casual_tokenizer(corpus, parallel_mode=False):
    """
    Tokenize the text using the casual_tokenize function from NLTK.
    """
    return casual_tokenize(corpus)


def tfidf_vectorizer(corpus, parallel_mode=False):
    """
    Vectorize the text using the TfidfVectorizer.

    :param corpus: The text to vectorize.
    :param parallel_mode: A boolean indicating whether to run the function in parallel.
    :return: The vectorized text.
    """

    return config.vectorizer.fit_transform(corpus)


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
    if ".json" in filepath:
        return pd.read_json(filepath)
    else:
        raise ValueError(f"The given path {filepath} is not a valid path.")
