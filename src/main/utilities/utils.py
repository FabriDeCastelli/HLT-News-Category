""" This module contains utility functions for the project. """

import nltk
import pandas as pd
import re
import string
from config.config import nlp, vectorizer, stemmer
from nltk.tokenize import casual_tokenize
from unicodedata import normalize

from config.config import (
    DATASET_PATH,
    drop_column,
    entertainment,
    life,
    new_names,
    politics,
    sports,
    voices,
)


def label_renaming(df):
    """
    Rename the labels in the dataset to the new names.

    :param df: The dataframe containing the dataset.
    :return: The dataframe with the labels renamed.
    """
    di = {}
    for new, orig in zip(new_names, [life, entertainment, voices, sports, politics]):
        for label in orig:
            di[label] = new
    return df.replace({"category": di})


def clean(df):
    """
    Clean the dataset by dropping columns and removing duplicates.

    :param df: The dataframe containing the dataset.
    :return: The cleaned dataframe.
    """
    df.drop(labels=drop_column, inplace=True, axis=1)
    df = df[df["short_description"] != ""]
    df = df.drop_duplicates(subset="short_description")
    df = df.drop_duplicates(subset="headline")
    df = df[~df["short_description"].str.contains("https")]
    return df


def get_dataset(filepath=DATASET_PATH, remove_target=False):
    """
    Read the dataset from the given filepath and return the dataframe.

    :param filepath: The path to the dataset file.
    :param remove_target: A boolean indicating whether to remove the target column.
    :return: The dataframe containing the dataset.
    """
    df = pd.read_json(filepath, lines=True)
    df = label_renaming(df)
    df = df[df["category"].isin(new_names)]
    df = clean(df)
    if remove_target:
        df = df.drop(columns=["category"])
    return df


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
    stop_words = set(nltk.corpus.stopwords.words("english"))
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

    doc = nlp(corpus)
    return " ".join([token.lemma_ for token in doc])


def stemming(corpus, parallel_mode=True):
    """
    Stem the text.
    """
    res = " ".join(stemmer.stem(word) for word in corpus.split(" "))
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
    if not isinstance(corpus, list):
        corpus = [corpus]
    return vectorizer.fit_transform(corpus)
