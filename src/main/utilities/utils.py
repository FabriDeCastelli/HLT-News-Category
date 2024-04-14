""" This module contains utility functions for the project. """

import nltk
import numpy as np
import pandas as pd
import spacy
from nltk.stem import PorterStemmer
from nltk.tokenize import casual_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

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


def set_up():
    """
    Set up the environment.
    """
    nltk.download("punkt")


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
    return df


def get_dataset(filepath=DATASET_PATH, remove_target=False):
    """
    Read the dataset from the given filepath and return the dataframe.

    :param filepath: The path to the dataset file.
    :return: The dataframe containing the dataset.
    """
    df = pd.read_json(filepath, lines=True)
    df = label_renaming(df)
    df = df[df["category"].isin(new_names)]
    df = clean(df)
    if remove_target:
        df = df.drop(columns=["category"])
    return df


def case_folding(corpus):
    """
    Convert the text to lowercase.
    """
    return corpus.lower()


def lemmatization(corpus):
    """
    Lemmatize the text.
    lemmatizer = WordNetLemmatizer()
    return lemmatizer.lemmatize(corpus)
    """
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(corpus)
    return " ".join([token.lemma_ for token in doc])


def stemming(corpus):
    """
    Stem the text.
    """
    stemmer = PorterStemmer()
    return stemmer.stem(corpus)


def casual_tokenizer(corpus):
    """
    Tokenize the text using the casual_tokenize function from NLTK.
    """
    return casual_tokenize(corpus)


def tfidf_vectorizer(corpus):
    """
    Vectorize the text using TF-IDF.
    """
    vectorizer = TfidfVectorizer(
        max_features=60000,
        lowercase=False,
        analyzer="word",
        tokenizer=word_tokenize,
        ngram_range=(1, 3),
        dtype=np.float32,
    )
    return vectorizer.fit_transform(corpus)
