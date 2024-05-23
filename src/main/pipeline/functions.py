""" Functions Module.
The functions inside this module are used in the pipeline on a data stream.
Every function should take a single argument and return a single value.
There are two types of functions:
- Parallelizable functions: these kind of functions have the default value for the parallel_mode parameter to be true.
                            moreover they must take a single string object as input and return a single string object
                            as output.
- Not parallelizable functions: these kind of functions have the default value for the parallel_mode parameter to be
                            false. Conversely, they must take a single list of strings as input and return a single
                            list of strings as output.
The second parameter (parallel_mode) is only inspected at run_time by the Pipeline class, but not used.
In the end of the file there are some general functions that are not strictly part of the automatic pipeline.
"""

import os
from typing import List
from unicodedata import normalize

import scipy
from sklearn.feature_selection import SelectKBest, f_classif

import numpy as np
from keras.utils import pad_sequences
from config import config
import re
import contractions
import pickle


# region Parallelizable functions
def clean_text(text: str, parallel_mode=True) -> str:
    """
    Remove all the unwanted elements from a string.
    Example: "'This' is > a test. string\u2014" -> "this is a test string"

    :param text: The text to clean, as a string.
    :param parallel_mode: Default is True, ignored
    :return: A string with the unwanted elements removed.
    """
    text = text.lower()
    # remove square brackets
    text = re.sub(r"\[|\]", "", text)
    # remove (VIDEO), (PHOTOS) or combinations of them
    pattern = r"\(PHOTOS\) | \(VIDEO\ | \(VIDEO, PHOTOS\) | \(PHOTOS, VIDEO\)"
    text = re.sub(pattern, "", text)
    # remove punctuation
    text = re.sub(r"[.,;:\-_!?^#\"]", " ", text)
    # remove angle brackets
    text = re.sub(r"[<>]", "", text)
    # remove ' in the text and substitute with a space
    text = re.sub(r"'", " ", text)
    # remove newlines
    text = re.sub("\n", "", text)
    # remove special characters (example \u2014)
    text = normalize("NFKD", text).encode("ascii", "ignore")
    return text.decode("utf-8")


def remove_contractions(text: str, parallel_mode=True) -> str:
    """
    Remove contracted form from corpus using the 'contractions' library.
    Example: "I'm the avocado's king" -> "I am the avocado's king"

    :param text: The sentence to remove contractions from, as a string.
    :param parallel_mode: Default is True, ignored
    :return: The text without contractions, as a string.
    """
    if isinstance(text, bytes):
        text = text.decode("utf-8")
    words = text.split(" ")
    filtered_words = [contractions.fix(str(word)) for word in words]
    return " ".join(filtered_words)


def stop_words_removal(text: str, parallel_mode=True) -> str:
    """
    Remove the stop words from the text using the english stop words list from nltk.
    See https://www.nltk.org/search.html?q=stopwords for more information.
    Example: "this is a test" -> "test"

    :param text: The text to remove the stop words from, as a string.
    :param parallel_mode: Default is True, ignored
    :return: The text with the stop words removed, as a string.
    """
    stop_words = config.stop_words
    if isinstance(text, bytes):
        text = text.decode("utf-8")
    words = text.split(" ")
    words = [word for word in words if word not in stop_words]
    return " ".join(words)


def lemmatization(text: str, parallel_mode=True) -> str:
    """
    Lemmatize the text using spaCy.
    See https://spacy.io/usage/linguistic-features#lemmatization for more information.
    Example: "I am a test" -> "I be a test"

    :param text: The text to lemmatize.
    :param parallel_mode: Default is True, ignored
    :return: The lemmatized text, as a string.
    """

    doc = config.nlp(text)
    return " ".join([token.lemma_ for token in doc])


def stemming(text: str, parallel_mode=True) -> str:
    """
    Stem the text using the nltk SnowballStemmer.
    See https://www.nltk.org/howto/stem.html for more information.
    Example: "I am an apple" -> "i am an appl"

    :param text: The text to stem as a string.
    :param parallel_mode: Default is True, ignored
    :return: The stemmed text, as a string.
    """
    res = " ".join(config.stemmer.stem(word) for word in text.split(" "))
    return res


def unify_numbers(text: str, parallel_mode=True) -> str:
    """
    Unify all numbers in the text to a single token.
    Example: "I have 2 apples and 3 bananas" -> "I have NUM apples and NUM bananas"

    :param text: The text to unify the numbers in, as a string.
    :param parallel_mode: Default is True, ignored
    :return: The text with the numbers unified, as a string.
    """
    return re.sub(r"\b\d+\b", config.numbers_token, text)


# endregion


# region Not parallelizable functions


def tokenize(corpus: List[str], parallel_mode=False) -> np.ndarray:
    """
    Tokenize the corpus using the Tokenizer from keras. After the tokenization, the word_index and num_words are updated
    in the config file. The return value is the padded sequences with a predefined length (config).
    See https://keras.io/api/keras_nlp/base_classes/tokenizer/ for more information.
    Example: ["I am a test", "This is another test"] -> [[1, 2, 3, 4], [5, 6, 7, 4]]

    :param corpus: A numpy array of strings to tokenize.
    :param parallel_mode: Default is False, ignored
    :return: Tokenization and padding of the corpus in a numpy array.
    """
    tokenizer = config.tokenizer
    tokenizer.fit_on_texts(corpus)
    word_index = tokenizer.word_index
    # save the word index
    path = os.path.join(config.PIPELINE_DATASET_PATH, "word_index.pkl")
    with open(path, "wb") as f:
        pickle.dump(word_index, f)

    config.num_words = len(word_index) + 1
    sequences = tokenizer.texts_to_sequences(corpus)
    return pad_sequences(sequences, maxlen=config.MAX_SEQ_LENGTH)


def tfidf_vectorizer(corpus: List[str], parallel_mode=False) -> scipy.sparse.csr_matrix:
    """
    Vectorize the text using the TfidfVectorizer.
    See https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html for more
    information.
    Example: ["I am a test", "This is another test"] -> result is of type scipy.sparse.csr_matrix

    :param corpus: The text to vectorize.
    :param parallel_mode: A boolean indicating whether to run the function in parallel.
    :return: The vectorized text.
    """

    return config.vectorizer.fit_transform(corpus)


def count_vectorizer(corpus: List[str], parallel_mode=False) -> scipy.sparse.csr_matrix:
    """
    Vectorize the text using the CountVectorizer.

    :param corpus: The text to vectorize as an iterable of strings.
    :param parallel_mode: Default is False, ignored
    :return: The vectorized text in a sparse matrix of integers.
    """
    return config.count_vectorizer.fit_transform(corpus)


# endregion


# region General functions (beyond Pipeline)
def select_features(corpus, targets, k) -> np.ndarray:
    """
    Select the features from the text.

    :param corpus: The text to select the features from in an array-like format.
    :param targets: The targets to select the features from in an array-like format.
    :param k: The number of features to select.
    :return: The selected features.
    """
    selector = SelectKBest(f_classif, k=k)
    return selector.fit_transform(corpus, targets)


def tfidf_transformer(corpus: np.ndarray) -> np.ndarray:
    """
    Transform the data using the TfidfTransformer from sklearn.
    See https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html for more
    information.
    Example: ["I am a test", "This is another test"] -> result is of type np.ndarray

    :param corpus: input samples in an array like format of shape (n_samples, n_features)
    :return: The transformed text in a numpy ndarray format.
    """

    return config.transformer.fit_transform(corpus)


# endregion
