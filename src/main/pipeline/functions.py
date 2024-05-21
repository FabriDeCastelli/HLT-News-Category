""" Functions Module. """

import numpy as np
from unicodedata import normalize
from keras.utils import pad_sequences
from config import config
import re
import string
import contractions


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
    # remove (VIDEO), (PHOTOS) or combinations of them
    pattern = r"\(PHOTOS\) | \(VIDEO\ | \(VIDEO, PHOTOS\) | \(PHOTOS, VIDEO\)"
    text = re.sub(pattern, "", text)
    # remove punctuation
    text = re.sub(r"[.,;:!?^#\"]", "", text)
    # remove angle brackets
    text = re.sub(r"[<>]", "", text)
    # remove ' in the text
    text = re.sub(r"'(\w+)'", r"\1", text)
    # remove newlines
    text = re.sub("\n", "", text)
    # remove special characters (example \u2014)
    text = normalize("NFKD", text).encode("ascii", "ignore")
    return text.decode("utf-8")


def remove_contractions(corpus, parallel_mode=True) -> str:
    """
    Remove contracted form from corpus

    :param corpus: the sentence to remove contractions from
    :return: the cleaned sentence
    """
    if isinstance(corpus, bytes):
        corpus = corpus.decode("utf-8")
    words = corpus.split(" ")
    filtered_words = [contractions.fix(str(word)) for word in words]
    return " ".join(filtered_words)


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


def tokenize(corpus, parallel_mode=False):
    tokenizer = config.tokenizer
    tokenizer.fit_on_texts(corpus)
    config.word_index = tokenizer.word_index
    config.num_words = len(config.word_index) + 1
    sequences = tokenizer.texts_to_sequences(corpus)
    return pad_sequences(sequences, maxlen=config.MAX_SEQ_LENGHT)


def select_features(corpus, targets, k, parallel_mode=False):
    """
    Select the features from the text.

    :param corpus: The text to select the features from.
    :param parallel_mode: A boolean indicating whether to run the function in parallel.
    :return: The selected features.
    """
    from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif

    selector = SelectKBest(f_classif, k=k)
    return selector.fit_transform(corpus, targets)


def tfidf_vectorizer(corpus, parallel_mode=False):
    """
    Vectorize the text using the TfidfVectorizer.

    :param corpus: The text to vectorize.
    :param parallel_mode: A boolean indicating whether to run the function in parallel.
    :return: The vectorized text.
    """

    return config.vectorizer.fit_transform(corpus)


def tfidf_transformer(corpus, parallel_mode=False):
    """
    Transform the data using the TfidfTransformer.

    :param corpus: The text to transform.
    :param parallel_mode: A boolean indicating whether to run the function in parallel.
    :return: The transformed text.
    """

    return config.transformer.fit_transform(corpus)


def count_vectorizer(corpus, parallel_mode=False):
    """
    Vectorize the text using the CountVectorizer.

    :param corpus: The text to vectorize.
    :param parallel_mode: A boolean indicating whether to run the function in parallel.
    :return: The vectorized text.
    """

    return config.count_vectorizer.fit_transform(corpus)
