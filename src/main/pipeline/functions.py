""" Functions Module. """

from unicodedata import normalize
from keras.utils import pad_sequences
from config import config
import re
import string


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


def tokenize(corpus, parallel_mode=False):
    tokenizer = config.tokenizer
    tokenizer.fit_on_texts(corpus)
    sequences = tokenizer.texts_to_sequences(corpus)
    return pad_sequences(sequences, maxlen=40)


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
