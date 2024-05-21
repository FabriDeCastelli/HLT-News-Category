import unittest

import numpy as np
import scipy

import src.main.pipeline.functions as functions
import config.config as config


class TestFunctions(unittest.TestCase):

    # region Parallelizable functions
    def test_clean_text2(self):
        string = "I 'This' is. a test's. Pippo: > plito mario\u2014mario"
        result = functions.clean_text(string)
        expected = "i this is a test's pippo  plito mariomario"
        self.assertEqual(result, expected)

    def test_contractions(self):
        string = "I'm the avocado's king"
        result = functions.remove_contractions(string)
        expected = "I am the avocado's king"
        self.assertEqual(result, expected)

    def test_stop_words_removal(self):
        string = "I am an avocado with a lot of seeds"
        result = functions.stop_words_removal(string)
        expected = "I avocado lot seeds"
        self.assertEqual(result, expected)

    def test_lemmatization(self):
        string = "I am an avocado with a lot of seeds"
        result = functions.lemmatization(string)
        expected = "I be an avocado with a lot of seed"
        self.assertEqual(result, expected)

    def test_stemmer(self):
        string = "I am an apple"
        result = functions.stemming(string)
        expected = "i am an appl"
        self.assertEqual(result, expected)

    # endregion

    # region Non-parallelizable functions: only return types and shapes are tested
    def test_keras_tokenizer_config_update(self):
        corpus = ["I am an apple", "I am a banana"]
        tokenizer = functions.tokenize(corpus)
        self.assertEqual(
            config.word_index["apple"], 4
        )  # apple is the 4th word in the vocabulary
        self.assertEqual(tokenizer.shape, (len(corpus), config.MAX_SEQ_LENGTH))
        self.assertIsInstance(tokenizer, np.ndarray)

    def test_tfidf_vectorizer(self):
        corpus = ["I am an apple", "I am a banana"]
        vectorizer = functions.tfidf_vectorizer(corpus)
        self.assertIsInstance(vectorizer, scipy.sparse.csr_matrix)

    def test_count_vectorizer(self):
        corpus = ["I am an apple", "I am a banana"]
        vectorizer = functions.count_vectorizer(corpus)
        self.assertIsInstance(vectorizer, scipy.sparse.csr_matrix)
