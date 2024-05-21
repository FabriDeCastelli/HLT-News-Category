import unittest
from src.main.pipeline.pipeline import Pipeline
from src.main.utilities.utils import get_dataset
from src.main.pipeline.functions import clean_text, remove_contractions
import pandas as pd


class TestFunctions(unittest.TestCase):
    def test_clean_text2(self):
        string = "I 'This' is. a test's. Pippo: > plito mario\u2014mario"
        result = clean_text(string)
        expected = "i this is a test's pippo  plito mariomario"
        self.assertEqual(result, expected)

    def test_contractions(self):
        string = "I'm going to the store and I'll get some milk russia's empire don't"
        result = remove_contractions(string)
        expected = "I am going to the store and I will get some milk russia's empire do not"
        self.assertEqual(result, expected)
