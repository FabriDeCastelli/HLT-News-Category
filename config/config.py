""" Configuration file for the project. """

import os

import nltk
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize

# PROJECT FOLDER PATH
PROJECT_FOLDER_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# DATASET PATH
DATASET_PATH = PROJECT_FOLDER_PATH + "/dataset/News_Category_Dataset.json"

#MODELS PATH
MODELS_PATH = PROJECT_FOLDER_PATH + "/models/"

PIPELINE_DATASET_PATH = PROJECT_FOLDER_PATH + "/dataset/preprocessing/{}"

# RENAME CATEGORIES
life = ["WELLNESS", "TRAVEL", "STYLE & BEAUTY", "FOOD & DRINK"]
entertainment = ["ENTERTAINMENT", "COMEDY"]
voices = ["LATINO VOICES", "BLACK VOICES", "QUEER VOICES"]
sports = ["SPORTS"]
politics = ["POLITICS"]
new_names = ["Life", "Entertainment", "Voices", "Sports", "Politics"]
drop_column = ["link", "authors", "date"]
merged_categories = [life, entertainment, voices, sports, politics]
rename_y = {'Entertainment': 0, 'Life': 1, 'Politics': 2, 'Sport': 3, 'Voices': 3}

nltk.download("stopwords")
nlp = spacy.load("en_core_web_sm")
stemmer = nltk.SnowballStemmer("english")
vectorizer = TfidfVectorizer(
    # tokenizer=word_tokenize,
    # stop_words="english",
)
