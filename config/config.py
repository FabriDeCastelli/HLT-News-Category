""" Configuration file for the project. """

import os

import nltk
from nltk.corpus import stopwords
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer

# PROJECT FOLDER PATH
PROJECT_FOLDER_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# DATASET PATH
DATASET_PATH = os.path.join(
    PROJECT_FOLDER_PATH, "dataset", "News_Category_Dataset.json"
)

# HYPERPARAMETERS PATH
HYPERPARAMETERS_PATH = os.path.join(PROJECT_FOLDER_PATH, "hyperparameters", "{}.yaml")

# RESULT PATH
RESULTS_DIRECTORY = os.path.join(PROJECT_FOLDER_PATH, "results", "{}")

# MODELS PATH
MODELS_PATH = os.path.join(PROJECT_FOLDER_PATH, "models")

# PIPELINE PATH
PIPELINE_DATASET_PATH = os.path.join(PROJECT_FOLDER_PATH, "dataset", "preprocessing")

# TENSORBOARD PATH
TENSORBOARD_LOGS = os.path.join(PROJECT_FOLDER_PATH, "logs", "{}")

# RENAME CATEGORIES
life = ["WELLNESS", "TRAVEL", "STYLE & BEAUTY", "FOOD & DRINK"]
entertainment = ["ENTERTAINMENT", "COMEDY"]
voices = ["LATINO VOICES", "BLACK VOICES", "QUEER VOICES"]
sports = ["SPORTS"]
politics = ["POLITICS"]
new_names = ["Life", "Entertainment", "Voices", "Sports", "Politics"]
drop_column = ["link", "authors", "date"]
merged_categories = [life, entertainment, voices, sports, politics]
rename_y = {"Entertainment": 0, "Life": 1, "Politics": 2, "Sport": 3, "Voices": 4}

nltk.download("stopwords", quiet=True)
nlp = spacy.load("en_core_web_sm")
stemmer = nltk.SnowballStemmer("english")
vectorizer = TfidfVectorizer()
stop_words = set(stopwords.words("english"))
tokenizer = Tokenizer()
count_vectorizer = CountVectorizer()
