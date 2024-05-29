""" Configuration file for the project. """

import os
import nltk
from nltk.corpus import stopwords
import spacy
from sklearn.feature_extraction.text import (
    TfidfVectorizer,
    CountVectorizer,
    TfidfTransformer,
)
from tensorflow.keras.preprocessing.text import Tokenizer

# PROJECT FOLDER PATH
PROJECT_FOLDER_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# DATASET PATH
DATASET_PATH = os.path.join(
    PROJECT_FOLDER_PATH, "dataset", "News_Category_Dataset.json"
)

# PRETRAINED EMBEDDINGS PATH
EMBEDDINGS_PATH = os.path.join(PROJECT_FOLDER_PATH, "embeddings")

# HYPERPARAMETERS PATH
HYPERPARAMETERS_PATH = os.path.join(PROJECT_FOLDER_PATH, "hyperparameters", "{}.yaml")

# RESULT PATH
RESULTS_DIRECTORY = os.path.join(PROJECT_FOLDER_PATH, "results", "{}")

# MODELS PATH
MODELS_PATH = os.path.join(PROJECT_FOLDER_PATH, "models")

# PIPELINE PATH
PIPELINE_DATASET_PATH = os.path.join(PROJECT_FOLDER_PATH, "dataset", "preprocessing")
os.makedirs(PIPELINE_DATASET_PATH, exist_ok=True)

# TENSORBOARD PATH
LOGS_PATH = os.path.join(PROJECT_FOLDER_PATH, "logs", "{}")

# RENAME CATEGORIES
life = ["WELLNESS", "TRAVEL", "STYLE & BEAUTY", "FOOD & DRINK"]
entertainment = ["ENTERTAINMENT", "COMEDY"]
voices = ["LATINO VOICES", "BLACK VOICES", "QUEER VOICES"]
sports = ["SPORTS"]
politics = ["POLITICS"]
new_names = ["Life", "Entertainment", "Voices", "Sports", "Politics"]
drop_column = ["link", "authors", "date"]
merged_categories = [life, entertainment, voices, sports, politics]
label2id = {"Politics": 0, "Voices": 1, "Sports": 2, "Entertainment": 3, "Life": 4}
id2label = {int(v): str(k) for k, v in label2id.items()}


# PRETRAINED EMBEDDINGS
glove_file = os.path.join(EMBEDDINGS_PATH, "glove.6B.300d.txt")
google_file = os.path.join(EMBEDDINGS_PATH, "GoogleNews-vectors-negative300.bin")
fastText_file = os.path.join(EMBEDDINGS_PATH, "wiki-news-300d-1M-subword.vec")

# PIPELINE STUFF
nltk.download("stopwords", quiet=True)
nlp = spacy.load("en_core_web_sm")
stemmer = nltk.SnowballStemmer("english")
vectorizer = TfidfVectorizer()
stop_words = set(stopwords.words("english"))
VOCAB_SIZE = 30000
EMBEDDING_DIM = 300
MAX_SEQ_LENGTH = 200
num_words = 0
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="[UNK]")
word_index = {}
count_vectorizer = CountVectorizer()
transformer = TfidfTransformer()

numbers_token = "[NUM]"
