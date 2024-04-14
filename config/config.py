""" Configuration file for the project. """

import os

# PROJECT FOLDER PATH
PROJECT_FOLDER_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# DATASET PATH
DATASET_PATH = PROJECT_FOLDER_PATH + "/dataset/News_Category_Dataset.json"

LOGISTIC_PIPELINE_DATASET_PATH = (
    PROJECT_FOLDER_PATH + "/dataset/logistic/logistic_pipeline_dataset.csv"
)

# RENAME CATEGORIES
life = ["WELLNESS", "TRAVEL", "STYLE & BEAUTY", "FOOD & DRINK"]
entertainment = ["ENTERTAINMENT", "COMEDY"]
voices = ["LATINO VOICES", "BLACK VOICES", "QUEER VOICES"]
sports = ["SPORTS"]
politics = ["POLITICS"]
new_names = ["Life", "Entertainment", "Voices", "Sports", "Politics"]
drop_column = ["link", "authors", "date"]
