import numpy as np   # a useful datastructure
import pandas as pd  # for data preprocessing
import nltk
import re
import pickle

nltk.download('punkt')
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))

pos = pd.read_csv("files/positive_processed.csv", sep='delimiter', header=None, engine='python')
neg = neg = pd.read_csv("files/negative_processed.csv", sep='delimiter', header=None, engine='python')

pos['class'] = 1
neg['class'] = 0

df = pd.concat([pos, neg]).sample(frac=1).reset_index(drop = True)


def preprocess(review):
    #convert the tweet to lower case
    review.lower()
    #convert all urls to sting "URL"
    review = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',review)
    #convert all @username to "AT_USER"
    review = re.sub('@[^\s]+','AT_USER', review)
    #correct all multiple white spaces to a single white space
    review = re.sub('[\s]+', ' ', review)
    #convert "#topic" to just "topic"
    review = re.sub(r'#([^\s]+)', r'\1', review)
    tokens = word_tokenize(review)
    tokens = [w for w in tokens if not w in stop_words]
    return " ".join(tokens)


def load_file(filepath):
    """
    This function is used to load a file from the specified file path
    This was used to load the mapping dictionaries for this script
    Parameters
    ----------
    filepath: str

    Returns
    Any file
    -------

    """

    with open(filepath, 'rb') as f:
        file = pickle.load(f)
        return file


def save_file(filepath, data):
    """
    This function is used to save picklfiles
    Args:
        filepath: This is the location where it will be saved
        data: This is the data that you want to save
    Returns:
        None
    """
    pickle.dump(data, open(filepath, "wb"))



# print(df.head())

# Separate the X and y values
y = df["class"]
X = df[0]
# Apply the created function on this value
X = X.apply(preprocess)

twitter_data = pd.DataFrame()
twitter_data["clean_text"] = X
twitter_data["class"] = y

save_file("files/twitter_data.pkl", twitter_data)  # Save the results

# Amazon Data Pre processing
amazon_data = pd.read_csv(r"files/amazon_more.csv")

usable = amazon_data[['reviews.rating', 'reviews.text']]

usable["class"] = usable["reviews.rating"] > 4
usable["class"] = usable["class"].replace([True, False], [1, 0])

usable["processed"] = usable["reviews.text"].apply(preprocess)

save_file("files/amazon_data.pkl", usable)  # Save the results

# IMDB review
imdb_data = pd.read_csv(r"files/imdb.csv")


imdb_data["sentiment"] = imdb_data["sentiment"].replace(["positive", "negative"], [1, 0])
imdb_data["review"] = imdb_data["review"].apply(preprocess)

save_file("files/imdb_data.pkl", imdb_data)  # Save the results

