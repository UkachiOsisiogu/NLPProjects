import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
# from sklearn.cross_validation import cross_val_score
# from sklearn.cross_validation import StratifiedShuffleSplit


# TRAIN_SET_PATH = "r8-no-stop.txt"

GLOVE_6B_50D_PATH = "files/glove_6B.txt"
GLOVE_27B_200D_PATH = "files/glove.twitter.27B.200d.txt"
encoding="utf-8"

# Evaluation
from sklearn import metrics


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


def split_data(data, label, percentage):
    """
    This function is used to split the data
    Args:
        data: data
        label: target
        percentage: test size

    Returns:
        X_train, X_test, y_train, y_test

    """
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=percentage)
    return X_train, X_test, y_train, y_test


imdb_data = load_file("files/imdb_data.pkl")
X = imdb_data.review
y = imdb_data.sentiment

data = np.array(X)
target = np.array(y)


# Splitting the data set
train_data, test_data, train_label, test_label = split_data(data, target, 0.2)

results = {}

with open(GLOVE_6B_50D_PATH, "rb") as lines:
    wvec = {line.split()[0].decode(encoding): np.array(line.split()[1:], dtype=np.float32)
               for line in lines}

lr_glove_small = Pipeline([("glove vectorizer", MeanEmbeddingVectorizer(glove_small)),
                        ("logistic regression", LogisticRegression(max_iter=10000, tol=0.1, solver="lbfgs"))])
lr_glove_small_tfidf = Pipeline([("glove vectorizer", TfidfEmbeddingVectorizer(glove_small)),
                        ("logistic regression", LogisticRegression(max_iter=10000, tol=0.1, solver="lbfgs"))])
lr_glove_big = Pipeline([("glove vectorizer", MeanEmbeddingVectorizer(glove_big)),
                        ("logistic regression", LogisticRegression(max_iter=10000, tol=0.1, solver="lbfgs"))])
lr_glove_big_tfidf = Pipeline([("glove vectorizer", TfidfEmbeddingVectorizer(glove_big)),
                        ("logistic regression", LogisticRegression(max_iter=10000, tol=0.1, solver="lbfgs"))])
lr_w2v = Pipeline([("w2v vectorizer", MeanEmbeddingVectorizer(w2v)),
                        ("logistic regression", LogisticRegression(max_iter=10000, tol=0.1, solver="lbfgs"))])
lr_w2v_tfidf = Pipeline([("w2v vectorizer", TfidfEmbeddingVectorizer(w2v)),
                        ("logistic regression", LogisticRegression(max_iter=10000, tol=0.1, solver="lbfgs"))])