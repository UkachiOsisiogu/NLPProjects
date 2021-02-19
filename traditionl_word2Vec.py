import pandas as pd
import numpy as np
import pickle
from collections import Counter, defaultdict
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from word2vec_preprocess import (preprocess, get_word_vectors,
                                 load_file, store_data)


# Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier

# Evaluation
from sklearn import metrics

# amazon_data = load_file("files/amazon_data.pkl")
# name_1 = "Amazon"
# vocab, X_data, y_data = preprocess(name_1, amazon_data)
# w2v, glove_small, glove_big = get_word_vectors(name_1, vocab, X_data)

name_2 = "Twitter_Airline"
twitter_data = load_file("files/airline_twitter.pkl")
vocab_2, X_data_2, y_data_2 = preprocess(name_2, twitter_data)
w2v_2, glove_small_2, glove_big_2 = get_word_vectors(name_2, vocab_2, X_data_2)

# w2v = load_file("files/Twitter_Airline_w2v.pkl")
# glove_small = load_file("files/Twitter_Airline_glove_small.pkl")
# glove_big = load_file("files/Twitter_Airline_glove_big.pkl")


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


class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        if len(word2vec) > 0:
            self.dim = len(word2vec[next(iter(glove_small))])
        else:
            self.dim = 0

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])


# and a tf-idf version of the same
class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        if len(word2vec) > 0:
            self.dim = len(word2vec[next(iter(glove_small))])
        else:
            self.dim = 0

    def fit(self, X, y):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] * self.word2weight[w]
                     for w in words if w in self.word2vec] or
                    [np.zeros(self.dim)], axis=0)
            for words in X
        ])


def use_classifier(clf, classifier_name,
                   X_train, y_train,
                   X_test, y_test,
                   glove_small, glove_big, w2v):
    """
    This function is used to apply the classifiers and collate the results
    Args:
        clf: Machine Learning classifier
        classifier_name: Description of classifier

    Returns:

    """

    print(f"Working on {classifier_name}")

    # Create Pipelines
    clf_glove_small = Pipeline([("glove vectorizer", MeanEmbeddingVectorizer(glove_small)),
                               ("classifier", clf)])

    clf_glove_small_tfidf = Pipeline([("glove vectorizer", TfidfEmbeddingVectorizer(glove_small)),
                                     ("classifier", clf)])

    clf_glove_big = Pipeline([("glove vectorizer", MeanEmbeddingVectorizer(glove_big)),
                             ("classifier", clf)])

    clf_glove_big_tfidf = Pipeline([("glove vectorizer", TfidfEmbeddingVectorizer(glove_big)),
                                   ("classifier", clf)])

    clf_w2v = Pipeline([("w2v vectorizer", MeanEmbeddingVectorizer(w2v)),
                       ("classifier", clf)])
    clf_w2v_tfidf = Pipeline([("w2v vectorizer", TfidfEmbeddingVectorizer(w2v)),
                             ("classifier", clf)])

    all_clf_models = [
        (f"{classifier_name}_glove_small", clf_glove_small),
        (f"{classifier_name}_glove_small_tfidf", clf_glove_small_tfidf),
        (f"{classifier_name}_glove_big", clf_glove_big),
        (f"{classifier_name}_glove_big_tfidf", clf_glove_big_tfidf),
        (f"{classifier_name}_w2v", clf_w2v),
        (f"{classifier_name}_w2v_tfidf", clf_w2v_tfidf)
    ]

    clf_results = {}

    for name, pipeline in all_clf_models:
        print(f"Working on {name}")
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        print(metrics.classification_report(y_test, predictions, target_names=["Negative", "Positive"]))
        f1_score = metrics.f1_score(y_test, predictions)

        print(f"F1-score for {classifier_name} = {f1_score}")

        clf_results[name] = f1_score

    store_data(f"files/{classifier_name}_Word2Vec_results.pkl", clf_results)
    return clf_results


rf_clf = RandomForestClassifier(n_estimators=200)

etree_clf = ExtraTreesClassifier(n_estimators=200)

dt_clf = DecisionTreeClassifier()


# Amazon
# data = np.array(X_data)
# target = np.array(y_data)
#
# train_data, test_data, train_label, test_label = split_data(data, target, 0.2)
#
# use_classifier(dt_clf, "DTree_Classifier_Amazon", train_data, train_label, test_data, test_label,
#                 glove_small=glove_small, glove_big=glove_big, w2v=w2v)

# Twitter
data_2 = np.array(X_data_2)
target_2 = np.array(y_data_2)

X_train, X_test, y_train, y_test = split_data(data_2, target_2, 0.2)

use_classifier(dt_clf, "DTree_Classifier_Twitter", X_train, y_train, X_test, y_test,
                   glove_small=glove_small_2, glove_big=glove_big_2,
                   w2v=w2v_2)



