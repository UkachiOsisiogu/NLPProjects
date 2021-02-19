from word2vec_preprocess import load_file
from traditionl_word2Vec import (split_data, np,
                                 use_classifier)


# Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier

# Twitter data set
X_data = load_file("files/Twitter_Airline.pkl")
twitter_data = load_file("files/airline_twitter.pkl")

w2v = load_file("files/Twitter_Airline_w2v.pkl")
glove_small = load_file("files/Twitter_Airline_glove_small.pkl")
glove_big = load_file("files/Twitter_Airline_glove_big.pkl")

data = np.array(X_data)
target = np.array(twitter_data.sentiment)

train_data, test_data, train_label, test_label = split_data(data, target, 0.2)

# Create the models
lr_clf = LogisticRegression(max_iter=10000, tol=0.1, solver="lbfgs")
# Use the classifier
use_classifier(lr_clf, "Twitter_Logistic_Regression",
               train_data, train_label, test_data, test_label,
               glove_small=glove_small, glove_big=glove_big,
               w2v=w2v)
