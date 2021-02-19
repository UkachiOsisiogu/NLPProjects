import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

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


def store_data(filepath, data):
    """
    This function is used for object serialization just to store what is going on
    Parameters
    ----------
    filepath: str The path where data is stored
    data: The data being stored

    Returns
    -------

    """
    pickle.dump(data, open(filepath, "wb"))
    print("Data stored successfully")


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
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size = percentage)
    return X_train, X_test, y_train, y_test


def use_classifier(clf, 
                   classifier_name, 
                   train_data, 
                   train_label, 
                   test_data,
                   test_label,
                   results,
                   classifier):
    """
    This function is used to apply the classifiers and collate the results
    Args:
        clf: Machine Learning classifier
        classifier_name: Description of classifier

    Returns:

    """
    print(f"Working on {classifier_name}")
    clf_pipeline = Pipeline([
        ('tfidf_vec', TfidfVectorizer(sublinear_tf=True, stop_words="english")),
        ('clf', clf),
    ])

    clf_pipeline.fit(train_data, train_label)
    predictions = clf_pipeline.predict(test_data)

    print(metrics.classification_report(test_label, predictions, target_names=["Negative", "Positive"]))
    f1_score = metrics.f1_score(test_label, predictions)

    print(f"F1-score for {classifier_name} = {f1_score}")
    results.append(f1_score)
    classifier.append(classifier_name)


def apply_all(dataset_name, dataset_filepath):
    dataset = load_file(dataset_filepath)
    X = dataset.review
    y = dataset.sentiment

    data = np.array(X)
    target = np.array(y)
    
    # Splitting the data set
    train_data, test_data, train_label, test_label = split_data(data, target, 0.2)
    
    
    results = []
    classifier = []
    
    # Creating a list of different classifier objects
    
    #
    lr_clf = LogisticRegression(max_iter=10000, tol=0.1, solver="lbfgs")
    use_classifier(lr_clf, "Logistic Regression", train_data, train_label, test_data, test_label,
                   results, classifier)
    
    knn_clf = KNeighborsClassifier(n_neighbors=2)
    use_classifier(knn_clf, "KNN", train_data, train_label, test_data, test_label,
                   results, classifier)
    
    # SVM Classifiers
    for penalty in ["l2", "l1"]:
    
        svm_clf = LinearSVC(penalty=penalty, tol=1e-3, dual=False)
        use_classifier(svm_clf, f"SVM_{penalty}", train_data, train_label, test_data, test_label,
                   results, classifier)
        sgd_clf = SGDClassifier(alpha=.0001, max_iter=50, penalty=penalty)
        use_classifier(sgd_clf, f"SGD_{penalty}", train_data, train_label, test_data, test_label,
                   results, classifier)
    
    # Naive Bayes Classifiers
    mnb = MultinomialNB(alpha=0.01)
    use_classifier(mnb, "MultinomialNB", train_data, train_label, test_data, test_label,
                   results, classifier)
    bnb = BernoulliNB(alpha=0.01)
    use_classifier(bnb, "BernoulliNB", train_data, train_label, test_data, test_label,
                   results, classifier)
    cnb = ComplementNB(alpha=0.01)
    use_classifier(cnb, "ComplementNB", train_data, train_label, test_data, test_label,
                   results, classifier)
    
    # Decision Tree Classifier
    dt_clf = DecisionTreeClassifier()
    use_classifier(dt_clf, "DecisionTree", train_data, train_label, test_data, test_label,
                   results, classifier)
    rf_clf = RandomForestClassifier(n_estimators=200)
    use_classifier(rf_clf, "RandomForest", train_data, train_label, test_data, test_label,
                   results, classifier)
    etree_clf = ExtraTreesClassifier(n_estimators=200)
    use_classifier(etree_clf, "ExtraTrees", train_data, train_label, test_data, test_label,
                   results, classifier)
    
    print("Done")
    
    all_results = pd.DataFrame()
    all_results["Classifiers"], all_results["F1 Score"] = classifier, results
    
    print(all_results)
    
    store_data(f"files/{dataset_name}_traditional_classifier_results.pkl", all_results)
    

# apply_all("IMDB", "files/imdb_data.pkl")
# apply_all("Amazon", "files/amazon_data.pkl")
apply_all("Airline_Twitter", "files/airline_twitter.pkl")
apply_all("Twitter", "files/twitter_data.pkl")




