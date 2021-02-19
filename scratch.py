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