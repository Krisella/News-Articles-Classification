from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn import cross_validation
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.svm import SVC
import csv

import pandas as pd

train_data = pd.read_csv('../datasets/train_set.csv', sep="\t")
test_data = pd.read_csv('../datasets/test_set.csv', sep="\t")

weighted_titles_train = train_data["Title"]
for i in range(5):
	weighted_titles_train = weighted_titles_train + " " + train_data["Title"]
train_data["Content"] = weighted_titles_train + " " + train_data["Content"]

le = preprocessing.LabelEncoder()
le.fit(train_data["Category"])
y = le.transform(train_data["Category"])

vectorizer = TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS)
svd_model = TruncatedSVD(n_components=400)

svd_transformer = Pipeline([('tfidf', vectorizer), 
                            ('svd', svd_model)])
svd_matrix_train = svd_transformer.fit_transform(train_data["Content"])
svd_matrix_test = svd_transformer.transform(test_data["Content"])

# Set the parameters by cross-validation
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
k_fold = cross_validation.KFold(len(train_data["Content"]), n_folds=10, shuffle=True, random_state=42)
clf = GridSearchCV(SVC(), tuned_parameters, cv=k_fold,
                       scoring=make_scorer(precision_score, average='micro'))
clf.fit(svd_matrix_train,y)
print("Best parameters set found on development set:")
print()
print(clf.best_params_)
