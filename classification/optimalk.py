from knearest import knnClassifier
from sklearn.cross_validation import KFold, cross_val_score
import pandas as pd
from config import CLASS_CONFIG as conf
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.pipeline import Pipeline
import numpy as np
import time

train_data = pd.read_csv('../datasets/train_set.csv', sep="\t")
test_data = pd.read_csv('../datasets/test_set.csv', sep="\t")

weighted_titles_train = train_data["Title"]
for i in range(conf['title_weight']-1):
	weighted_titles_train = weighted_titles_train + " " + train_data["Title"]

le = preprocessing.LabelEncoder()
le.fit(train_data["Category"])
y = le.transform(train_data["Category"])

train_data["Content"] = weighted_titles_train + " " + train_data["Content"]
vectorizer = TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS)
svd_model = TruncatedSVD(n_components=100)
svd_transformer = Pipeline([('tfidf', vectorizer), 
							('svd', svd_model)])
svd_matrix_train = svd_transformer.fit_transform(train_data["Content"])

# creating odd list of K for KNN
myList = list(range(1,50))

# subsetting just the odd ones
neighbors = filter(lambda x: x % 2 != 0, myList)

# empty list that will hold cv scores
cv_scores = []
start_time = time.time()

# perform 10-fold cross validation
for k in neighbors:
    knn = knnClassifier()
    scores = cross_val_score(knn, svd_matrix_train, y,fit_params={'k':k},
     cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())

# changing to misclassification error
MSE = [1 - x for x in cv_scores]

# determining best k
optimal_k = neighbors[MSE.index(min(MSE))]
print "The optimal number of neighbors is %d" % optimal_k

# plot misclassification error vs k
plt.plot(neighbors, MSE)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Misclassification Error')
plt.show()
print("--- %s seconds ---" % (time.time() - start_time))
