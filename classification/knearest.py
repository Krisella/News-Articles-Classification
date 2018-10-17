from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from collections import Counter

class knnClassifier(BaseEstimator, ClassifierMixin):

	def __init__(self, k=7):
		self.k=k

	def fit(self, X, y, **kwargs):
		self.k = kwargs['k'] 
		self.X = X
		self.y = y
		return self

	def predict(self, X):
		# loop over all observations
		predictions = []
		for i in range(len(X)):
			predictions.append(self.knn_predict(self.X, self.y, X[i, :], self.k))
		return predictions

	def knn_predict(self, X_train, y_train, x_test, k):
		# create list for distances and targets
		distances = []
		targets = []

		for i in range(len(X_train)):
			# first we compute the euclidean distance
			distance = np.sqrt(np.sum(np.square(x_test - X_train[i, :])))
			# add it to list of distances
			distances.append([distance, i])

		# sort the list
		distances = sorted(distances)

		# make a list of the k neighbors' targets
		for i in range(k):
			index = distances[i][1]
			targets.append(y_train[index])

		# return most common target
		return Counter(targets).most_common(1)[0][0]