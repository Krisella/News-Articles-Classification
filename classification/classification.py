from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.linear_model import SGDClassifier
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn import cross_validation
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from nltk.stem.snowball import SnowballStemmer
import csv
from collections import Counter
import numpy as np
import pandas as pd
import time
from config import CLASS_CONFIG as conf
from knearest import knnClassifier
from multiscorer import MultiScorer
import nltk
import sys



reload(sys)  
sys.setdefaultencoding('utf8')


def run_svm_opt():
	from config import SVM_OPT as svm_opt

	scorer = MultiScorer({
	    'Accuracy' : (accuracy_score, {}),
	    'Precision' : (precision_score, {'average':'macro'}),
	    'Recall' : (recall_score, {'average':'macro'}),
	    'F1' : (f1_score, {'average':'macro'})
	})
	train_data = pd.read_csv('../datasets/train_set.csv', sep="\t")
	test_data = pd.read_csv('../datasets/test_set.csv', sep="\t")

	le = preprocessing.LabelEncoder()
	le.fit(train_data["Category"])
	y = le.transform(train_data["Category"])

	print "Preprocessing training data..."
	weighted_titles_train = train_data["Title"]
	for i in range(svm_opt['title_weight']-1):
		weighted_titles_train = weighted_titles_train + " " + train_data["Title"]

	train_data["Content"] = weighted_titles_train + " " + train_data["Content"]
	vectorizer = TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS)
	svd_model = TruncatedSVD(n_components=svm_opt['n-components'])
	svd_transformer = Pipeline([('tfidf', vectorizer),
							('svd', svd_model)])
	svd_matrix_train = svd_transformer.fit_transform(train_data["Content"])
	if(svm_opt['SVM-kernel'] == 'linear'):
		clf = svm.SVC(kernel = 'linear', C=svm_opt['SVM-C'], random_state=42)
	else:
		clf = svm.SVC(kernel = svm_opt['SVM-kernel'], C=svm_opt['SVM-C'], gamma=svm_opt['SVM-gamma'], random_state=42)
	print "Running SVM classifier..."
	k_fold = cross_validation.KFold(len(train_data["Content"]), n_folds=10, shuffle=True, random_state=42)
	cross_val_score(clf, svd_matrix_train, y, cv=k_fold, scoring=scorer)
	return scorer.get_results()

def run_rf_opt():
	from config import RF_OPT as rf_opt

	scorer = MultiScorer({
	    'Accuracy' : (accuracy_score, {}),
	    'Precision' : (precision_score, {'average':'macro'}),
	    'Recall' : (recall_score, {'average':'macro'}),
	    'F1' : (f1_score, {'average':'macro'})
	})
	train_data = pd.read_csv('../datasets/train_set.csv', sep="\t")
	test_data = pd.read_csv('../datasets/test_set.csv', sep="\t")

	le = preprocessing.LabelEncoder()
	le.fit(train_data["Category"])
	y = le.transform(train_data["Category"])

	print "Preprocessing training data..."
	weighted_titles_train = train_data["Title"]
	for i in range(rf_opt['title_weight']-1):
		weighted_titles_train = weighted_titles_train + " " + train_data["Title"]

	train_data["Content"] = weighted_titles_train + " " + train_data["Content"]
	vectorizer = TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS)
	svd_model = TruncatedSVD(n_components=rf_opt['n-components'])
	svd_transformer = Pipeline([('tfidf', vectorizer),
							('svd', svd_model)])
	svd_matrix_train = svd_transformer.fit_transform(train_data["Content"])
	clf = RandomForestClassifier(n_estimators = rf_opt['RF-estimators'], random_state=42)
	print "Running RandomForest classifier..."
	k_fold = cross_validation.KFold(len(train_data["Content"]), n_folds=10, shuffle=True, random_state=42)
	cross_val_score(clf, svd_matrix_train, y, cv=k_fold, scoring=scorer)
	return scorer.get_results()

def run_mnb_opt():
	from config import MNB_OPT as mnb_opt
	scorer = MultiScorer({
	    'Accuracy' : (accuracy_score, {}),
	    'Precision' : (precision_score, {'average':'macro'}),
	    'Recall' : (recall_score, {'average':'macro'}),
	    'F1' : (f1_score, {'average':'macro'})
	})
	train_data = pd.read_csv('../datasets/train_set.csv', sep="\t")
	test_data = pd.read_csv('../datasets/test_set.csv', sep="\t")

	le = preprocessing.LabelEncoder()
	le.fit(train_data["Category"])
	y = le.transform(train_data["Category"])

	print "Preprocessing training data..."
	weighted_titles_train = train_data["Title"]
	for i in range(mnb_opt['title_weight']-1):
		weighted_titles_train = weighted_titles_train + " " + train_data["Title"]

	train_data["Content"] = weighted_titles_train + " " + train_data["Content"]
	vectorizer = TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS)
	svd_transformer = Pipeline([('tfidf', vectorizer)])
	svd_matrix_train = svd_transformer.fit_transform(train_data["Content"])
	clf = MultinomialNB()
	print "Running MultinomialNB classifier..."
	k_fold = cross_validation.KFold(len(train_data["Content"]), n_folds=10, shuffle=True, random_state=42)
	cross_val_score(clf, svd_matrix_train, y, cv=k_fold, scoring=scorer)
	return scorer.get_results()

def run_knn_opt():
	from config import KNN_OPT as knn_opt
	scorer = MultiScorer({
	    'Accuracy' : (accuracy_score, {}),
	    'Precision' : (precision_score, {'average':'macro'}),
	    'Recall' : (recall_score, {'average':'macro'}),
	    'F1' : (f1_score, {'average':'macro'})
	})
	train_data = pd.read_csv('../datasets/train_set.csv', sep="\t")
	test_data = pd.read_csv('../datasets/test_set.csv', sep="\t")

	le = preprocessing.LabelEncoder()
	le.fit(train_data["Category"])
	y = le.transform(train_data["Category"])

	print "Preprocessing training data..."
	weighted_titles_train = train_data["Title"]
	for i in range(knn_opt['title_weight']-1):
		weighted_titles_train = weighted_titles_train + " " + train_data["Title"]

	train_data["Content"] = weighted_titles_train + " " + train_data["Content"]
	vectorizer = TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS)
	svd_model = TruncatedSVD(n_components=knn_opt['n-components'])
	svd_transformer = Pipeline([('tfidf', vectorizer),
							('svd', svd_model)])
	svd_matrix_train = svd_transformer.fit_transform(train_data["Content"])
	clf = knnClassifier()
	print "Running K-Nearest-Neighbor classifier..."
	k_fold = cross_validation.KFold(len(train_data["Content"]), n_folds=10, shuffle=True, random_state=42)
	cross_val_score(clf, svd_matrix_train,
					  y, fit_params={'k':knn_opt['knn-k']},
					  cv=k_fold,
					  scoring = scorer)
	return scorer.get_results()

def run_svm_stem():
	train_data = pd.read_csv('../datasets/train_set.csv', sep="\t")
	test_data = pd.read_csv('../datasets/test_set.csv', sep="\t")
	scorer = MultiScorer({
	    'Accuracy' : (accuracy_score, {}),
	    'Precision' : (precision_score, {'average':'macro'}),
	    'Recall' : (recall_score, {'average':'macro'}),
	    'F1' : (f1_score, {'average':'macro'})
	})
	le = preprocessing.LabelEncoder()
	le.fit(train_data["Category"])
	y = le.transform(train_data["Category"])

	print "Preprocessing training data..."
	weighted_titles_train = train_data["Title"]
	for i in range(5):
		weighted_titles_train = weighted_titles_train + " " + train_data["Title"]

	train_data["Content"] = weighted_titles_train + " " + train_data["Content"]
	stemmer = SnowballStemmer("english", ignore_stopwords=True)

	class StemmedCountVectorizer(CountVectorizer):
	    def build_analyzer(self):
	        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
	        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])

	stemmed_count_vect = StemmedCountVectorizer(stop_words='english')
	vectorizer = TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS)
	svd_model = TruncatedSVD(n_components=1000)
	svd_transformer = Pipeline([('vect', stemmed_count_vect),('tfidf', TfidfTransformer()),
							('svd', svd_model)])
	svd_matrix_train = svd_transformer.fit_transform(train_data["Content"])
	clf = svm.SVC(kernel = 'rbf', C=1000, gamma=0.001, random_state=42)
	print "Running SVM classifier with stemming..."
	k_fold = cross_validation.KFold(len(train_data["Content"]), n_folds=10, shuffle=True, random_state=42)
	cross_val_score(clf, svd_matrix_train, y, cv=k_fold, scoring=scorer)
	return scorer.get_results()

def run_all():
	# svm_results = run_svm_opt()
	# rf_results = run_rf_opt()
	# mnb_results = run_mnb_opt()
	# knn_results = run_knn_opt()
	svm_stem = run_svm_stem()
	print sum(svm_stem['Accuracy'])/len(svm_stem['Accuracy']),sum(svm_stem['Precision'])/len(svm_stem['Precision']),sum(svm_stem['Recall'])/len(svm_stem['Recall']),sum(svm_stem['F1'])/len(svm_stem['F1'])
	with open("./EvaluationMetric_10fold.csv", "wb") as csv_file:
	        writer = csv.writer(csv_file)
	        writer.writerow(("Statistic Measure","Naive_Bayes","Random Forest","SVM","KNN","My Method"))
	        writer.writerow(("Accuracy", sum(mnb_results['Accuracy'])/len(mnb_results['Accuracy']),
	        					sum(rf_results['Accuracy'])/len(rf_results['Accuracy']),
	        					sum(svm_results['Accuracy'])/len(svm_results['Accuracy']),
	        					sum(knn_results['Accuracy'])/len(knn_results['Accuracy']),
	        					sum(svm_stem['Accuracy'])/len(svm_stem['Accuracy'])))
	        writer.writerow(("Precision", sum(mnb_results['Precision'])/len(mnb_results['Precision']),
					sum(rf_results['Precision'])/len(rf_results['Precision']),
					sum(svm_results['Precision'])/len(svm_results['Precision']),
					sum(knn_results['Precision'])/len(knn_results['Precision']),
					sum(svm_stem['Precision'])/len(svm_stem['Precision'])))
        	writer.writerow(("Recall", sum(mnb_results['Recall'])/len(mnb_results['Recall']),
					sum(rf_results['Recall'])/len(rf_results['Recall']),
					sum(svm_results['Recall'])/len(svm_results['Recall']),
					sum(knn_results['Recall'])/len(knn_results['Recall']),
					sum(svm_stem['Recall'])/len(svm_stem['Recall'])))
        	writer.writerow(("F-Measure", sum(mnb_results['F1'])/len(mnb_results['F1']),
					sum(rf_results['F1'])/len(rf_results['F1']),
					sum(svm_results['F1'])/len(svm_results['F1']),
					sum(knn_results['F1'])/len(knn_results['F1']),
					sum(svm_stem['F1'])/len(svm_stem['F1'])))


if(conf['mode'] == 'calc_all'):
	run_all()
else:

	train_data = pd.read_csv('../datasets/train_set.csv', sep="\t")
	test_data = pd.read_csv('../datasets/test_set.csv', sep="\t")

	le = preprocessing.LabelEncoder()
	le.fit(train_data["Category"])
	y = le.transform(train_data["Category"])

	weighted_titles_train = train_data["Title"]
	for i in range(conf['title_weight']-1):
		weighted_titles_train = weighted_titles_train + " " + train_data["Title"]

	train_data["Content"] = weighted_titles_train + " " + train_data["Content"]

	# if(conf['stemming'] == 1):
	# 	stemmer = SnowballStemmer("english", ignore_stopwords=True)
	# 	train_data['Content']=train_data['Content'].apply(lambda x : filter(None,x.split(" ")))
	# 	stemmed=train_data['Content'].apply(lambda x : [stemmer.stem(token) for token in x])
	# 	train_data['Content']=stemmed.apply(lambda x : " ".join(x))
	# 	print("Finished stemming")
	stemmer = SnowballStemmer("english", ignore_stopwords=True)

	class StemmedCountVectorizer(CountVectorizer):
	    def build_analyzer(self):
	        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
	        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])

	stemmed_count_vect = StemmedCountVectorizer(stop_words='english')

	vectorizer = TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS)
	svd_model = TruncatedSVD(n_components=conf['n-components'])

	if conf['classifier'] != 'MNB':
		# svd_transformer = Pipeline([('tfidf', vectorizer), 
		# 							('svd', svd_model)])
		if(conf['stemming'] == 1):
			svd_transformer = Pipeline([('vect', stemmed_count_vect), ('tfidf', TfidfTransformer()),
										('svd', svd_model)])
		else:
			svd_transformer = Pipeline([('tfidf', vectorizer),
										('svd', svd_model)])
	elif conf['stemming'] == 1:
		svd_transformer = Pipeline([('vect', stemmed_count_vect), ('tfidf', TfidfTransformer())])
	else:
		svd_transformer = Pipeline([('tfidf', vectorizer)])

	svd_matrix_train = svd_transformer.fit_transform(train_data["Content"])


	clf=None
	if(conf['classifier'] == 'SVM'):
		if(conf['SVM-kernel'] == 'linear'):
			clf = svm.SVC(kernel = 'linear', C=conf['SVM-C'], random_state=42)
		else:
			clf = svm.SVC(kernel = conf['SVM-kernel'], C=conf['SVM-C'], gamma=conf['SVM-gamma'], random_state=42)
	elif(conf['classifier'] == 'RF'):
		clf = RandomForestClassifier(n_estimators = conf['RF-estimators'], random_state=42)
	elif(conf['classifier'] == 'MNB'):
		clf = MultinomialNB()
	elif(conf['classifier'] == 'KNN'):
		clf = knnClassifier()
	elif(conf['classifier'] == 'SGD'):
		clf = SGDClassifier(loss=conf['SGD-loss'], random_state=42)




	if conf['mode'] == '10fold':

		k_fold = cross_validation.KFold(len(train_data["Content"]), n_folds=10, shuffle=True, random_state=42)
		if conf['classifier'] == 'KNN':
			cross_val_score(clf,
						  svd_matrix_train,
						  y, fit_params={'k':conf['knn-k']},
						  cv=k_fold,
						  scoring = scorer)
		else:
			cross_val_score(clf, svd_matrix_train, y, cv=k_fold, scoring=scorer)
		results = scorer.get_results()
		print sum(results['Accuracy'])/len(results['Accuracy']), sum(results['Precision'])/len(results['Precision']), sum(results['Recall'])/len(results['Recall']), sum(results['F1'])/len(results['F1'])

	elif conf['mode'] == 'pred':
		weighted_titles_test = test_data["Title"]
		for i in range(conf['title_weight']-1):
			weighted_titles_test = weighted_titles_test + " " + test_data["Title"]

		test_data["Content"] = weighted_titles_test + " " + test_data["Content"]
		svd_matrix_test = svd_transformer.transform(test_data["Content"])


		clf.fit(svd_matrix_train,y)
		results = le.inverse_transform(clf.predict(svd_matrix_test))

		with open("./testSet_categories.csv", "wb") as csv_file:
		        writer = csv.writer(csv_file)
		        writer.writerow(("Id","Predicted_Category"))
		        for i in range(len(results)):
		            writer.writerow( (test_data["Id"][i], results[i]))

