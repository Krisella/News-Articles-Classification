CLASS_CONFIG = {
	# SVM, RF, MNB, KNN, SGD
    'classifier': 'SVM',
    'title_weight': 5,
    'n-components': 1000,
    'knn-k': 7,
    'SVM-kernel': 'rbf',
    'SVM-C': 1000,
    'SVM-gamma': 0.001,
    'stemming' : 0,
    #10fold, pred, calc_all
    'mode' : 'pred',
    'RF-estimators' : 100,
    'SGD-loss' : 'log'
}

# optimized params
SVM_OPT = {
	'title_weight': 5,
	'n-components': 1000,
    'SVM-kernel': 'rbf',
    'SVM-C': 1000,
    'SVM-gamma': 0.001,
}

RF_OPT = {
	'title_weight': 5,
	'n-components': 500,
    'RF-estimators' : 200,	
}

MNB_OPT = {
	'title_weight': 8
}

KNN_OPT = {
    'title_weight': 5,
	'knn-k': 5,
	'n-components': 10,
}

