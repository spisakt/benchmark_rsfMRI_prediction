import warnings
import os
from os.path import join
import numpy as np
import pandas as pd

from downloader import fetch_abide

from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

from my_estimators import sklearn_classifiers
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, Lasso, LassoCV, RidgeClassifier, ElasticNet, ElasticNetCV
from sklearn.feature_selection import SelectPercentile, f_classif
import random

# prepare dictionary for saving results
columns = ['sparsity', 'estimator', 'score']
results = dict()
for column_name in columns:
    results.setdefault(column_name, [])

################################################
# 1. create 100x100 random matrices
################################################

np.random.seed(0)

N = 600 # sample size
M = 100 # number of "regions"
P = M*(M-1)/2 # number of features

features = []

for i in  range(N):
    rand = np.random.randn(P) # drawn from the std norm.
    features.append(rand)

################################################
# 2. add ground truth: sparse vs. non-sparse
################################################

G = int(np.floor(N/2)) # group size, first G subjects will be in Class 0, remaining in Class 1
classes = np.array([0] * G + [1] * (N-G))

P_sparse = int(np.floor(P*.01)) # number of ground truth features
P_nonsparse = int(np.floor(P*0.5))  # number of ground truth features
print(P_sparse)
print(P_nonsparse)

W=0.05 # general weight to adjust SNR
# adjust weights to be a "bit more fair" when comparing sparse with nonsparse
W_sparse = W*np.sqrt((P_nonsparse/P_sparse))
W_nonsparse = W

print(W_sparse )
print(W_nonsparse )


#ground_truth_sparse = np.array([1] * P_sparse + [0] * (P-P_sparse))
#ground_truth_nonsparse = np.array([1] * P_nonsparse + [0] * (P-P_nonsparse))
#np.random.shuffle(ground_truth_sparse)
#np.random.shuffle(ground_truth_nonsparse)
# I know random shuffles are not neccesary here, but it feels sooo good.

features_sparse = np.copy(features)
features_nonsparse = np.copy(features)

for i in range(G):
    features_sparse[i] = features[i] + np.concatenate(
        (
            #random.sample(np.array([1]*3 +[0]*(P_sparse-3) ),P_sparse),
            np.random.randn(P_sparse) + W_sparse,
            np.repeat(0, (P-P_sparse))
        )
    )

    features_nonsparse[i] = features[i] +  np.concatenate(
        (
            #random.sample(np.array([1]*3 + [0] * (P_nonsparse-3) ), P_nonsparse),
            np.random.randn(P_nonsparse) + W_nonsparse,
            np.repeat(0, (P - P_nonsparse))
        )
    )

    #features_sparse[i] = features[i] + W_sparse*ground_truth_sparse
    #features_nonsparse[i] = features[i] + W_nonsparse * ground_truth_nonsparse

features_sparse = np.array(features_sparse)
features_nonsparse = np.array(features_nonsparse)

################################################
# 3. run cross-validation to fit models
################################################

cv = StratifiedShuffleSplit(n_splits=100, test_size=0.25,
                            random_state=0)

def my_crossval(estimator, features, classes, estimator_key, sparsity_key):
    iter_for_prediction = cv.split(features, classes)

    for index, (train_index, test_index) in enumerate(iter_for_prediction):
        print(index)
        est_fit_train = estimator.fit(features[train_index], classes[train_index]) # train
        prediction_test = est_fit_train.predict_proba(features[test_index]) # predict test
        score = roc_auc_score(classes[test_index], prediction_test[:, 1]) # evaluate test
        results['sparsity'].append(sparsity_key)
        results['estimator'].append(estimator_key)
        results['score'].append(score)


# my estimators:
# Logistic Regression 'l1'
logregression_l1 = LogisticRegression(penalty='l1', dual=False, random_state=0)
# Logistic Regression 'l2'
logregression_l2 = LogisticRegression(penalty='l2', dual=False, random_state=0)

feature_selection = SelectPercentile(f_classif, percentile=5)

# ANOVA + logit_l2
anova_logregression_l2 = Pipeline([('anova', feature_selection), ('svc', logregression_l2)])

my_classifiers = { 'logistic_l1': logregression_l1,
                   'logistic_l2': logregression_l2,
                   'anova_logistic_l2': anova_logregression_l2
                   }

# for sparse:
for est_key in my_classifiers.keys():
    my_crossval(my_classifiers[est_key], features_sparse, classes, est_key, "sparse ground truth")

# for nonspoarse
for est_key in my_classifiers.keys():
    my_crossval(my_classifiers[est_key], features_nonsparse, classes, est_key, "non-sparse ground truth")

################################################
# 4. Save results for each fold
################################################

res=pd.DataFrame(results)
res.to_csv("simulation_results.csv")