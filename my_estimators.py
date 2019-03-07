
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, Lasso, LassoCV, RidgeClassifier, ElasticNetCV
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, RobustScaler

feature_selection = SelectPercentile(f_classif, percentile=10)
# ANOVA + SVC_l1
svc_l1 = LinearSVC(penalty='l1', dual=False, random_state=0)
# ANOVA + SVC_l1
anova_svcl1 = Pipeline([('anova', feature_selection), ('svc', svc_l1)])
# SVC_l2
svc_l2 = LinearSVC(penalty='l2', random_state=0)
# ANOVA + SVC_l1
anova_svcl2 = Pipeline([('anova', feature_selection), ('svc', svc_l2)])
# Gaussian NaiveBayes
gnb = GaussianNB()
# RandomForestClassifier
randomf = RandomForestClassifier(random_state=0)
# Logistic Regression 'l1'
logregression_l1 = LogisticRegression(penalty='l1', dual=False, random_state=0)
logregression_l1_C10000 = LogisticRegressionCV(penalty='l1', solver='liblinear', dual=False, random_state=0)

# tspisak
#logregression_l1_CV = Pipeline([('scale', StandardScaler()),('logreg',LogisticRegressionCV(penalty='l1', solver='liblinear', dual=False,
#                                           random_state=0, cv=20))])


# Logistic Regression 'l2'
logregression_l2 = LogisticRegression(penalty='l2', random_state=0)

elastic_net_CV = ElasticNetCV(alphas=5, random_state=0)


# Lasso
lasso = Lasso(alpha=0.1, random_state=0)
# tspisak
lassoCV = Pipeline([('scale', StandardScaler()), ('lasso', LassoCV(alphas=[0.01, 0.1, 1, 10], cv=10))])
# tspisak

knn = KNeighborsClassifier(n_neighbors=1)
ridge = RidgeClassifier()

"""
sklearn_classifiers = {'GaussianNB': gnb,
                       'RandomF': randomf,
                       'logistic_l1': logregression_l1,
                       'logistic_l2': logregression_l2,
                       'lasso': lasso,
                       'anova_svcl1': anova_svcl1,
                       'anova_svcl2': anova_svcl2,
                       'svc_l1': svc_l1,
                       'svc_l2': svc_l2,
                       'ridge': ridge,
                       'knn': knn}
"""
# tspisak
sklearn_classifiers = {'logistic_l1': logregression_l1,
                        'logistic_l1_C10000': logregression_l1_C10000,
                       'logistic_l2': logregression_l2,
                       #'elasticNetCV' : elastic_net_CV,
                       #'lasso': lasso,
                       #'ts_lassoCV': lassoCV
                       #'ridge': ridge
                        }
