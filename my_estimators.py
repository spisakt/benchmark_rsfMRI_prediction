
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, Lasso, LassoCV, RidgeClassifier, ElasticNet, ElasticNetCV
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, RobustScaler

# tspisak
# Logistic Regression 'l1'
logregression_l1 = LogisticRegression(penalty='l1', dual=False, random_state=0)

# tspisak
# Logistic Regression 'l1'
logregression_l2 = LogisticRegression(penalty='l2', dual=False, random_state=0)

# tspisak
sklearn_classifiers = { 'logistic_l1': logregression_l1,
                        'logistic_l2': logregression_l2
                        }
