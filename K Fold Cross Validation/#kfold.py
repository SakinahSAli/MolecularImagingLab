#kfold

#importing libraries
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC 
from sklearn.ensemble import RandomForestClassifier         
from sklearn.tree import DecisionTreeClassifier 
import numpy as np
from sklearn.model_selection import cross_val_score 

#load dataset
iris = load_iris()

#import models
def get_score (model, x_train, y_train, x_test, y_test):
    model.fit(x_train, y_train)
    return model.score(x_test, y_test)

# use StratifiedKFold with iris data
from sklearn.model_selection import StratifiedKFold
folds = StratifiedKFold(n_splits=3)
scores_logistic = []
scores_rf = []
scores_svm = [] 



for train_index, test_index in folds.split(iris.data, iris.target):
    x_train, x_test =iris.data [train_index],iris.data [test_index]
    y_train, y_test = iris.target[train_index], iris.target[test_index]
    # use the local split variables (lowercase) and pass arguments in the order expected by get_score
    scores_logistic.append(get_score(LogisticRegression(solver='liblinear', multi_class='ovr'), x_train, y_train, x_test, y_test))
    scores_svm.append(get_score(SVC(gamma='auto'), x_train, y_train, x_test, y_test))
    scores_rf.append(get_score(RandomForestClassifier(n_estimators=40), x_train, y_train, x_test, y_test))

# quick summary
print('Logistic scores per fold:', scores_logistic)
print('SVM scores per fold:', scores_svm)
print('Random Forest scores per fold:', scores_rf)
if scores_logistic:
    print('Logistic avg:', np.mean(scores_logistic))
if scores_svm:
    print('SVM avg:', np.mean(scores_svm))
if scores_rf:
    print('Random Forest avg:', np.mean(scores_rf))