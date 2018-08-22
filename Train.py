#!/usr/bin/python
# -*- coding: utf-8 -*-
import pandas as pd
import sklearn
import xgboost as xgb
import numpy as np
from mlxtend.regressor import StackingRegressor


trainData=pd.read_csv('./Data/Train.csv')


from sklearn.metrics import mean_absolute_error

from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV

def get_model(estimator, parameters, X_train, y_train, scoring):  
    model = GridSearchCV(estimator, param_grid=parameters, scoring=scoring)
    model.fit(X_train, y_train)
    return model.best_estimator_

from sklearn.metrics import accuracy_score
scoring = make_scorer(accuracy_score, greater_is_better=True)


from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

code_test_data=trainData[:100]

X = code_test_data.drop(['SK_ID_CURR','TARGET'], axis=1)
y = pd.DataFrame(code_test_data['TARGET'])
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)


import xgboost as XGB
xgb = XGB.XGBClassifier(seed=42, max_depth=3, objective='binary:logistic', n_estimators=40)
parameters = {'learning_rate':[0.1],
              'reg_alpha':[3.0], 'reg_lambda': [4.0]}
clf_xgb1 = get_model(xgb, parameters, X_train, y_train, scoring)
joblib.dump(clf_xgb1, './Model/clf_xgb1.pkl')


from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier(weights='uniform')
parameters = {'n_neighbors':[3,4,5], 'p':[1,2]}
clf_knn = get_model(KNN, parameters, X_train, y_train, scoring)
joblib.dump(clf_knn, './Model/clf_knn.pkl')


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(random_state=42, criterion='entropy', min_samples_split=5, oob_score=True)
parameters = {'n_estimators':[50], 'min_samples_leaf':[12]}
clf_rfc1 = get_model(rfc, parameters, X_train, y_train, scoring)
joblib.dump(clf_rfc1 , './Model/clf_rfc1.pkl')


from sklearn.linear_model import LogisticRegression
lg = LogisticRegression(random_state=42, penalty='l1')
parameters = {'C':[0.5]}
clf_lg1 = get_model(lg, parameters, X_train, y_train, scoring)
joblib.dump(clf_lg1  , './Model/clf_lg1.pkl')


from sklearn.svm import SVC
svc = SVC(random_state=42, kernel='poly', probability=True)
parameters = {'C': [35], 'gamma': [0.0055], 'coef0': [0.1],
              'degree':[2]}
clf_svc = get_model(svc, parameters, X_train, y_train, scoring)
joblib.dump(clf_svc, './Model/clf_svc.pkl')


from sklearn.ensemble import VotingClassifier
clf_vc = VotingClassifier(estimators=[('xgb1', clf_xgb1), ('lg1', clf_lg1), ('svc', clf_svc), 
                                      ('rfc1', clf_rfc1), ('knn', clf_knn)], 
                          voting='hard', weights=[4,1,1,1,2])
clf_vc = clf_vc.fit(X_train, y_train)
joblib.dump(clf_vc, './Model/clf_vc.pkl')
