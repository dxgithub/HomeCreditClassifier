#!/usr/bin/python
# -*- coding:utf-8 -*-



import sklearn
from sklearn.externals import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error

from sklearn.metrics import accuracy_score


#load trained model.

clf_knn=joblib.load('./Model/clf_knn.pkl')
clf_rfc1=joblib.load('./Model/clf_rfc1.pkl')
clf_lg1=joblib.load('./Model/clf_lg1.pkl')
clf_svr=joblib.load('./Model/clf_svc.pkl')
clf_xgb1=joblib.load('./Model/clf_xgb1.pkl')
clf_vc=joblib.load('./Model/clf_vc.pkl')


# test Model

X_Test=pd.read_csv('./Data/X_Test.csv')
y_Test=pd.read_csv('./Data/y_Test.csv')

f=open('./Model/ModelEvalResult.txt',"w")
print >> f, "Calssifcation error of clf_knn: {}".format(accuracy_score(y_Test, clf_knn.predict(X_Test)))
print >> f, "Calssifcation error of clf_rfc1: {}".format(accuracy_score(y_Test, clf_rfc1.predict(X_Test)))
print >> f, "Calssifcation error of clf_lg1: {}".format(accuracy_score(y_Test, clf_lg1.predict(X_Test)))
print >> f, "Calssifcation error of clf_svr: {}".format(accuracy_score(y_Test, clf_svr.predict(X_Test)))
print >> f, "Calssifcation error of clf_xgb1: {}".format(accuracy_score(y_Test, clf_xgb1.predict(X_Test)))
print >> f, "Calssifcation error of clf_vc: {}".format(accuracy_score(y_Test, clf_vc.predict(X_Test)))

f.close()
