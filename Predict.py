#!/usr/bin/python
# -*- coding:utf-8 -*-

from sklearn.externals import joblib
import pandas as pd
import numpy as np

application_test = pd.read_csv('./Data/application_test.csv')

cols=['AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY','AMT_GOODS_PRICE']

testData=pd.DataFrame([])
testData['SK_ID_CURR']=application_test['SK_ID_CURR']

for col in cols:
    temp=pd.DataFrame(application_test[col])
    temp['SK_ID_CURR']=application_test['SK_ID_CURR']
    testData=pd.merge(testData,temp,on='SK_ID_CURR')
#产生一个新的feature : amt_credit_annuity_ratio
testData['AMT_CREDIT_ANNUITY_RATIO']=testData['AMT_CREDIT'].div(testData['AMT_ANNUITY'])

#Data normalization

cols=testData.columns.tolist()
# the index 'SK_ID_CURR' does't calculated. 
for col in cols[1:]:
    testData.loc[:,[col]]=pd.to_numeric(testData[col])
for col in cols[1:]:
    testData.loc[:,[col]]=testData[col].sub(testData[col].mean()).div(testData[col].std()).tolist()

def submission(model,X,fname):
    ans = pd.DataFrame(columns=['SK_ID_CURR', 'TARGET'])
    ans.SK_ID_CURR = X.SK_ID_CURR
    X1=X.drop(['SK_ID_CURR'], axis=1)
    ans.TARGET = pd.Series(np.reshape(model.predict(X1),[-1]), index=ans.index)
    ans.to_csv(fname, index=False)


#input the test Data to generate the prediction result.

predic_test=testData
predic_test=predic_test.dropna()   
    
clf_knn =joblib.load('./Model/clf_knn.pkl')
submission(clf_knn,predic_test,'./Data/clf_knn_predict.csv')

clf_lg1=joblib.load('./Model/clf_lg1.pkl')
submission(clf_lg1,predic_test,'./Data/clf_lg1_predict.csv')

clf_rfc1=joblib.load('./Model/clf_rfc1.pkl')
submission(clf_rfc1,predic_test,'./Data/clf_rfc1_predict.csv')

clf_svc=joblib.load('./Model/clf_svc.pkl')
submission(clf_svc,predic_test,'./Data/clf_svc_predict.csv')

clf_xgb1=joblib.load('./Model/clf_xgb1.pkl')
submission(clf_xgb1,predic_test,'./Data/clf_xgb1_predict.csv')

clf_vc=joblib.load('./Model/clf_vc.pkl')
submission(clf_vc,predic_test,'./Data/clf_vc_predict.csv')



