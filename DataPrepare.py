#!/usr/bin/python
# -*- coding: utf-8 -*-

import pandas as pd

# prepare training data.

application_train=pd.read_csv('~/DataAnalysis/jupyter-notebook/HomeCredit/homeCredit/application_train.csv')
cols=['AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY','AMT_GOODS_PRICE','TARGET']

trainData=pd.DataFrame([])
trainData['SK_ID_CURR']=application_train['SK_ID_CURR']

for col in cols:
    temp=pd.DataFrame(application_train[col])
    temp['SK_ID_CURR']=application_train['SK_ID_CURR']
    trainData=pd.merge(trainData,temp,on='SK_ID_CURR')
#产生一个新的feature : amt_credit_annuity_ratio
trainData['AMT_CREDIT_ANNUITY_RATIO']=trainData['AMT_CREDIT'].div(trainData['AMT_ANNUITY'])

#Data normalization

cols=trainData.columns.tolist()
# the index 'SK_ID_CURR' does't calculated. 
for col in cols[1:-2]:
    trainData.loc[:,[col]]=pd.to_numeric(trainData[col])
for col in cols[1:-2]:
    trainData.loc[:,[col]]=trainData[col].sub(trainData[col].mean()).div(trainData[col].std()).tolist()

trainData=trainData.dropna()
trainData.to_csv('./Data/Train.csv',index=0)

# divide train data into train part and cross_validation_part

from sklearn.model_selection import train_test_split
X = modelTestData.drop(['SK_ID_CURR','TARGET'], axis=1)
y = modelTestData.TARGET
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

X_train.to_csv('./Data/X_Train.csv',index=0)
X_test.to_csv('./Data/X_Test.csv',index=0)
y_train.to_csv('./Data/y_Train.csv',index=0)
y_test.to_csv('./Data/y_Test.csv',index=0)

# prepare test data.

application_test = pd.read_csv('~/DataAnalysis/jupyter-notebook/HomeCredit/homeCredit/application_test.csv')
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
testData.to_csv('./Data/Test.csv',index=0)
