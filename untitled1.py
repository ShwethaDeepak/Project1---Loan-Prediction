#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  3 11:21:50 2018

@author: swetu
"""


import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

df_train = pd.read_csv('/home/swetu/anaconda3/Loan predictor/train.csv')
df_test = pd.read_csv('/home/swetu/anaconda3/Loan predictor/test.csv')

df_train.isnull().sum()
labels = df_train.iloc[:,-1]
df_train = df_train.drop(['Loan_Status'],axis=1)
df_train['Source']=111
df_test['Source']=222
data = pd.concat([df_train,df_test],ignore_index=True)
data.isnull().sum()
data.dtypes
data['Gender'].fillna(data['Gender'].mode()[0],inplace = True)
data['Married'].fillna(data['Married'].mode()[0],inplace = True)
data['Dependents'].fillna(data['Dependents'].mode()[0],inplace = True)
#data['Education'].fillna(data['Education'].mode()[0],inplace = True)
data['Self_Employed'].fillna(data['Self_Employed'].mode()[0],inplace = True)
#data['Property_Area'].fillna(data['Property_Area'].mode()[0],inplace = True)
data['Loan_Amount_Term'].fillna(data['Loan_Amount_Term'].mean(),inplace = True)
data['LoanAmount'].fillna(data['LoanAmount'].mean(),inplace = True)
data['Credit_History'].fillna(data['Credit_History'].mean(),inplace = True)

data.isnull().sum()
labels

df1 = pd.get_dummies(data['Gender'])
frames= [data,df1]
result1 = pd.concat(frames,axis = 1)

df2 = pd.get_dummies(data['Married'])
frames1= [df2,result1]
result2 = pd.concat(frames1,axis = 1)

df3 = pd.get_dummies(data['Education'])
frames2= [df3,result2]
result3 = pd.concat(frames2,axis = 1)

df4= pd.get_dummies(data['Self_Employed'])
frames3= [df4,result3]
result4 = pd.concat(frames3,axis = 1)

df5= pd.get_dummies(data['Property_Area'])
frames4= [df5,result4]
result5 = pd.concat(frames4,axis = 1)


result5 = result5.drop(['Property_Area'],axis = 1)
result5 = result5.drop(['Gender'],axis = 1)
result5 = result5.drop(['Married'],axis = 1)
result5 = result5.drop(['Self_Employed'],axis = 1)
result5 = result5.drop(['Education'],axis = 1)
result5 = result5.drop(['Loan_ID'],axis = 1)


from sklearn.preprocessing import LabelEncoder
encode = LabelEncoder()
result6 = encode.fit_transform(data['Loan_ID'])
labels = encode.fit_transform(labels)

result6 = pd.DataFrame(result6)
frames5 = [result5,result6]
result7 = pd.concat(frames5,axis = 1)
result7.dtypes
df6 = pd.get_dummies(data['Dependents'])
frames6 = [result7,df6]
result9 = pd.concat(frames6,axis = 1)
result9= result9.drop(['Dependents'],axis = 1)
result9.dtypes




df_train = result9.loc[data['Source']==111]
df_test= result9.loc[data['Source']==222]

df_train= df_train.drop(['Source'],axis = 1)
df_test= df_test.drop(['Source'],axis = 1)
df_train

#result7 = result7.drop(['Source'],axis = 1)

from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(df_train,labels,test_size = 0.25,random_state = 0)

df_train.dtypes

lr = LogisticRegression(random_state = 0)
Knn = KNN(n_neighbors = 27)
dt = DecisionTreeClassifier(min_samples_leaf = 0.13,random_state = 0)
classifiers = [('LogisticRegression',lr),('KNearestNeighbors',Knn),('DecisionTree',dt)]
for df_name,clf in classifiers:
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_pred,y_test)
    print('{:s} : {:.3f}'.format (df_name,accuracy))
    

from sklearn.ensemble import VotingClassifier
VC = VotingClassifier(estimators = classifiers)
VC.fit(X_train,y_train)
y_pred_VC = VC.predict(X_test)
accuracy_VC = accuracy_score(y_pred_VC,y_test)
print('Voting Classifier:',accuracy_VC)
