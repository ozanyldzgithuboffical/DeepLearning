# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 16:10:46 2019

@author: ozanyildiz
"""

import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
#Used  to create test amd train data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

data=pd.read_csv('winequality_white.csv')
X=data.iloc[:,0:11].values
y=data.iloc[:,11].values
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state=0)

#Standardization of X_train
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

#Dimention reduction from 13 to 2
pca=PCA(n_components=2)
X_train2=pca.fit_transform(X_train)
X_test2=pca.transform(X_test)

#has 13 features in that dimension
lg=LogisticRegression(random_state=0)
lg.fit(X_train,y_train)

#has 2 features in that dimension
lg2=LogisticRegression(random_state=0)
lg2.fit(X_train2,y_train)

#predictions
y_pred=lg.predict(X_test)
y_pred2=lg2.predict(X_test2)

#confusion matrix calculation
#Actual vs Predicted
cm=confusion_matrix(y_test,y_pred)
print(cm)

#Actual vs PCA Predicted
cm2=confusion_matrix(y_test,y_pred2)
print(cm2)

#Before PCA/After PCA
cm3=confusion_matrix(y_pred,y_pred2)
print(cm3)

#lda definition
lda=LDA(n_components=2)
X_train_lda=lda.fit_transform(X_train,y_train)
X_test_lda=lda.fit_transform(X_test)

lg_lda=LogisticRegression(random_state=0)
lg_lda.fit(X_train_lda,y_train)

y_predict_lda=lg_lda.predict(X_test_lda)
