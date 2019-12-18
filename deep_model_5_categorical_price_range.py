# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 15:17:14 2019

@author: ozanyildiz
About the code:
    The mobile price range is classified with deep neural network
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#Keras imports
import keras
from keras.models import Sequential
from keras.layers import Dense

#dataset import
#You need to change #directory accordingly
dataset = pd.read_csv(‘mobile_price_train.csv’) 
 #Return 20 rows of data
dataset.head(20)

X = dataset.iloc[:,:20].values
y = dataset.iloc[:,20:21].values

#standard scaler applied on train data since ANN works with inputs between range 0-1
sc = StandardScaler()
X = sc.fit_transform(X)

#train actual data is applied with one-hot-encoder since integer is converted to binary formation
ohe = OneHotEncoder()
y = ohe.fit_transform(y).toarray()

#test and train data is splitted. 0.9 rate of data used for training and the rest is for test.
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.1,random_state=0)


model = Sequential()
model.add(Dense(20, input_dim=20, activation=’relu’))
model.add(Dense(15, activation=’relu’))
#for dimensions that price can range
model.add(Dense(4, activation=’softmax’))

#compile model
opt = SGD(lr=0.01, momentum=0.6)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
# fit model
history = model.fit(Train_X, Train_Y, validation_data=(Test_X, Test_Y), epochs=500,batch_size=4,verbose=0)

#predicting the model
y_pred = model.predict(X_test)
#Converting predictions to label
predictions = list()
for i in range(len(y_pred)):
    predictions.append(np.argmax(y_pred[i]))
#Converting one hot encoded test label to label
test = list()
for i in range(len(y_test)):
    test.append(np.argmax(y_test[i]))
    
a = accuracy_score(predictions,test)
print('Accuracy is:', a*100)