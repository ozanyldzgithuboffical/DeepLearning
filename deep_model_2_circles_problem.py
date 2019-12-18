"""
Created on Tue Sep 24 11:33:12 2019

@author: Ozan YILDIZ

About the Code:
Circles problem:
    outer circles are 0 and inner ones are 1
    determined sample is:1500 and noise is :0.15
"""
#imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Used to impute the missing data values
from sklearn.preprocessing import Imputer
#Used to convert nominal data to numerical data for machine learning model
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
#Used  to create test amd train data
from sklearn.model_selection import train_test_split
#import Linear Regression
from sklearn.linear_model import LinearRegression
#import statmodel
import  statsmodels.api as sm

from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from matplotlib import pyplot
from numpy import where
# scatter plot of the circles dataset with points colored by class
from sklearn.datasets import make_circles
	
# generate circles
X, y = make_circles(n_samples=1000, noise=0.1, random_state=1)

# select indices of points with each class label
for i in range(2):
	samples_ix = where(y == i)
	pyplot.scatter(X[samples_ix, 0], X[samples_ix, 1], label=str(i))
pyplot.legend()
pyplot.show()

#create test,train dependent and independent variables
Train_X,Test_X,Train_Y,Test_Y=train_test_split(X,y,test_size=0.33,random_state=0)


# define model
model = Sequential()
model.add(Dense(50, input_dim=2, activation='relu', kernel_initializer='he_uniform'))
#the cross entropy requires
model.add(Dense(1, activation='sigmoid'))

opt = SGD(lr=0.02, momentum=0.7)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
# fit model
history = model.fit(Train_X, Train_Y, validation_data=(Test_X, Test_Y), epochs=500, verbose=0)