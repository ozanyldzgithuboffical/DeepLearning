"""
Created on Tue Sep 24 11:33:12 2019

@author: Ozan YILDIZ

About the Code:
Regression Loss Function Application with mean squared error
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

#reading csv
data=pd.read_csv('data.csv')
#read missing data csv
missingData=pd.read_csv('missingdata.csv')

#instantiation of imputer
imputerObj=Imputer(missing_values='NaN',strategy='mean',axis=0)

#get numerical ones
numericalData=missingData.iloc[:,1:4].values

#compute numerical data according to the strategy 
imputerObj=imputerObj.fit(digitalData[:,0:4])
#transform and write to data 
numericalData[:,0:4]=imputerObj.transform(numericalData[:,0:4])


#extracting length data frame of age & gender
#dataFrame1=data[['age','gender']]


#label encoding object is instantiated
labelEncoding=LabelEncoder()

#first we get country data values
country=data.iloc[:,0:1].values
#print(country)
country[:,0]=labelEncoding.fit_transform(country[:,0])

#instance of OneHotEncoding
oneHotEncoding=OneHotEncoder()

country=oneHotEncoding.fit_transform(country).toarray()

#Create Data Frame for country
countryDataFrame=pd.DataFrame(data=country,index=range(22),columns=['tr','fr','us'])
#print(countryDataFrame)

#Create numerical data frame
numericalDataFrame=pd.DataFrame(data=numericalData,index=range(22),columns=['length','weight','age'])
#print(numericalDataFrame)

#concat two data frame I do not add the gender since it is the descriptor field of model
ultimateDataFrame=pd.concat([countryDataFrame,numericalDataFrame],axis=1)
#print(ultimateDataFrame)

#create frane for  independent gender
genderData=data.iloc[:,-1:].values
genderData[:,0]=labelEncoding.fit_transform(genderData[:,0])

#instance of OneHotEncoding
oneHotEncoding=OneHotEncoder()

genderData=oneHotEncoding.fit_transform(genderData).toarray()
genderDataFrame=pd.DataFrame(data=genderData[:,:1],index=range(22),columns=['gender'])
latestDataFrame=pd.concat([ultimateDataFrame,genderDataFrame],axis=1)
#print(genderDataFrame)
#create test,train dependent and independent variables
Train_X,Test_X,Train_Y,Test_Y=train_test_split(ultimateDataFrame,genderDataFrame,test_size=0.33,random_state=0)

#define model
model=Sequential()
model.add(Dense(20,input_dim=4,activation='relu'))
model.add(Dense(8,activation='relu'))
#output
model.add(Dense(1,activation='sigmoid'))

#create Sthocastic Gradient Descent
optimizer=SGD(lr=0.01,momentum=0.87)
model.compile(loss='mean_square_error',optimizer)
model.fit(Train_X,Test_Y,validation_data=(Test_X,Test_Y),epochs=1000,verbose=0)
