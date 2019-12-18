# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 14:24:34 2019

@author: ozanyildiz
About the code:
    Solves multi-categorical blob problems using keras
"""
# mlp for the blobs multi-class classification problem with cross-entropy loss
from sklearn.datasets.samples_generator import make_blobs
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import to_categorical
from matplotlib import pyplot
# generate 2d classification dataset
X, y = make_blobs(n_samples=1600, centers=4, n_features=2, cluster_std=2, random_state=2)
# one hot encode output variable
y = to_categorical(y)
#create test,train dependent and independent variables
Train_X,Test_X,Train_Y,Test_Y=train_test_split(X,y,test_size=0.33,random_state=0)
# define model
model = Sequential()
model.add(Dense(60, input_dim=2, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(4, activation='softmax'))
# compile model
opt = SGD(lr=0.01, momentum=0.6)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
# fit model
history = model.fit(Train_X, Train_Y, validation_data=(Test_X, Test_Y), epochs=500, verbose=0)
# evaluate the model
_, train_acc = model.evaluate(Train_X, Train_Y, verbose=0)
_, test_acc = model.evaluate(Test_X, Test_Y, verbose=0)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
# plot loss during training
pyplot.subplot(211)
pyplot.title('Loss')
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
# plot accuracy during training
pyplot.subplot(212)
pyplot.title('Accuracy')
pyplot.plot(history.history['accuracy'], label='train')
pyplot.plot(history.history['val_accuracy'], label='test')
pyplot.legend()
pyplot.show()