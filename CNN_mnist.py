# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 01:48:43 2019

@author: OranTeknoloji
About the Code:MINst image classification example
taking 28x28 gray scale image 
"""

import numpy as np
import mnist
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.utils import to_categorical

num_filters = 8
filter_size = 3
pool_size = 2

#test and train data gathered
train_images = mnist.train_images()
train_labels = mnist.train_labels()
test_images = mnist.test_images()
test_labels = mnist.test_labels()

# Normalization
train_images = (train_images / 255) + 0.5
test_images = (test_images / 255) + 0.5


# Reshaping
train_images = np.expand_dims(train_images, axis=3)
test_images = np.expand_dims(test_images, axis=3)

#modelling
model = Sequential([
  Conv2D(num_filters, filter_size, input_shape=(28, 28, 1)),
  MaxPooling2D(pool_size=pool_size),
  Flatten(),
  Dense(10, activation='softmax'),
])

#compling
model.compile(
  'adam',
  loss='categorical_crossentropy',
  metrics=['accuracy'],
)

#traning model
#to categorical test data is converted via one hot encoding
model.fit(
  train_images,
  to_categorical(train_labels),
  epochs=3,
  validation_data=(test_images, to_categorical(test_labels)),
)
