# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

#Excel Version Import
#excel_dataset = pd.read_excel('djita.pre.xlsx', parse_dates=['Date']) //Can't convert to float64 due to Nan, infinity, or a value too large for dtype. Can't fix it. 

excel_dataset_nodate = pd.read_excel('djita.pre2.xlsx') #without date column

mscaler = MinMaxScaler()
type(mscaler)
mscaler.fit_transform(excel_dataset_nodate) #Input contains NaN, infinity or a value too large for dtype('float64')

import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 152, kernel_initializer = 'uniform', activation = 'relu', input_dim = 52))

# Adding the first second hidden layer
classifier.add(Dense(units = 275, kernel_initializer = 'uniform', activation = 'tanh'))

# Adding the second hidden layer
classifier.add(Dense(units = 198, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the third hidden layer - CMK2
classifier.add(Dense(units = 225, kernel_initializer = 'uniform', activation = 'tanh'))

# Adding the fourth hidden layer - CMK2
classifier.add(Dense(units = 52, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 50, epochs = 100)

y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

