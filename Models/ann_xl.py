# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cntk as C
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import 

#Excel Version Import
#excel_dataset = pd.read_excel('djita.pre.xlsx', parse_dates=['Date']) //Can't convert to float64 due to Nan, infinity, or a value too large for dtype. Can't fix it. 

excel_dataset_nodate = pd.read_excel('H:\MISGraduateStudy\djita.pre2.xlsx') #without date column

mscaler = QuantileTransform(output_distribution='normal')
type(mscaler)
mscaler.fit_transform(excel_dataset_nodate) #Input contains NaN, infinity or a value too large for dtype('float64')

import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 52, kernel_initializer = 'uniform', activation = 'relu', input_dim = 52))

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

# Evaluating the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 152, kernel_initializer = 'uniform', activation = 'relu', input_dim = 52))
    classifier.add(Dense(units = 275, kernel_initializer = 'uniform', activation = 'tanh'))
    classifier.add(Dense(units = 198, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 225, kernel_initializer = 'uniform', activation = 'tanh'))
    classifier.add(Dense(units = 52, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 10)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
mean = accuracies.mean()
variance = accuracies.std()

# Improving the ANN
# Dropout Regularization to reduce overfitting if needed

# Tuning the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 152, kernel_initializer = 'uniform', activation = 'relu', input_dim = 52))
    classifier.add(Dense(units = 275, kernel_initializer = 'uniform', activation = 'tanh'))
    classifier.add(Dense(units = 198, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 225, kernel_initializer = 'uniform', activation = 'tanh'))
    classifier.add(Dense(units = 52, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [25, 75],
              'epochs': [100, 500],
              'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_