# Artificial Neural Network


# This information below is from the template from my class on Udemy
# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler


# Importing the dataset; the goal here is to import the data from the CSV file 
# and map the independent variables (X) to the dependent variable (Y)
dataset = pd.read_csv('\MISGraduateStudy\Data\djita2_train.csv', delimiter=',', encoding='latin1', low_memory=False)
dataset_test = pd.read_csv('\MISGraduateStudy\Data\djita2_test.csv', delimiter=',', encoding='latin1', low_memory=False)

# New preprocessing method using MinMaxScaler to test training methods over the previous StandardScaler method.

# Error on variable is "ndarray object of numpy module is not currently used" **Fixed with the help of Dr. Zhang
X = dataset.iloc[:, 1:53].values 
X_test = dataset_test.iloc[:, 1:53].values

# These values are the last column of the dataset 
# which is a binary variable. 1 represents that today's 
# closing price is less than the closing price tomorrow and 0 vice versa.
# **Corrected with the help of Dr. Zhang.
y = dataset.iloc[:, 53:54].values 
y_test = dataset_test.iloc[:, 53:54].values

# Encoding categorical data; My thought was that I would 
# need to encode the date in the first column of the data set, 
# but I am open to suggestions.**Corrected with the help of Dr. Zhang.

# Feature Scaling
mscaler = RobustScaler()
X_train = mscaler.fit_transform(X)
y_train = mscaler.fit_transform(y)

X_test = mscaler.fit_transform(X_test)
y_test = mscaler.fit_transform(y_test)

# Imporing the Keras libraries and packages

import tensorflow as tf 
from keras.callbacks import TensorBoard
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.utils.vis_utils import plot_model
import pydot
import graphviz

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 100, kernel_initializer = 'VarianceScaling', activation = 'relu', input_dim = 52))

# Adding the first second hidden layer
classifier.add(Dense(units = 200, activation = 'sigmoid'))

#Added Recurrent Layer to iterate over the dataset
# classifier.add(LSTM(units = 275, input_dim=3, return_sequences=True)) #Won't work, input 0 is incompatible with lstm ndim

# Adding the second hidden layer
classifier.add(Dense(units = 100, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the third hidden layer - CMK2
classifier.add(Dense(units = 150, kernel_initializer = 'Orthogonal', activation = 'tanh'))

# Adding the fourth hidden layer - CMK2
classifier.add(Dense(units = 50, kernel_initializer = 'RandomNormal', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'linear'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=['mae', 'acc'])

#Visualization in Real Time
tensorboard = TensorBoard(log_dir="logs/{}", histogram_freq=0, write_graph=True, write_grads=False)
                          
# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 50, epochs = 100, verbose=1, callbacks=[tensorboard])

y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#Visualizing the model
keras.utils.print_summary(classifier, line_length=None, positions=None, print_fn=None)
keras.utils.plot_model(classifier, to_file='model.png', show_shapes=False, show_layer_names=True, rankdir='TB')
'''
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
'''