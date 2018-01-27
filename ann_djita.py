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

# Importing the dataset; the goal here is to import the data from the CSV file 
# and map the independent variables (X) to the dependent variable (Y)
dataset = pd.read_csv('djita2_train.csv', delimiter=',', encoding='latin1', low_memory=False)

# Error on variable is "ndarray object of numpy module is not currently used"
X = dataset.iloc[1:, :].values 

# These values are the last column of the dataset 
# which is a binary variable. 1 represents that today's 
# closing price is less than the closing price tomorrow and 0 vice versa.
y = dataset.iloc[:, 53:54].values 

# Encoding categorical data; My thought was that I would 
# need to encode the date in the first column of the data set, 
# but I am open to suggestions.
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()

X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set

# Feature Scaling

# Imporing the Keras libraries and packages

# Initialising the ANN

# Adding the input layer and the first hidden layer

# Adding the second hidden layer

# Adding the output layer

# Compiling the ANN

# Fitting the ANN to the Training set

# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results

# Making the Confusion Matrix
