# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 14:23:57 2018

@author: Al
Using Pandas, numpy, matplotlib, keras, and tensorflow for the backend,
to analyze the changes in prices of the components of the Dow Jones index 
and in composite. 
"""

import math
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import Input, Embedding, LSTM, GRU, ConvLSTM2D, CuDNNGRU, CuDNNLSTM
from keras.models import Sequential, Model

#Importing the excel worksheet into pandas
xlsx = pd.ExcelFile('stocks1.xlsx')
stex = pd.read_excel(xlsx, 'Dow-Issue Data', index_col=0, na_values=['NA'], parse_date=['date_strings'])
stex.index = pd.to_datetime(stex.index)

stex.info()

#Descriptive Statistics on the dataset
#stex.info()
#stex.head()
#stex.tail()
#stex.describe()

#Improve the model by increasing the amount of LSTM layers and neurons per layer.
#Perhaps also adding a convolutional layer in between the LSTM layers may help to identify market signals.

#Define the input, embedding, and output shapes of the model
data_dimensions = 11 #characteristics that describe a stock
timesteps = 20 #set of moving averages over the last n days for training
num_class = 10 #number of possible classes for ranking

# The goal is to set the model up so that for each 12 columns to break the rows up into training
# sets. And then move onto train the next set of 12 columns and do the same thing.
# Then generalize all of the models by combining their signals from each training set. 

model = Sequential()
model.add(ConvLSTM2D(12, filters = 11, kernel_size = (1,12), padding = "same", return_sequences = True))
layer_batch_normalization()
model.add(LSTM(20, return_sequences=True, input_shape=(timesteps, data_dimensions)))
model.add(LSTM(48, return_sequences=True))
model.add(LSTM(18))
model.add(Dense(10, activation="softmax"))
model.compile(optimizer="rmsprop", loss="mean_squared_error", metrics=['accuracy'])

x = stex.iloc[:, :12]



#Evaluate the model using the RMSE (Root mean squared error)
#rmse = math.sqrt(mean_squared_error(actual_close, predicted_close)
