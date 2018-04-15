# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 14:23:57 2018

@author: Al
Using Pandas, numpy, matplotlib, keras, and tensorflow for the backend,
to analyze the changes in prices of the components of the Dow Jones index. 
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
xlsx = pd.ExcelFile('stocks.xlsx')
stex = pd.read_excel(xlsx, 'Dow-Issue Data', index_col=None, na_values=['NA'])

#Descriptive Statistics on the dataset
#st.head()
#st.tail()
#st.describe()

#Improve the model by increasing the amount of LSTM layers and neurons per layer.
#Perhaps also adding a convolutional layer in between the LSTM layers may help to identify market signals.

#Define the input, embedding, and output shapes of the model



model = Sequential()
model.add(LSTM(stex))
model.add(Dense(activation="leakyRELU"))
model.compile(optimizer="rmsprop", loss="root_mean_squared_error")


#Evaluate the model using the RMSE (Root mean squared error)
rmse = math.sqrt(mean_squared_error(actual_close, predicted_close)
