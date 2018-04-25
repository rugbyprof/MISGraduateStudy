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
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import Input, Embedding, LSTM, GRU, ConvLSTM2D, CuDNNGRU, CuDNNLSTM
from keras.models import Sequential, Model

# Randomness seed
np.random.seed(7)

#Importing the excel worksheet into pandas
xlsx = pd.ExcelFile('stocks1.xlsx')
#for index in range(len(xlxs))
stex = pd.read_excel(xlsx, 'Dow-Issue Data', index_col=0, na_values=['NA'], parse_date=['date_strings'], usecols=[3,4,5,6,7])
stex = stex.reindex(index = stex.index[::1])

stex.info()

# Define a custom index for flexibility
observations = np.arange(1, len(stex) + 1, 1)

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

#Evaluate the model using the RMSE (Root mean squared error)
#rmse = math.sqrt(mean_squared_error(actual_close, predicted_close)

# TAKING DIFFERENT INDICATORS FOR PREDICTION
# OHLC_avg = stex.mean(axis = 1)
# close_val = stex[['Close']]

# PLOTTING ALL INDICATORS IN ONE PLOT
plt.plot(obs, OHLC_avg, 'r', label = 'OHLC avg')
plt.plot(obs, HLC_avg, 'b', label = 'HLC avg')
plt.plot(obs, close_val, 'g', label = 'Closing price')
plt.legend(loc = 'upper right')
plt.show()

# PREPARATION OF TIME SERIES DATASE
OHLC_avg = np.reshape(OHLC_avg.values, (len(OHLC_avg),1)) # 1664
scaler = MinMaxScaler(feature_range=(0, 1))
OHLC_avg = scaler.fit_transform(OHLC_avg)

# TRAIN-TEST SPLIT
train_OHLC = int(len(OHLC_avg) * 0.75)
test_OHLC = len(OHLC_avg) - train_OHLC
train_OHLC, test_OHLC = OHLC_avg[0:train_OHLC,:], OHLC_avg[train_OHLC:len(OHLC_avg),:]

# TIME-SERIES DATASET (FOR TIME T, VALUES FOR TIME T+1)
trainX, trainY = preprocessing.new_dataset(train_OHLC, 1)
testX, testY = preprocessing.new_dataset(test_OHLC, 1)

# RESHAPING TRAIN AND TEST DATA
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
step_size = 1

# LSTM MODEL
model = Sequential()
model.add(LSTM(32, input_shape=(1, step_size), return_sequences = True))
model.add(LSTM(16))
model.add(Dense(1))
model.add(Activation('linear'))

# MODEL COMPILING AND TRAINING
model.compile(loss='mean_squared_error', optimizer='adagrad') # Try SGD, adam, adagrad and compare!!!
model.fit(trainX, trainY, epochs=5, batch_size=1, verbose=2)

# PREDICTION
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# DE-NORMALIZING FOR PLOTTING
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])


# TRAINING RMSE
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train RMSE: %.2f' % (trainScore))

# TEST RMSE
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test RMSE: %.2f' % (testScore))

# CREATING SIMILAR DATASET TO PLOT TRAINING PREDICTIONS
trainPredictPlot = np.empty_like(OHLC_avg)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[step_size:len(trainPredict)+step_size, :] = trainPredict

# CREATING SIMILAR DATASSET TO PLOT TEST PREDICTIONS
testPredictPlot = np.empty_like(OHLC_avg)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(step_size*2)+1:len(OHLC_avg)-1, :] = testPredict

# DE-NORMALIZING MAIN DATASET 
OHLC_avg = scaler.inverse_transform(OHLC_avg)

# PLOT OF MAIN OHLC VALUES, TRAIN PREDICTIONS AND TEST PREDICTIONS
plt.plot(OHLC_avg, 'g', label = 'original dataset')
plt.plot(trainPredictPlot, 'r', label = 'training set')
plt.plot(testPredictPlot, 'b', label = 'predicted stock price/test set')
plt.legend(loc = 'upper right')
plt.xlabel('Time in Days')
plt.ylabel('OHLC Value of Apple Stocks')
plt.show()

# PREDICT FUTURE VALUES
last_val = testPredict[-1]
last_val_scaled = last_val/last_val
next_val = model.predict(np.reshape(last_val_scaled, (1,1,1)))
print "Last Day Value:", np.asscalar(last_val)
print "Next Day Value:", np.asscalar(last_val*next_val)
# print np.append(last_val, next_val)