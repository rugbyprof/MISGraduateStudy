# Importing Libraries
from math import sqrt
from numpy import argmax, array, concatenate
from pandas import read_csv, DataFrame, concat
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM
import keras.losses
import pandas as pd
import pymysql.cursors
import pprint as pp
import json

# Convert the multivariate times series problem to become a supervised problem, method from machinelearningmastery.com
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ..., t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t,t+1,t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# Pulling everything together in aggregate
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows w/ NAN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

# Load the dataset through read_csv by placing header=0 and index_col='Date' or 1
df = pd.read_csv('Copy of DowJonesComponentsDowIndexComp.csv', header=0, index_col=1) #Where the index_col[1] is equal to the index column of 0 in the dataset
print(df.head(5))
vals = df.values
encode = LabelEncoder()
vals[:,0] = encode.fit_transform(vals[:,0]) #Encodes the stock ticker as a value
# Make sure all the data is float32
vals = vals.astype('float32')
# Normalize the features
scaler = MinMaxScaler(feature_range=(0,1))
scaled = scaler.fit_transform(vals)
#dscaled = pd.DataFrame(scaled)
#dsort = dscaled.sort_values(by=[0,1])
# Use the series_to_supervised function to change the problem to be supervised
reframed = series_to_supervised(scaled, 1, 2) #Each variable (8 of them) goes into the method and it returns a t-1, t, t+1 for each variable resulting in 24 columns
print(reframed.head())
# Remove fields we don't want to predict. We have 3 timesteps for each variable (St, O, C, H, L, Vol, V , #) 
# We will be using the previous and current vars to predict the last column, var3(t+1) which is Close price at t+1
reframed.drop(reframed.columns[[8,16,17,19,20,21,22,23]], axis=1, inplace=True) 
print(reframed)

# Split the data into train and test sets
vals = reframed.values
# Split the train & test sets into input and outputs
# Predict column 2, close price
train_x, test_x, train_y, test_y = train_test_split(vals[:,:-1], vals[:,-1], stratify=vals[:,0], test_size=0.25)
# Reshape the input for 3 dimensions = [samples=n, timesteps=1 due to t-1,t,t+1, features=the num of vars used to predict the y at each timestep]
train_x = train_x.reshape((train_x.shape[0], 1, train_x.shape[1]))
test_x = test_x.reshape((test_x.shape[0], 1, test_x.shape[1]))
print(train_x.shape, train_y.shape, test_x.shape, test_y.shape) #Expect (734001,1,15) for train_x and (314572,1,15) for test_x and one column with same rows for train,test y's.

# Fun part... finally. Design the network
ledom = Sequential()
ledom.add(LSTM(16, input_shape=(train_x.shape[1], train_x.shape[2])))
ledom.add(Dense(1))
ledom.compile(loss='mean_absolute_percentage_error', optimizer='adam')
# Fit network
math = ledom.fit(train_x, train_y, epochs=50, batch_size=90, validation_data=(test_x, test_y), verbose=2, shuffle=False)
# Plot the fitting process. Without GPU, each epoch takes an average 30 seconds on my pc
plt.plot(math.history['loss'], label='train')
plt.plot(math.history['val_loss'], label='test')
plt.legend()
plt.show()
# Define a prediction for the newly trained model
yhat = ledom.predict(test_x) #ValueError: operands could not be broadcast together with shapes (314572,15) (8,) (314572,15)
test_x = test_x.reshape((test_x.shape[0], test_x.shape[2]))
# Change the scaling back to original by inverse for forecast
inv_yhat = concatenate((yhat, test_x[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat) #ValueError: operands could not be broadcast together with shapes (314572,15) (8,) (314572,15)
inv_yhat = inv_yhat[:, 0]
# Now we invert the scaling for the actual data points for comparison
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_x[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:, 0]
# Calculate the Root Mean Squared Error, have to import the sqrt function from python, keras library only has the mean squared error function
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse) #0.233 but I don't think this value has been inv_transformed under 10 epochs
