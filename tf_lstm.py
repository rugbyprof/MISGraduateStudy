from pandas import DataFrame
from pandas import concat
import tensorflow as tf
from tensorflow.keras import layers
import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt

def data_model(nodes):
    model = tf.keras.Sequential()
    model.add(layers.Dense(nodes, activation='relu'))
    model.add(layers.Dense(nodes, activation='relu'))
    model.add(layers.Dense(np.round(nodes/2), activation='softmax'))
    model.compile(optimizer=tf.train.AdamOptimizer(
        0.001), loss='categorical_crossentropy', metrics=['categorical_accuracy'])


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	"""
	Frame a time series as a supervised learning dataset.
	Arguments:
		data: Sequence of observations as a list or NumPy array.
		n_in: Number of lag observations as input (X).
		n_out: Number of observations as output (y).
		dropnan: Boolean whether or not to drop rows with NaN values.
	Returns:
		Pandas DataFrame of series framed for supervised learning.
	"""
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg


data = DataFrame()
data = pd.read_csv('Copy of DowJonesComponentsDowIndexComp.csv')
data['Stock'] = data['Stock'].astype("category")
values = data.values
df = data.melt(id_vars=[0,1])
data = series_to_supervised(values, 1, 1)
print(data)
