# Artificial Neural Network; this code streams data from a mysql database and uses a MLP NN to predict stock price trends.
# Importing Libraries
import scipy 
from numpy import argmax, array
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, LabelBinarizer
import tensorflow as tf
from keras.callbacks import TensorBoard
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.utils.vis_utils import plot_model
from keras.utils import to_categorical
import pydot
import graphviz
from sklearn.metrics import confusion_matrix 
import pymysql.cursors
import pprint as pp
import json

# Importing the dataset; the goal here is to import the data from the mysql database
# and map all of the database fields into the dataset variable
# SQL script from main.py written by Dr. Griffin
'''
with open('config.json', encoding='utf-8') as data_file:
   config = json.loads(data_file.read())

# Connect to the database
connection = pymysql.connect(host=config['host'],
                             user=config['user'],
                             password=config['password'],
                             db=config['db'],
                             charset='utf8mb4',
                             cursorclass=pymysql.cursors.DictCursor)


def getAllStockNames():
    sql =  "SELECT DISTINCT(Stock) FROM `DowJonesComponentsDowIndexComp` "

    with connection.cursor() as cursor:
        cursor.execute(sql)
        result = cursor.fetchall()

    return result

def selectData(sql):

    print(sql)

    with connection.cursor() as cursor:
        cursor.execute(sql)
        result = cursor.fetchall()

    return result
'''


# Initial plan: Save the Sql query as a dataframe, either use result variable or type SQL query. Backup: Export table as CSV and parse using existing code. 
df = pd.read_csv('Copy of DowJonesComponentsDowIndexComp.csv')
df = df.sort_values('Stock', ascending=False).sort_values('Date', ascending=True)
print(df.describe()) 

# Formatting Datetime in Pandas
df['Date'] = pd.to_datetime(df['Date']) 
df.index = df['Date']
print(df.head(5))


# Encode the Stock name
#df = pd.DataFrame(df)
lb = LabelBinarizer()
df['stock'] = lb.fit_transform(df['Stock']).tolist() #.inverse_transform(y) back to encoding
print(df)
# Add column, OHLC average to be the y dependent variable to predict
OHLC_avg = df[['Open','Close','High','Low']].mean(axis = 1)
# Add column to df
dfs = pd.concat([df,OHLC_avg], axis=1)
dfs = dfs.sort_values('Stock').sort_values('Date')
dfs = dfs.rename(columns = ['0', 'OHLC'])

sf = array(df)
# Must convert Dataframe to array to iterate over in for loop and then convert back into dataframe for Scaling
# Split the dataset into samples of 200 observations
samples = list()
length = 200
n = len(sf)
# Step over the range in steps of 200
for i in range(0, n, length):
	#grab from i to i+200
	sample = sf[i:i+length]
	samples.append(sample)
print(len(samples))

# Convert samples to an array
data = array(samples)
print(data.shape)

# Scale the data through normalization
# scaler = MinMaxScaler(feature_range=(0, 1))
# normalized = pd.DataFrame(scaler.fit_transform(data))
# Manual normalization - Error "ValueError: setting an array element with a sequence." - Haven't figured it out. 

# Reshape the dataset for an LSTM network. Must be 3d data. Contains 8 features.
normalized = tf.reshape((len(samples), length, 8))
print(normalized.shape)  # Expect (250, 200, 8) Haven't quite figured this out.

# Save the dataset to csv for future use
normalized.to_csv('ml_stock_normalized.csv')

# Convert the multivariate times series problem to become a supervised problem
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
		agg.dropna(inplace=true)
	return agg


# Reframe the multivariate times series to become a supervised problem
reframed = series_to_supervised(normalized, 1, 1)
print(reframed.head())

# Save the prepared dataset to csv for future use
reframed.to_csv('ml_stock_reframed.csv')

# Drop columns we don't want to predict
# reframed.drop(reframed.columns[[, , , , , , , ]], axis=1, inplace=True)
print(reframed.head())

# Split data into train & test sets
values = reframed.values
data_split = values.len*0.75
train = values[:data_split, :]
test = values[data_split:, :]

# Split the train & test data into inputs and outputs
train_x, train_y = train[:, :-1], train[:, -1]
test_x, test_y = test[:, :-1], test[:, -1]

# Reshape input? May move reshape method to this point.
train_x = train_x.reshape((train_x.shape[0], 1, train_x.shape[1]))
test_x = test_x.reshape((test_x.shape[0], 1, test_x.shape[1]))
print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)

# Design the neural network
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_x.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')

# Fit network
historical = model.fit(train_x, train_y, epochs=50, batch_size=200,
                       validation_data=(test_x, test_y), verbose=2, shuffle=False)

# Visualize
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

# Make a prediction
yhat = model.predict(test_x)
test_x = test_x.reshape((test_x.shape[0], test_x.shape[2]))

# Invert scaling for forecast
inv_yhat = concatenate((yhat, test_x[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:, 0]

# Invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_x[:, 1:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:, 0]

# Calculate RMSE (root mean squared error)
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)


'''
This if block runs when you call this file directly. Its also a good way to test programs
by keeping the driver code below
'''
'''
if __name__=='__main__':
    stock_names = getAllStockNames()
    #pp.pprint(stock_names)

    for stock in stock_names:
        print(stock['Stock'])
        # 1st Goal is to pull price information (OHLC) for each stock by ticker and date.
        sql =  "SELECT `Stock`, `Close` FROM `DowJonesComponentsDowIndexComp` WHERE `Stock` = '%s'" % stock['Stock']
        print(sql)
        result = selectData(sql)

        pp.pprint(result)
        print(len(result))


# Stuff that is not important. Will Delete later. 

# New preprocessing method using MinMaxScaler to test training methods over the previous StandardScaler method.
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
cm = confusion_matrix(y_test, y_pred)

#Visualizing the model
keras.utils.print_summary(classifier, line_length=None, positions=None, print_fn=None)
keras.utils.plot_model(classifier, to_file='model.png', show_shapes=False, show_layer_names=True, rankdir='TB')
'''
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
