from sklearn.metrics import confusion_matrix
#import tensorflow as tf
import pandas as pd
import json
import pprint as pp
import pymysql.cursors
import graphviz
import pydot
from keras.utils.vis_utils import plot_model
from keras.layers import Dense, LSTM
from keras.models import Sequential
import keras
from keras.callbacks import TensorBoard
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
import csv

print("hello world")


# Importing the dataset; the goal here is to import the data from the mysql database
# and map all of the database fields into the dataset variable
# SQL script from main.py written by Dr. Griffin
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
    sql = "SELECT DISTINCT(Stock) FROM `DowJonesComponentsDowIndexComp` "

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


# Save the Sql query as a dataframe, either use result variable or type SQL query
