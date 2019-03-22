import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.ticker as ticker 
import datetime as dt 
import matplotlib.dates as mdates
import os
from matplotlib import style
import pickle
import pandas_datareader
from collections import Counter
from sklearn import svm, neighbors
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler

TOTAL_DATASET = pd.read_csv('sp500_joined_close.csv')
real_stock_price_unscaled = TOTAL_DATASET.iloc[5000:, 1:].values
np.nan_to_num(real_stock_price_unscaled, copy=False)

sc = MinMaxScaler(feature_range = (0,1))
sc.fit(real_stock_price_unscaled)

real_stock_price = sc.transform(real_stock_price_unscaled)

print(real_stock_price[real_stock_price.shape[0] - 49:, 49])

train_test_split_amount = 0.99
train_size = int(len(real_stock_price) * train_test_split_amount)
trainset_x, testset_x = real_stock_price[0:train_size], real_stock_price[train_size- 60:len(real_stock_price)]
print('Observations: %d' % (len(real_stock_price)))
print('Training Observations: %d' % (len(trainset_x)))
print('Testing Observations: %d' % (len(testset_x)))




X_test = []
y = 0

for x in range(inputs.shape[1]):
    X_test_temp = []
    for i in range(time_steps, inputs.shape[0]):
        X_test_temp.append(inputs[i - time_steps:i, x])
    X_test_temp = np.array(X_test_temp)
    if X_test == []:
        X_test = X_test_temp
    elif y == 0:
        X_test = np.stack((X_test, X_test_temp), axis = 2)
        y += 1
    else:
        X_test_temp = np.reshape(X_test_temp, (X_test_temp.shape[0], X_test_temp.shape[1],1))
        X_test = np.append(X_test, X_test_temp, axis = 2)
y_test = inputs[inputs.shape[0] - time_steps:, :]
print(X_test.shape)

X_1 = []
inputs = trainset_x
X_test_temp = []
X_test_temp.append(inputs[0:60,0])
X_test_temp.append(inputs[1:61,0])
X_test_temp.append(inputs[2:62,0])


X_test_temp = np.array(X_test_temp)
print(X_test_temp.shape)

print(X_test_temp[0,:])
print(X_test_temp[1,:])
# X_1 = np.stack((X_1,X_test_temp),axis = 2)

# print(X_1.shape)

print(X_1)






def preprocess_FR(inputs, time_steps = 60):
    X_test = []
    y = 0
    for x in range(inputs.shape[1]):
        X_test_temp = []
        for i in range(time_steps, inputs.shape[0]):
            X_test_temp.append(inputs[i - time_steps:i, x])
        X_test_temp = np.array(X_test_temp)
        if X_test == []:
            X_test = X_test_temp
        elif y == 0:
            X_test = np.stack((X_test, X_test_temp), axis = 2)
            y += 1
        else:
            X_test_temp = np.reshape(X_test_temp, (X_test_temp.shape[0], X_test_temp.shape[1],1))
            X_test = np.append(X_test, X_test_temp, axis = 2)
    y_test = inputs[inputs.shape[0] - time_steps:, :]
    print(X_test.shape)
    return X_test, y_test

X_train1, y_train1 = preprocess_FR(trainset_x)
X_test1, y_test1 = preprocess_FR(testset_x)
