import datetime as dt
import os
import pickle
import warnings
# import pandas_datareader
from collections import Counter
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
from keras.layers import LSTM, Dense, Dropout, TimeDistributed
from keras.models import Sequential, load_model
from matplotlib import style
from sklearn import neighbors, svm
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.client import device_lib

warnings.filterwarnings("ignore", category=DeprecationWarning) 


style.use('ggplot')



# allstock = pd.read_csv('all_stocks_5yr.csv')
# print(allstock.head())

# print(allstock)

# type(allstock)
# allstock.set_index('date', inplace=True)
# allstock.drop(['open', 'high', 'low', 'volume'], 1, inplace=True)
# print(allstock.head())
# ticker = pd.pivot_table(allstock, values='close', index='date', columns='Name')
# print(ticker.head())
# print(ticker.tail())
# len(ticker)
# ticker.to_csv('modifiedsp500.csv')

#Visualizing data
# df = pd.read_csv('modifiedsp500.csv')
# df['AAPL'].plot()
# plt.show()

def visualize_data():
    df = pd.read_csv('modifiedsp500.csv')
    df_corr = df.corr()

    df_corr.head()

    data = df_corr.values
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    heatmap = ax.pcolor(data, cmap = plt.cm.RdYlGn)
    fig.colorbar(heatmap)
    ax.set_xticks(np.arange(data.shape[0]) + 0.5, minor=False)
    ax.set_yticks(np.arange(data.shape[1]) + 0.5, minor = False)
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    column_labels = df_corr.columns
    row_labels = df_corr.index
    ax.set_xticklabels(column_labels)
    ax.set_yticklabels(row_labels)
    plt.xticks(rotation=90)
    heatmap.set_clim(-1,1)
    plt.tight_layout()
    plt.show()
#visualize_data()

def process_data_for_labels(ticker):
    hm_days = 7
    df = pd.read_csv('modifiedsp500.csv', index_col=0)
    tickers = df.columns.values.tolist()
    df.fillna(0, inplace=True)

    for i in range(1, hm_days+1):
        df['{}_{}d'.format(ticker, i)] = (df[ticker].shift(-i) - df[ticker]) / df[ticker]
    
    df.fillna(0,inplace=True)
    return tickers,df


# def process(ticker):
#     number_days = 60
#     df = pd.read_csv('modifiedsp500.csv', index_col = 0)
#     tickers = df.columns.values.to.list()
#     df.fillna(0, inplace=True)

def buy_sell_hold(*args):
    cols = [c for c in args]
    requirement = 0.028
    for col in cols:
        if col >=  0.029:
            return 1
        if col < -0.024:
            return -1
    return 0

def extract_featuresets(ticker):
    tickers, df = process_data_for_labels(ticker)

    df['{}_target'.format(ticker)] = list(map(buy_sell_hold, 
                                                df['{}_1d'.format(ticker)],
                                                df['{}_2d'.format(ticker)],
                                                df['{}_3d'.format(ticker)],
                                                df['{}_4d'.format(ticker)],
                                                df['{}_5d'.format(ticker)],
                                                df['{}_6d'.format(ticker)],
                                                df['{}_7d'.format(ticker)]
                                                ))
    vals = df['{}_target'.format(ticker)].values.tolist()
    str_vals = [str(i) for i in vals]
    print('Data spread:', Counter(str_vals))
    df.fillna(0, inplace=True)
    df = df.replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True)
    df_vals = df[[ticker for ticker in tickers]].pct_change()
    df_vals = df_vals.replace([np.inf, -np.inf], 0)
    df_vals.fillna(0, inplace=True)

    X = df_vals.values
    y = df['{}_target'.format(ticker)].values

    return X,y,df


def do_ml(ticker):
    X,y, df = extract_featuresets(ticker)

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,
                                                                        y,
                                                                        test_size= 0.25)
    clf = VotingClassifier([('lsvc', svm.LinearSVC()), ('knn', neighbors.KNeighborsClassifier()), ('rfor', RandomForestClassifier())])
    clf.fit(X_train, y_train)
    confidence = clf.score(X_test, y_test)
    print('Accuracy:', confidence)
    predictions = clf.predict(X_test)
    print('Predicted spread:', Counter(predictions))

    mat = confusion_matrix(y_test, predictions)
    sns.heatmap(mat, square=True, annot=True, cbar=False)
    plt.xlabel('predicted value')
    plt.ylabel('true value')

    plt.show()
    return confidence

# do_ml('AAPL')


dataset_train = pd.read_csv('sp500_joined_close.csv')
training_set = dataset_train.iloc[:, 1:]. values

print(training_set)
np.nan_to_num(training_set, copy=False)

# train_test_split_amount = 0.90
# train_size = int(len(training_set) * train_test_split_amount)
# x_train, x_test = training_set[0:train_size], training_set[train_size:len(training_set)]
# print('Observations: %d' % (len(training_set)))
# print('Training Observations: %d' % (len(x_train)))
# print('Testing Observations: %d' % (len(x_test)))

sc = MinMaxScaler(feature_range = (0,1))
training_set_scaled = sc.fit_transform(training_set)
print(training_set.shape)

time_steps = 60
X_train = []
y_train = []
for x in range(training_set_scaled.shape[1]):
    X_train_temp = []
    y_train_temp = []
    for i in range(time_steps, training_set_scaled.shape[0]):
        X_train_temp.append(training_set_scaled[i - time_steps:i, x])
        y_train_temp.append(training_set_scaled[i,x])
    X_train.append(X_train_temp)
    y_train.append(y_train_temp)

with open("sp500tickers.pickle", "rb") as f:
    tickers = pickle.load(f)
print(type(tickers))

ticker = "ABBV"
index = tickers.index(ticker)
print(index)

for i in range(time_steps, training_set_scaled.shape[0]):
        y_train.append(training_set_scaled[i,index])

y_train = np.array(y_train)
X_train = np.array(X_train)

X_train = np.reshape(X_train, (X_train.shape[1], X_train.shape[2], X_train.shape[0]))
y_train = np.reshape(y_train, (y_train.shape[1], y_train.shape[0]))
print(X_train.shape)
print(y_train.shape)
print(x_test.shape)



X_train.shape
y_train.shape

regressor.fit(X_train, y_train, epochs= 100, batch_size = 32)

regressor.save('my_model.h5')
regressor.save_weights('my_model_weights.h5')


regressor = load_model('train_1980.h5')

dataset_test = pd.read_csv('TEST_sp500_joined_close_TEST.csv')
real_stock_price = dataset_test.iloc[:, 1:].values






total_dataset = pd.concat([dataset_train, dataset_test])
np.nan_to_num(total_dataset, copy=False)
inputs = total_dataset[len(total_dataset) - len(dataset_test) - 60:].values
print(inputs.shape)
print(inputs)
training_set_scaled = sc.fit_transform(x_train)


time_steps = 60
X_test = []
y_test = []
for x in range(x_test.shape[1]):
    X_test_temp = []
    y_test_temp = []
    for i in range(time_steps, x_test.shape[0]):
        X_test_temp.append(x_test[i - time_steps:i, x])
        y_test_temp.append(x_test[i,x])
    X_test.append(X_test_temp)
    y_test.append(y_test_temp)

y_test = np.array(y_test)
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[1], X_test.shape[2], X_test.shape[0]))
y_test = np.reshape(y_test, (y_test.shape[1], y_test.shape[0]))
print(X_test.shape)
print(y_test.shape)

# with open("sp500tickers.pickle", "rb") as f:
#     tickers = pickle.load(f)
# print(type(tickers))

# ticker = "ABBV"
# index = tickers.index(ticker)
# print(index)

# for i in range(time_steps, training_set_scaled.shape[0]):
#         y_train.append(training_set_scaled[i,index])

y_test = np.array(y_test)
X_test = np.array(X_test)

X_train = np.reshape(X_train, (X_train.shape[1], X_train.shape[2], X_train.shape[0]))
y_train = np.reshape(y_train, (y_train.shape[1], y_train.shape[0]))
print(X_train.shape)
print(y_train.shape)

print(device_lib.list_local_devices())

##########################################################

TOTAL_DATASET = pd.read_csv('sp500_joined_close.csv')
real_stock_price_unscaled = TOTAL_DATASET.iloc[-5060:, 1:].values
np.nan_to_num(real_stock_price_unscaled, copy=False)

sc = MinMaxScaler(feature_range = (0,1))
sc.fit(real_stock_price_unscaled)

real_stock_price = sc.transform(real_stock_price_unscaled)

train_test_split_amount = 0.95
train_size = int(len(real_stock_price) * train_test_split_amount)
train_size = 4810
trainset_x, testset_x = real_stock_price[0:train_size,:], real_stock_price[train_size- 60:len(real_stock_price)]
print('Observations: %d' % (len(real_stock_price)))
print('Training Observations: %d' % (len(trainset_x)))
print('Testing Observations: %d' % (len(testset_x)))

time_steps = 60

ticker_name = 'BBY'
with open("sp500tickers.pickle", "rb") as f:
    tickers = pickle.load(f)

index = tickers.index(ticker_name)
y_total = real_stock_price_unscaled[:,index]
y_total = np.reshape(y_total, (-1,1))
y_scalar = MinMaxScaler(feature_range = (0,1))
y_scalar.fit(y_total)

y_train = real_stock_price_unscaled[time_steps:train_size, index]
y_train = np.reshape(y_train, (-1,1))
y_train = y_scalar.transform(y_train)
y_test = real_stock_price_unscaled[train_size:, index]
y_test = np.reshape(y_test, (-1,1))
y_test = y_scalar.transform(y_test)


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
            X_test = np.concatenate((X_test, X_test_temp), axis = 2)
    return X_test

X_train= preprocess_FR(trainset_x)
X_test = preprocess_FR(testset_x)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# regressor = Sequential()

# regressor.add(LSTM(units = 256,stateful=True, return_sequences = True, batch_input_shape = (25, X_train.shape[1],X_train.shape[2])))
# regressor.add(Dropout(0.2))

# regressor.add(LSTM(units = 256,stateful=True, return_sequences = True))
# regressor.add(Dropout(0.2))

# regressor.add(LSTM(units = 256, stateful=True,return_sequences = True))
# regressor.add(Dropout(0.2))

# regressor.add(LSTM(units = 256,stateful=True))
# regressor.add(Dropout(0.2))

# regressor.add((Dense(units= 1024)))
# regressor.add(Dropout(0.2))

# regressor.add(Dense(units = y_train.shape[1]))

# regressor.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=['mean_squared_error'])

regressor = Sequential()

regressor.add(LSTM(units = 256, return_sequences = True, input_shape = (X_train.shape[1],X_train.shape[2])))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 256,return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 256, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 256))
regressor.add(Dropout(0.2))

regressor.add((Dense(units= 1024)))
regressor.add(Dropout(0.2))

regressor.add(Dense(units = y_train.shape[1]))

regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

regressor.fit(X_train, y_train, epochs= 100, batch_size = 25)

regressor.save('256_BBY_UNSTATEFUL.h5')
regressor.save_weights('256_BBY_UNSTATEFUL_weights.h5')


regressor = load_model('256,stateful,timedistributed_2000_allstocks.h5')

############### DISPLAY GRAPH ##############


predicted_stock_price = regressor.predict(X_test, batch_size = 25)

predicted_stock_price = y_scalar.inverse_transform(predicted_stock_price)

y_test = y_scalar.inverse_transform(y_test)



plt.plot(y_test[-60:], color = 'red', label = 'Real {} Stock Price'.format(ticker_name))
plt.plot(predicted_stock_price[-60:], color='blue', label='Predicted {} Stock Price'.format(ticker_name))
plt.title('{} Stock Price Prediction'.format(ticker_name))
plt.xlabel('Time in days')
plt.ylabel('{} Stock Price'.format(ticker_name))
plt.legend()
plt.show()




plt.plot(y_train[-210:], color = 'red', label = 'Real {} Stock Price'.format(ticker_name))
plt.plot(X_train[-150:,0,index], color='blue', label='Predicted {} Stock Price'.format(ticker_name))
plt.title('{} Stock Price Prediction'.format(ticker_name))
plt.xlabel('Time in days')
plt.ylabel('{} Stock Price'.format(ticker_name))
plt.legend()
plt.show()







plt.plot(real_stock_price_unscaled[(real_stock_price_unscaled.shape[0] - X_test1.shape[0]):,index], color = 'red', label = 'Real {} Stock Price'.format(ticker_name))




