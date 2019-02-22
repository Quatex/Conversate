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
from sklearn import svm, cross_validation, neighbors
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
import warnings
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

def process_data_for_labels(ticker):
    hm_days = 7
    df = pd.read_csv('modifiedsp500.csv', index_col=0)
    tickers = df.columns.values.tolist()
    df.fillna(0, inplace=True)

    for i in range(1, hm_days+1):
        df['{}_{}d'.format(ticker, i)] = (df[ticker].shift(-i) - df[ticker]) / df[ticker]
    
    df.fillna(0,inplace=True)
    return tickers,df


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

    # mat = confusion_matrix(y_test, predictions)
    # sns.heatmap(mat, square=True, annot=True, cbar=False)
    # plt.xlabel('predicted value')
    # plt.ylabel('true value');

    # plt.show()
    return confidence

do_ml('BAC')