import os
import bs4 as bs
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import datetime as dt
from collections import Counter 
import quandl
import pandas_datareader
import requests
import pandas_datareader.data as web

quandl.ApiConfig.api_key = 'TrrV5dCZUX4fwAY1UyqU'  # Optional
#quandl.ApiConfig.api_version = '2015-04-09'  # Optional
token = "3Z31rgQO0c9w9SyoQBoBs7DVxTVTOObF8Wj1lYEwxpsjX8gl7vaQ7co1UyM0"


# ibm = quandl.get("WIKI/IBM", start_date="2000-01-01", end_date="2012-01-01", collapse="monthly", returns="pandas")
# print(ibm.head())
# print(ibm.tail())

#quandl.bulkdownload("WIKI")
# ticker.to_csv("test.csv")
#print(pandas_datareader.get_data_yahoo("GOOG").Close.tail())

def save_sp500_tickers():
    resp = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, "lxml")
    table = soup.find('table', {'class':'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[1].text
        tickers.append(ticker)
    with open("sp500tickers.pickle", "wb") as f:
        pickle.dump(tickers, f)
    print(tickers)
    return tickers

#save_sp500_tickers()

def get_data_from_yahoo(reload_sp500=False):
    if reload_sp500:
        tickers = save_sp500_tickers()
    else:
        with open("sp500tickers.pickle", "rb") as f:
            tickers = pickle.load(f)
    if not os.path.exists('stock_dfs'):
        os.makedirs('stock_dfs')

    start = dt.datetime(1980, 1, 1)
    end = dt.datetime.now()
    for ticker in tickers:
        # just in case your connection breaks, we'd like to save our progress!
        if not os.path.exists('stock_dfs/{}.csv'.format(ticker)):
            df = web.DataReader(ticker, 'yahoo', start, end)
            # df.reset_index(inplace=True)
            # df.set_index("Date", inplace=True)
            # df = df.drop("Symbol", axis=1)
            df.to_csv('stock_dfs/{}.csv'.format(ticker))
        else:
            print('Already have {}'.format(ticker))


get_data_from_yahoo()