import os
import bs4 as bs
import matplotlib.pyplot as plt
from matplotlib import style
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
import pickle
import datetime as dt
from collections import Counter 
import quandl
import pandas_datareader
import requests
import pandas_datareader.data as web
from mpl_finance import candlestick_ohlc
import csv

style.use('ggplot')
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
    #for sp500 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    resp = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, "lxml")
    table = soup.find('table', {'class':'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[1].text
        tickers.append(ticker)
    with open("sp500tickers.pickle", "wb") as f:
        pickle.dump(tickers, f)
    return tickers

#save_sp500_tickers()

#first date = 1980, 1,1
def get_data_from_yahoo(reload_sp500=False):
    if reload_sp500:
        tickers = save_sp500_tickers()
    else:
        with open("sp500tickers.pickle", "rb") as f:
            tickers = pickle.load(f)
    if not os.path.exists('stock_test'):
        os.makedirs('stock_test')

    start = dt.datetime(2019, 2, 23)
    end = dt.datetime.now()
    for ticker in tickers:
        # just in case your connection breaks, we'd like to save our progress!
        if not os.path.exists('stock_test/{}.csv'.format(ticker)):
            df = web.DataReader(ticker, 'yahoo', start, end)
            # df.reset_index(inplace=True)
            # df.set_index("Date", inplace=True)
            # df = df.drop("Symbol", axis=1)
            df.to_csv('stock_test/{}.csv'.format(ticker))
        else:
            print('Already have {}'.format(ticker))

df = web.DataReader("^HSI", 'yahoo', dt.datetime(2019,2,23), dt.datetime.now())
print(df.head())
print(df.tail())
# get_data_from_yahoo()

# with open("sp500tickers.pickle", "rb") as f:
#     current_tickers = pickle.load(f)

# print(current_tickers)

def update_stocks(reload_sp500=False):
    if reload_sp500:
        get_data_from_yahoo(reload_sp500)

    if not os.path.exists('test'):
        print("Fix stock directory")
    else: 
        os.chdir(os.getcwd() + '\\test\\')
        for dirc in os.walk(os.getcwd(), topdown=True):
            for comps in dirc[2]:
                print("comps is:", comps)
                """
                Open up the csv and read the last line.
                [-1] accesses the last row of the csv file.
                [:10] gets the first 10 characters at index 0-9 of the last line.
                Should be equivalent to the date YYYY-MM-DD = 10 characters.
                """
                last_update = open(comps).readlines()[-1][:10]

                # while last_update.isspace():
                    
                # if (last_update.isspace()):
                #     print("line is empty")
                #     last_update = open(comps).readlines()[-2][:10]
                # else:
                #     print("last line is not empty")
                #     print("last update is:", last_update)

                #Split the string to be put into the datetime class.
                s_year, s_month, s_day = last_update.split('-')

                #Get today's date.
                e_year, e_month, e_day = str(dt.datetime.now().date()).split('-')
                
                #Check to see if the csv even needs to be updated. If it is move to next file.
                if(s_year == e_year and s_month == e_month and s_day == e_day):
                    print(comps,'is up to date.')
                    continue
                else:

                    #Create an end and start to put into the pandas dataframe.
                    end = dt.datetime(int(e_year), int(e_month), int(e_day))
                    start = dt.datetime(int(s_year), int(s_month), int(s_day))

                    #Get the ticker from the name of the file.
                    tckr = comps.split('.')[0]

                    #Get the daily stock information between last update and today.
                    df = web.DataReader(tckr, 'yahoo', start, end)

                    """
                    Convert the pandas dataframe into a string to be written by csv.
                    [1:] gets rid of first date because that date is already in csv from previous access.
                    The [:-1] gets rid of the last string because it is always empty.
                    header=None removes the column information so we can just take the information we want.
                    """
                    to_csv = str(df.to_csv(header=None)).split('\n')[1:-1]

                    #Write the new string to csv.
                    with open(comps, 'a', newline='') as f:
                        writer = csv.writer(f, delimiter='\n')
                        writer.writerow(to_csv)

                        #Confirm that we successfully updated the csv file.
                        print(comps,'updated')

# update_stocks()

# df = pd.read_csv("test/AAPL.csv")
# print(df.head())
# print(df.tail())
# df.to_csv("test.csv")


def group_everything():
    with open("sp500tickers.pickle", "rb") as f:
        tickers = pickle.load(f)

    main_df = pd.DataFrame()

    for count, ticker in enumerate(tickers):
        df = pd.read_csv('stock_test/{}.csv'.format(ticker))
        df.set_index('Date', inplace= True)

        df.rename(columns = {'Adj Close': ticker}, inplace= True)
        df.drop(['Open','High','Low','Close','Volume'], 1, inplace= True)

        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df, how= 'outer')
        
        if count % 10 == 0:
            print(count)
            
    print(main_df.head())
    main_df.to_csv('TEST_sp500_joined_close_TEST.csv')

# group_everything()
