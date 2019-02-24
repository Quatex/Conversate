import os
import bs4 as bs
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import datetime as dt
from collections import Counter 
import quandl

quandl.ApiConfig.api_key = 'TrrV5dCZUX4fwAY1UyqU'  # Optional
#quandl.ApiConfig.api_version = '2015-04-09'  # Optional

ibm = quandl.get("WIKI/IBM", start_date="2000-01-01", end_date="2012-01-01", collapse="monthly", returns="pandas")
print(ibm.head())
print(ibm.tail())