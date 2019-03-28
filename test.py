import pandas as pd 

df = pd.read_csv('stock_dfs/AAPL.csv')

df.set_index('Date', inplace= True)
print(df.index)