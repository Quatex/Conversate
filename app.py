import plotly.plotly as py
import plotly.graph_objs as go
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
from collections import deque
import pickle
from textwrap import dedent as d
import json


app = dash.Dash()
app.config['suppress_callback_exceptions']=True
with open("sp500tickers.pickle", "rb") as f:
    tickers = pickle.load(f)
df = pd.read_csv('stock_dfs/AAPL.csv')
stock = 'AAPL'
app.layout = html.Div([
    html.H1(children="Stock Prediction Graphs"),
    dcc.Dropdown(id='stock-ticker',
        options = [{'label': s, 'value': s} for s in tickers],
        placeholder = "Type or Select a Stock Ticker",
        multi=True 
        ),
    html.Div(id = 'graphs'),
    html.Div([dcc.Markdown(d(" **Hover Data** Mouse over Values")),
                                    html.Pre(id = 'hover-data') ]),
    html.Div([dcc.Markdown(d(" **Click Data** Mouse over Values")),
                                    html.Pre(id = 'click-data') ])
                      ],
    className = 'container', style = {'width':'98%', 'margin-left':10, 'margin-right':10, 'max-width':50000}
)


@app.callback(
    Output(component_id = 'graphs', component_property = 'children'),
    [Input(component_id='stock-ticker', component_property='value')]
)
def update_graph(value):
    graphs = []
    if not value:
        graphs.append(html.H3(
            "Select a stock ticker.",
            style={'marginTop': 20, 'marginBottom': 20}
        ))
    else:
        for ticker in value:
            df = pd.read_csv('stock_dfs/{}.csv'.format(ticker))
            df.set_index('Date', inplace= True)
            difference_df = df['Close']
            difference_df = difference_df.diff()
            difference_df_tmr = difference_df.shift(-1)
            difference_df_tmr = difference_df_tmr[:-1]
            difference = []
            for index in range(len(difference_df)- 1):
                difference.append("Change since yesterday: {} </br> Change from tomorrow: {}".format(difference_df.iloc[index], difference_df_tmr.iloc[index]))
            difference.append("Change since yesterday: {} </br> Change from tomorrow: {}".format(difference_df.iloc[index], "NaN"))
            
            candlestick = {
                'x': df.index,
                'open': df['Open'],
                'high': df['High'],
                'low': df['Low'],
                'close': df['Close'],
                'type': 'candlestick',
                'name': ticker,
                'legendgroup': ticker,
                'text':difference
            }
            graphs.append(dcc.Graph(id = ticker,
                figure={'data':[
                    {'x':df.index, 'y': df.Close,
                     'text': difference,
                     'type':'line', 'name':ticker
                     #, 'mode': 'markers'
                     },
                    
                ] + [candlestick],
            'layout':{'margin': {'b': 25, 'r': 10, 'l': 60, 't': 50},
                      'legend': {'x': 0},
                      'clickmode': 'event+select',
                      'uirevision': ticker,
                      'title': ticker}
        }
    ))
            graphs.append(html.Div([dcc.Markdown(d(" **Hover Data** Mouse over Values")),
                                    html.Pre(id = 'hover-data')]))

    return graphs

@app.callback(
    Output('hover-data', 'children'),
    [Input('graphs', 'hoverData')])
def display_hover_data(hoverData):
    return json.dumps(hoverData, indent = 2)


@app.callback(
    Output('click-data', 'children'),
    [Input('graphs', 'clickData')])
def display_click_data(clickData):
    return json.dumps(clickData, indent = 2)                                   
                                   
if __name__ == '__main__':
    app.run_server(debug=True)

