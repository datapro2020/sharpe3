#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 10:42:08 2021

@author: datapro
"""
import numpy as np
import pandas as pd
import pandas_datareader.data as web
from datetime import datetime, timedelta
#import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
#from bokeh.plotting import figure
#from bokeh.models import ColumnDataSource, Label, LabelSet, Range1d
#import hvplot.pandas
import altair as alt
import io
import dropbox
from fbprophet import Prophet
import yahoo_fin.stock_info as yf
import json
from pandas.io.json import json_normalize
import requests
import requests_html


#from math import sqrt
#import  pylab as pl



rf = 0 #Risk free return

#Fortmat output
pct = lambda x: '{:.2%}'.format(x)
dig = lambda x: '{:.2f}'.format(x)
to_float = lambda x: float(x.strip('%'))/100
to_date = lambda x: x.strftime('%F')
# Timing
D=400
date_D_days_ago = datetime.now() - timedelta(days=D)
now = datetime.now()
start_date = date_D_days_ago.strftime('%F')
end_date = now.strftime('%F')

US_index = ['^GSPC','^IXIC']
gold = 'GC=F'
oil  = 'CL=F'
bitcoin = 'BTC-USD'

#Data from Yahoo Finance
def StockData(ticker):
    data = web.get_data_yahoo(ticker, start = start_date, end = end_date)
    price =  pd.DataFrame(data['Adj Close'])
    #volume = pd.DataFrame(data['Volume'])
    price = price.dropna(axis=1, how='all')
    return price




def My_Corr(df1,df2):
    df1 = df1.reindex(df2.index, method='pad')
    df = pd.concat([df1,df2],axis=1)
    df = df.pct_change().apply(lambda x: np.log(1+x)).corr()
    return df



def GetTupper():
    
    tk ='5-UkyaE_0XoAAAAAAAAAAb-BCtdL-qKmMTbSNOKdSSXwxA5hFBjrERMGyHcjInpW'
    
    DBX = dropbox.Dropbox(tk)

    _, read = DBX.files_download("/data/tupper.csv")
    
    with io.BytesIO(read.content) as stream:
        df = pd.read_csv(stream, index_col=0)
    return df



# Pofolio Optimization and Efficient Frontier (TO REVIEW)
def P_Optimization(df):
    ind_er = df.pct_change().apply(lambda x: np.log(1+x)).mean().apply(lambda x: x*250)
    cov_matrix = df.pct_change().apply(lambda x: np.log(1+x)).cov()
    #corr_matrix = df.pct_change().apply(lambda x: np.log(1+x)).corr()
    ann_sd = df.pct_change().apply(lambda x: np.log(1+x)).std().apply(lambda x: x*np.sqrt(250))

    p_ret = [] # Define an empty array for portfolio returns
    p_vol = [] # Define an empty array for portfolio volatility
    p_weights = [] # Define an empty array for asset weights

    num_assets = len(df.columns)
    num_portfolios = 5000
    
    for portfolio in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights = weights/np.sum(weights)
        weights= weights.round(2)
        p_weights.append(weights)
        returns = np.dot(weights, ind_er) # Returns are the product of individual expected returns of asset and its 
                                      # weights 
        p_ret.append(returns)
        var = cov_matrix.mul(weights, axis=0).mul(weights, axis=1).sum().sum()# Portfolio Variance
        sd = np.sqrt(var) # Daily standard deviation
        ann_sd = sd*np.sqrt(250) # Annual standard deviation = volatility
        p_vol.append(ann_sd)
    
    data = {'Returns':p_ret, 'Volatility':p_vol}

    for counter, symbol in enumerate(df.columns.tolist()):
      
        data[symbol+' weight'] = [w[counter] for w in p_weights]

    portfolios  = pd.DataFrame(data) #Dataframe of the 5000 portfolios created
    min_vol_port = portfolios.loc[portfolios['Volatility'].idxmin()]
    optimal_risky_port = portfolios.loc[((portfolios['Returns']- rf)/portfolios['Volatility']).idxmax()]
    return optimal_risky_port, min_vol_port

def GetWeights(or_p,mv_p):
    df = pd.concat([or_p,mv_p], axis=1)
    columns_name = ['Maximum Sharpe Ratio', 'Minimum Volatility']
    df.columns = columns_name
    df = df.drop('Returns',axis=0)
    df = df.drop('Volatility',axis=0)
    df.applymap(dig)
    return df


fecha = lambda x: x.strftime('%F')

def Plot_Performance1(df):
  
    df = df.reset_index()
    df = df.melt('Date', var_name='ticker', value_name='price')
    df['price'] = df['price'].apply(dig)
    pic = alt.Chart(df).mark_area().encode(
        x="Date:T",
        y=alt.Y("price:Q", stack="normalize"),
        color="ticker:N",
        tooltip=[ 'Date:T','ticker:N','price:N']
    ).properties(height=400, width=800)
    return pic    

def Plot_Performance2(df):

    df = df.reset_index()

    df = df.melt('Date', var_name='ticker', value_name='price')
    df['price'] = df['price'].apply(dig)

    nearest = alt.selection(type='single', nearest=True, on='mouseover', fields=['Date'], empty='none')


    line = alt.Chart(df).mark_line().encode(
        alt.X('Date:T',),
        alt.Y('price:Q'),
        color='ticker:N',
        ).properties(height=400, width=800)


    selectors = alt.Chart(df).mark_point().encode(
        x='Date:T',
        opacity=alt.value(0),
        ).add_selection(
        nearest
        )

    points = line.mark_point().encode(
        opacity=alt.condition(nearest, alt.value(1), alt.value(0))
        )

    text = line.mark_text(align='left', dx=5, dy=-5).encode(
        text=alt.condition(nearest, 'price:Q', alt.value(' '))
        )

    rules = alt.Chart(df).mark_rule(color='gray').encode(
        x='Date:T',
        ).transform_filter(
        nearest
        )

    pic = alt.layer(
        line, selectors, points, rules, text
        ).properties(
            width=800, height=400
        )

    return pic

def Plot_Performance3(df,us):
    df = df.join(us, how='outer')
    df = df/df.iloc[0]
    df = df.reset_index()
    df = df.melt('Date', var_name='ticker', value_name='price')
    df['price'] = df['price'].apply(dig)

    selection = alt.selection_multi(fields=['ticker'], bind='legend')

    pic = alt.Chart(df).mark_line().encode(
        x='Date:T',
        y='price:Q',
        color='ticker:N',
        strokeDash='ticker:N',
        opacity=alt.condition(selection, alt.value(1), alt.value(0.2)),
        tooltip=[ 'Date:T','ticker:N','price:N']
    ).add_selection(selection).properties(
            width=800, height=400
        )
    return pic

def Plot_P_Optimization(df):
    
    df = df.reset_index()
    df['index'] = df['index'].apply(str)
    df['Return'] = df['Return'].apply(to_float).apply(dig)
    df['Volatility'] = df['Volatility'].apply(to_float).apply(dig)
    
    points = alt.Chart(df).mark_point().encode(alt.X('Volatility:Q',scale=alt.Scale(zero=False)
    ),y='Return:Q', color='index:N', tooltip=['index:N', 'Volatility:N','Return:N'])
    
    text = points.mark_text(align='left',baseline='middle',dx=7).encode(text='index')
    
    return points + text


#Forecasting based in FB Prophet
def Forecast(df,ticker):
    df = df.fillna(0)
    tit = ''+ticker
    df = df.reset_index()
    df.columns = ['ds','y']
    prophet = Prophet()
    prophet.fit(df)
    future_prices=prophet.make_future_dataframe(periods=365)
    future = prophet.predict(future_prices)
    future = future[['ds','trend','yhat_lower','yhat_upper']]
    future['trend'] = future['trend'].apply(dig)
    future['yhat_lower'] = future['yhat_lower'].apply(dig)
    future['yhat_upper'] = future['yhat_upper'].apply(dig)
    

    future = future.melt('ds', var_name='bands', value_name='price')
    pic = alt.Chart(future, title=tit).mark_line().encode(
        x='ds:T',
        y='price:Q',
        color='bands:N', tooltip=['bands:N', 'ds:N','price:N']
    )
    return pic


def Plot_EPS(price):
    performance = (price.resample('W').ffill().pct_change() + 1).cumprod().fillna(0).tail(1)
    df = pd.DataFrame(0, index= price.columns, columns =['performance','EPS'])
    df['performance'] = performance.tail(1)
    df['performance'] = performance.iloc[0,:]

    for i in df.index:
        EPS = yf.get_quote_table(i)
        df.loc[i,'EPS'] = EPS["EPS (TTM)"]

    df = df.reset_index()

    pic = alt.Chart(df).mark_point().encode(alt.X('performance:Q',scale=alt.Scale(zero=False)
        ),y='EPS:Q', 
        color='Symbols:N', 
        size=alt.Size("performance:Q", scale=alt.Scale(range=[0, 1000])), 
        tooltip=['Symbols:N', 'performance:N','EPS:N']).properties(height=400, width=800)
    return pic

def Galaxy(df):
    df.index.name='ticker'
    df = df.reset_index()

    df['performance'] = df['performance'].apply(to_float)
    df['yhat_lower'] = df['yhat_lower'].apply(to_float)
    df['Sharpe'] = df['Sharpe'].apply(dig)

    selection = alt.selection_multi(fields=['Sector'], bind='legend')
    scales = alt.selection_interval(bind='scales')
    pic = alt.Chart(df).mark_point().encode(x='performance:Q',y='yhat_lower:Q',color='Sector:N',
        size=alt.Size("Market Cap:Q", scale=alt.Scale(range=[0, 1000])),
        opacity=alt.condition(selection, alt.value(1), alt.value(0.2)),
        tooltip=['ticker:N', 'Sharpe','performance:N','trend','yhat_lower:N','Name:N','Country:N','Sector:N','Industry:N','IPO Year:N','Market Cap:N']).properties(
        width=800, height=800
        ).add_selection(
        scales
    ).add_selection(selection)
    return pic

#K-Means Clustering
def Clustering(ann_mean, ann_std):
        ret_var = pd.concat([ann_mean, ann_std], axis = 1).dropna()
        ret_var.columns = ["Return","Volatility"]

        X =  ret_var.values #Converting ret_var into nummpy arraysse = []for k in range(2,15):
        df = pd.DataFrame(X, index=ret_var.index, columns=['Return','Volatility'])
  
        kmeans = KMeans(n_clusters = 5).fit(X)
        centroids = kmeans.cluster_centers_
        cluster_labels = pd.DataFrame(kmeans.labels_, index=ret_var.index, columns=['Cluster'])
        df = pd.concat([df, cluster_labels],axis = 1) 
       
        return df
    
#Performace for each stock
def Performance(p):
    p =(p.resample('W').ffill().pct_change() + 1).cumprod().fillna(0)
    return p


def Daily_info():
    win = yf.get_day_gainers()
    win = win.sort_values(by='Market Cap',ascending = False).head(5).sort_values(by='% Change',ascending = False)
    lose = yf.get_day_losers()
    lose = lose.sort_values(by='Market Cap',ascending = False).head(5).sort_values(by='% Change',ascending = True)
    active = yf.get_day_most_active()
    active = lose.sort_values(by='Market Cap',ascending = False).head(5)
    return win,lose,active

#df1 in from Tupper. df2 is from input
def Join_Df(df1, df2):
    for i in df2.index:
        if df1.index.str.match(i).any()== True:
            df1.loc[i,:] = df2.loc[i,:]
        else:
            df1 = df1.append(df2.loc[i,:])
    return df1

def Core_Calculations(portfolio, price):

    #price = StockData(portfolio,start_date,end_date )
    #%time
    #Cleaning data
    #price = price.dropna(axis=1, how='all')
    #volume = volume.dropna(axis=1, how='all')
    
    
    #Calculations for annual view
    ann_mean = price.pct_change().apply(lambda x: np.log(1+x)).mean().apply(lambda x: x*250)
    ann_std = price.pct_change().apply(lambda x: np.log(1+x)).std().apply(lambda x: x*np.sqrt(250))
    #corr = price.pct_change().apply(lambda x: np.log(1+x)).corr()
    #cov  = price.pct_change().apply(lambda x: np.log(1+x)).cov()
    Sharpe = (ann_mean - rf)/ann_std

    #Optimization
    optimal_risky_port, min_vol_port= P_Optimization(price)
    
    #Aggregation of Returns, Volatility and Sharpes
    indice = portfolio+['Minimum Volatility','Maximum Sharpe ratio']
    df = pd.DataFrame(0, index = indice, columns = ['Return','Volatility','Sharpe'])
    for i in portfolio:
        df.loc[i,'Return'] = ann_mean[i]
        df.loc[i,'Volatility'] = ann_std[i]
        df.loc[i,'Sharpe'] = Sharpe[i]
    df.loc['Minimum Volatility','Return'] = min_vol_port['Returns']
    df.loc['Minimum Volatility','Volatility'] = min_vol_port['Volatility']
    df.loc['Maximum Sharpe ratio','Return'] = optimal_risky_port['Returns']
    df.loc['Maximum Sharpe ratio','Volatility'] = optimal_risky_port['Volatility']
    df.loc['Minimum Volatility','Sharpe'] = (min_vol_port['Returns'] - rf)/min_vol_port['Volatility']
    df.loc['Maximum Sharpe ratio','Sharpe'] = (optimal_risky_port['Returns'] - rf)/optimal_risky_port['Volatility']
    
    df['Sharpe'] = df['Sharpe'].apply(dig)
    df['Return'] = df['Return'].apply(pct)
    df['Volatility'] = df['Volatility'].apply(pct)

    return df, optimal_risky_port, min_vol_port, ann_mean, ann_std

