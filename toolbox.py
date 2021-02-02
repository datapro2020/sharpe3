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
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, Label, LabelSet, Range1d
import hvplot.pandas
import altair as alt

#from math import sqrt
#import  pylab as pl



rf = 0 #Risk free return

#Fortmat output
pct = lambda x: '{:.2%}'.format(x)
dig = lambda x: '{:.2f}'.format(x)
to_float = lambda x: float(x.strip('%'))/100

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

# Max Sharpe with Min correlation
def MaxSharpe_MinCorr(new_df, sharpe, asset, num):
    sharpe = sharpe.drop(asset)    
    max_sharpe = sharpe.sort_values(ascending = False).head(num)
    list_max = np.array(max_sharpe.index.values)
    porfolio_A = new_df[list_max].corrwith(new_df[asset]).abs().sort_values(ascending = True).head(9)  
    porfolio_A = pd.Series(porfolio_A.index.values)
    return porfolio_A

# Min correlation with Max Sharpe
def MinCorr_MaxSharpe(new_df, asset, num):
    porfolio_B = new_df.corrwith(new_df[asset]).abs().sort_values(ascending = True).head(num)
    new_df = new_df[porfolio_B.index.values]
    f_sharpe = (250**0.5)*(new_df.mean()/new_df.std())    
    f_sharpe.sort_values(ascending = False)
    max_sharpe = f_sharpe.head(9)
    porfolio_B = pd.Series(max_sharpe.index.values)
    return porfolio_B



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
        #print(counter, symbol)
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



def Plot_P_Optimization(df):
    
    df = df.reset_index()
    df['index'] = df['index'].apply(str)
    df['Return'] = df['Return'].apply(to_float)
    df['Volatility'] = df['Volatility'].apply(to_float)
    
    points = alt.Chart(df).mark_point().encode(alt.X('Volatility:Q',scale=alt.Scale(zero=False)
    ),y='Return:Q')
    
    text = points.mark_text(align='left',baseline='middle',dx=7).encode(text='index')
    
    return points + text


#K-Means Clustering
def Clustering(ann_mean, ann_std):
        ret_var = pd.concat([ann_mean, ann_std], axis = 1).dropna()
        ret_var.columns = ["Return","Volatility"]

        X =  ret_var.values #Converting ret_var into nummpy arraysse = []for k in range(2,15):

        #pl.scatter(X[:,1],X[:,0], c = kmeans.labels_, cmap ="rainbow")
        #plt.scatter(centroids[:,1],centroids[:,0], marker = 'x', color = 'b', label = 'Centroids')
        kmeans = KMeans(n_clusters = 5).fit(X)
        #centroids = kmeans.cluster_centers_
        #Company = pd.DataFrame(ret_var.index)
        cluster_labels = pd.DataFrame(kmeans.labels_, index=ret_var.index, columns=['Clustering'])
        #tupper = pd.concat([tupper, cluster_labels],axis = 1)
        print ('Builing Clustering with the ML Library K-Means') 
        return cluster_labels
    
#Performace for each stock
def Performance(p):
    window = ['D','W','M','3M','6M','A']
    perform = pd.DataFrame(0, index = p.columns, columns=window)

    for i in window:    
        df = p.resample(i).last().pct_change().tail(2)
        perform[i] = df.iloc[0,:].apply(pct) 
    
    df = p.resample('Y').last().pct_change().tail(1)
    perform['YTD'] = df.iloc[0,:].apply(pct)
    return perform




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

    return df, optimal_risky_port, min_vol_port

