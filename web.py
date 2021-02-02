#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 15:47:09 2021

@author: datapro
"""
import numpy as np
import pandas as pd
import pandas_datareader.data as web
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.image as mping
from string import Template
import math
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvas
import plotly.graph_objects as go
import panel as pn
import hvplot.pandas
from bokeh.models import ColumnDataSource
import panel.widgets as pnw
from panel.interact import interact, interactive, fixed, interact_manual
from panel import widgets
import param



# Panel Extension to prepare the web style
pn.extension('plotly')
pn.extension(comms='ipywidgets')
pn.extension('echarts')
pn.extension()
pn.config.js_files  = {'deck': 'https://unpkg.com/deck.gl@~5.2.0/deckgl.min.js'}
pn.config.css_files = ['https://api.tiles.mapbox.com/mapbox-gl-js/v0.44.1/mapbox-gl.css']



rf = 0


#Fortmat output
pct = lambda x: '{:.2%}'.format(x)
dig = lambda x: '{:.2f}'.format(x)
exp = lambda x: math.exp(x)


# Timing
D=400
date_D_days_ago = datetime.now() - timedelta(days=D)
now = datetime.now()
start_date = date_D_days_ago.strftime('%F')
end_date = now.strftime('%F')

#Test porfolio
portfolio =['PTON', 'TSLA', 'GM','SPY']


#Data from Yahoo Finance
def StockData(ticker, start_d, end_d):
    data = web.get_data_yahoo(ticker, start = start_d, end = end_d)
    price =  pd.DataFrame(data['Adj Close'])
    volume = pd.DataFrame(data['Volume'])
    
    return price, volume

# Pofolio Optimization and Efficient Frontier (TO REVIEW)
def P_Optimization(df):
    ind_er = df.pct_change().apply(lambda x: np.log(1+x)).mean().apply(lambda x: x*250)
    cov_matrix = df.pct_change().apply(lambda x: np.log(1+x)).cov()
    corr_matrix = df.pct_change().apply(lambda x: np.log(1+x)).corr()
    ann_sd = df.pct_change().apply(lambda x: np.log(1+x)).std().apply(lambda x: x*np.sqrt(250))

    p_ret = [] # Define an empty array for portfolio returns
    p_vol = [] # Define an empty array for portfolio volatility
    p_weights = [] # Define an empty array for asset weights

    num_assets = len(df.columns)
    num_portfolios = 10000
    
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

    portfolios  = pd.DataFrame(data) #Dataframe of the 10000 portfolios created
    
    return portfolios




def Core_Calculations(portfolio):

    price, volume = StockData(portfolio,start_date,end_date )

    #Cleaning data
    price = price.dropna(axis=1, how='all')
    volume = volume.dropna(axis=1, how='all')

    #Calculations
    ann_mean = price.pct_change().apply(lambda x: np.log(1+x)).mean().apply(lambda x: x*250)
    ann_std = price.pct_change().apply(lambda x: np.log(1+x)).std().apply(lambda x: x*np.sqrt(250))
    corr = price.pct_change().apply(lambda x: np.log(1+x)).corr()
    cov  = price.pct_change().apply(lambda x: np.log(1+x)).cov()
    Sharpe = (ann_mean - rf)/ann_std

    #Optimization
    portfolios = P_Optimization(price)
    min_vol_port = portfolios.loc[portfolios['Volatility'].idxmin()]
    optimal_risky_port = portfolios.loc[((portfolios['Returns']- rf)/portfolios['Volatility']).idxmax()]

    #Aggregation of Returns, Volatility and Sharpes
    indice = portfolio+['Minimum Volatility','Maximum Sharpe ratio']
    df = pd.DataFrame(0, index = indice, columns = ['Return','Volatility','Sharpe'])
    for i in portfolio:
        df.loc[i,'Return'] = ann_mean[i]
        df.loc[i,'Volatility'] = ann_mean[i]
        df.loc[i,'Sharpe'] = Sharpe[i]
    df.loc['Minimum Volatility','Return'] = min_vol_port['Returns']
    df.loc['Minimum Volatility','Volatility'] = min_vol_port['Volatility']
    df.loc['Maximum Sharpe ratio','Return'] = optimal_risky_port['Returns']
    df.loc['Maximum Sharpe ratio','Volatility'] = optimal_risky_port['Volatility']
    df.loc['Minimum Volatility','Sharpe'] = (min_vol_port['Returns'] - rf)/min_vol_port['Volatility']
    df.loc['Maximum Sharpe ratio','Sharpe'] = (optimal_risky_port['Returns'] - rf)/optimal_risky_port['Volatility']
    
    return price, df, corr, min_vol_port, optimal_risky_port




class ActionExample(param.Parameterized):
    """
    Demonstrates how to use param.Action to trigger an update.
    """

    new_stock = param.String(default='CSCO,ZM,PTON')
    
    action = param.Action(lambda x: x.param.trigger('action'), label='Run Analytics')
        
    @param.depends('action')
    def get_P(self):
        return self.new_stock
  
   # method is watching whether model_trained is updated
    @param.depends('new_stock')
    def update_graph(self):
        if self.new_stock:
            new_portfolio = self.new_stock
            new_portfolio = new_portfolio.split(',')
            price, df, corr,mv_p,or_p = Core_Calculations(new_portfolio)
            df_pane = pn.pane.DataFrame(df)
            corr_pane = pn.pane.DataFrame(corr)
            mv_pane = pn.pane.DataFrame(mv_p)
            or_pane = pn.pane.DataFrame(or_p)
            price_plot = price.hvplot.line()
            #lista = df.index.values.tolist()
            new_plot = df.hvplot.scatter(x='Volatility', y='Return', subplots=True, padding=0.1, responsive=True, min_height=300)
            return pn.Column(price_plot,'<br><h1>Correlation Matrix</h1>',corr_pane,'<br><h1>Portfolio Optimization</h1>', df_pane,
                             '<br>',new_plot,'<br><p>Minimum Volatility Portfolio</p>',mv_pane,'<br><p>Maximum Sharpe Portfolio</p>',or_pane)
        else:
            return "Model not trained yet"

        
        
action_example = ActionExample()

pn.Column(
    '# Sharpe 3', pn.Row(
        pn.Column(pn.panel(action_example.param, show_labels=False, show_name=False, margin=0),
            'Update the portfolio.'),
        action_example.get_P, action_example.update_graph)).show()

