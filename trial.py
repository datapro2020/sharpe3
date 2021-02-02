#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 15:24:41 2021

@author: datapro
"""

import numpy as np
import pandas as pd
import pandas_datareader.data as web
from datetime import datetime, timedelta
import matplotlib.pyplot as plt


#Get Symbol from Security
def getTicker(stock, df):
    p = df[df['Security'].str.contains(stock).tolist()]
    ticker = p.index.values[0]
    return ticker

#Get Security from Symbol
def getStock(ticker, df):
    stock = df.loc[ticker,'Security']
    return stock
    

# Read SP500 Tickers and Information
sp500_symbols = pd.read_csv ('data/S&P500-Symbols.csv')
sp500 = np.array(sp500_symbols['Symbol'])
info = pd.read_csv('data/S&P500-Info.csv', index_col=['Symbol'])
info = pd.DataFrame(info)


# Get Data from Yahoo Finance API (TO REVIEW)
#price = StockData(sp500, start_date, end_date)
#SPY  =  StockData('SPY', start_date, end_date)

custom_date_parser = lambda x: datetime.strptime(x,"%Y-%m-%d")
price = pd.read_csv('data/price.csv', index_col=['Date'], parse_dates=['Date'], date_parser=custom_date_parser)
price = pd.DataFrame(price)

tupper = pd.read_csv('data/tupper_2021-01-14.csv', index_col=['Unnamed: 0'])
tupper = pd.DataFrame(tupper)

print('-'*80)
print('-'*80)
print ('DISCLAIMER: Project3 provides data regarding public stock market.\nIt does NOT recommend or advice for any investment.\nData shared are a mathematical modeling based in public information.')
print('-'*80)

stock = input('Enter your stock: ')
ticker = getTicker(stock, info)
stock = getStock(ticker,info)

print('-'*80)
print('\n'*2)
print (stock)
print('\n')
p = tupper.loc[ticker,'Exp Return']
print("Annualised Return: "+"{:.2%}".format(p))
print("Annualised Volatility: "+str(tupper.loc[ticker,'Volatility']))
      

print('-'*80)


