#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 10:32:50 2021

@author: datapro
"""

import numpy as np
import pandas as pd
import pandas_datareader.data as web
from datetime import datetime, timedelta

def StockData(ticker, start_d, end_d):
    data = web.get_data_yahoo(ticker, start = start_d, end = end_d)
    price =  pd.DataFrame(data['Adj Close'])
    volume = pd.DataFrame(data['Volume'])
    return price, volume

D=400

date_D_days_ago = datetime.now() - timedelta(days=D)
now = datetime.now()

start_date = date_D_days_ago.strftime('%F')
end_date = now.strftime('%F')


sp500_symbols = pd.read_csv ('data/S&P500-Symbols.csv')
sp500 = np.array(sp500_symbols['Symbol'], )

price, volume = StockData(sp500,start_date,end_date )

#Cleaning data
price = price.dropna(axis=1, how='all')
volume = volume.dropna(axis=1, how='all')

price.to_csv('data/price.csv', index = 'date')
volume.to_csv('data/volume.csv', index = 'date')
p = price.columns.isna().any()
v = volume.columns.isna().any()


print('Price has columns '+str(p)+' data' )
print('Volume has columns'+str(v)+' data' )

