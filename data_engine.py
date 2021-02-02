import numpy as np
import pandas as pd
from datetime import datetime, timedelta
#import matplotlib.pyplot as plt
#from sklearn.cluster import KMeans
#from math import sqrt
#import  pylab as pl
import toolbox


D = 400
rf = 0 #Risk free return

date_D_days_ago = datetime.now() - timedelta(days=D)
now = datetime.now()

start_date = date_D_days_ago.strftime('%F')
end_date = now.strftime('%F')

#Fortmat output
pct = lambda x: '{:.2%}'.format(x)
dig = lambda x: '{:.2f}'.format(x)



    
# Read industry and sectors of stock market
info = pd.read_csv('data/S&P500-Info.csv', index_col=['Symbol'])
info = pd.DataFrame(info)



custom_date_parser = lambda x: datetime.strptime(x,"%Y-%m-%d")
price = pd.read_csv('data/price.csv', index_col=['Date'], parse_dates=['Date'], date_parser=custom_date_parser)
price = pd.DataFrame(price)

# Read  Tickers
tickers = price.columns



# Expected annualized Return, Volatility, Correlation and Sharpe
ann_mean = price.pct_change().apply(lambda x: np.log(1+x)).mean().apply(lambda x: x*250)
ann_std = price.pct_change().apply(lambda x: np.log(1+x)).std().apply(lambda x: x*np.sqrt(250))
corr = price.pct_change().apply(lambda x: np.log(1+x)).corr()
cov  = price.pct_change().apply(lambda x: np.log(1+x)).cov()
Sharpe = (ann_mean - rf)/ann_std
perform_df = toolbox.Performance(price)

f = lambda x: '{:.2%}'.format(x)

# Building Tupperware
tupper = pd.DataFrame(ann_mean, columns=['Exp Return'], index = tickers)
tupper ['Volatility'] = pd.DataFrame(ann_std, columns=['Volatility'])
tupper ['Sharpe'] = pd.DataFrame(Sharpe, columns=['Sharpe'])
tupper ['Min_Corr'] = pd.DataFrame(corr.abs().idxmin(), columns=['Min_Corr'])
tupper ['Corr_value'] = pd.DataFrame(corr.min(), columns=['Corr_Value'])

cluster_labels = toolbox.Clustering(ann_mean, ann_std)
tupper = pd.concat([tupper, cluster_labels,perform_df],axis = 1)

print ('DataFrame built')  
  


portfolio_df= pd.DataFrame(0, columns = ['a0','a1','a2','a3','a4','a5','a6','a7','a8','a9'], index = tickers)                                            
portfolio_df['a0'] = portfolio_df.index.values

min_vol_port_df = pd.DataFrame(0, columns = ['mv_Returns','mv_Volatility','mv_w0','mv_w1','mv_w2','mv_w3','mv_w4','mv_w5','mv_w6','mv_w7','mv_w8','mv_w9'], index = tickers)
optimal_risky_port_df = pd.DataFrame(0, columns = ['or_Returns','or_Volatility','or_w0','or_w1','or_w2','or_w3','or_w4','or_w5','or_w6','or_w7','or_w8','or_w9'], index = tickers)
    


#Building porfolios 
for r in portfolio_df.index.values:
    Test_A = toolbox.MaxSharpe_MinCorr(price, Sharpe, r, 100)
    for c in range(9):
        portfolio_df.loc[r,'a'+str(c+1)] = Test_A[c]
         
ti = 0
    
for s in portfolio_df.index.values:
    p = price[portfolio_df.loc[s,:]]
    ti = ti +1
    portfolios = toolbox.P_Optimization (p)
    min_vol_port = portfolios.loc[portfolios['Volatility'].idxmin()]
    optimal_risky_port = portfolios.loc[((portfolios['Returns']- rf)/portfolios['Volatility']).idxmax()]
    print('Porfolio '+str(ti)+' optimized.')
              
    # Plotting optimal portfolio
    #portfolios.plot.scatter(x='Volatility', y='Returns', marker='o', s=10, alpha=0.3, grid=True, figsize=[10,10])
    #Plot_P_Optimization(portfolios,p)
        
    #Saving optimal porfolios weights in a DataFrame
    min_vol_port_df.loc[s,:]  = min_vol_port.values
    optimal_risky_port_df.loc[s,:] = optimal_risky_port.values
    
   
print('\n Porfolio creation ='+str(ti)) 

#Saving data
tupper = pd.concat([tupper,info,portfolio_df,min_vol_port_df,optimal_risky_port_df], axis=1)
tupper.to_csv('data/tupper_'+end_date+'.csv')

print ('Done! /n Tupperware Data created on '+end_date)