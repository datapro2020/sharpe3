import streamlit as st

st.set_page_config(
    page_title='Sharpe 3 - Data Driven Investing', 
    page_icon ='🚀',
    layout='wide',
    initial_sidebar_state="expanded")



import streamlit as st
import numpy as np
import pandas as pd
import time
import toolbox
from datetime import datetime, timedelta
import altair as alt



dig = lambda x: '{:.2f}'.format(x)

US_index = ['^GSPC','^IXIC']
gold = 'GC=F'
bitcoin = 'BTC-USD'

now = datetime.now()
now = now.strftime('%F')

# Getting the main stock index as for benchmarking 
@st.cache
def benchmarks():
    US = toolbox.StockData(US_index)
    US.columns = ['SP500','Nasdaq']
    BTC = toolbox.StockData(bitcoin)
    BTC.columns = ['Bitcoin']
    GOLD = toolbox.StockData(gold)
    GOLD.columns = ['Gold SPOT']
    return US,BTC,GOLD


@st.cache
def Info():
    df = pd.read_csv('data/tupper_'+now+'.csv', index_col='ticker')
    return df
 


st.markdown('<h1>Data Driving Investing</h1><br>', unsafe_allow_html=True
)



# Sidebar stuff
title_side_bar = st.sidebar.markdown(
    '<h1>Sharpe 3</h1><br>', unsafe_allow_html=True
)


search_side_bar = st.sidebar.text_input(
    'eg. CSCO,TSLA,NIO',
    'TSLA,GOOG,CSCO'
)

button_side_bar = st.sidebar.button(
  'Run Analytics',
)

progress_side_bar = st.sidebar.progress(
    0,
)

for percent_complete in range(100):
    time.sleep(0.1)
    progress_side_bar.progress(percent_complete + 1)


text_side_bar = st.sidebar.markdown(
    '<br><br><br>', unsafe_allow_html=True
)
text_side_bar = st.sidebar.markdown(
    '<b>How it works:</b><br>1 - Enter your portfolio. Just add the tickers with comman in the middle (see the example above) <br>2 - Click the button "Run Analytics". It makes 10000 mathematical calulations in less than 10seg.',
    unsafe_allow_html=True
)




portfolio = search_side_bar.split(',')





st.markdown('<br>', unsafe_allow_html=True)  

# Performance
st.markdown('<h2>Performance</h2><br>', unsafe_allow_html=True)

#st.title('Performance')
with st.beta_expander('Description: click here 👉'):
        st.write('PMT')



price = toolbox.StockData(portfolio)
df = toolbox.Performance(price)
st.dataframe(df.style.highlight_max(axis=0))



st.markdown('<br>', unsafe_allow_html=True)  






st.markdown('<br>', unsafe_allow_html=True)  


# Correlations
st.title('Correlations')
with st.beta_expander('Description'):
        st.write('Correlation, in the finance and investment studies, is a statistic that measures the degree to which two assets move in relation to each other. Correlations are used in advanced portfolio management, computed as the correlation coefficient, which has a value that must fall between -1.0 and +1.0. The closer to 0, the less correlated.')
        st.write('Diversification Index is computed with the sum of correlations of every asset in a portfolio divided by the number of assets. The closer to 0, the more diversicated is the portfolio')









st.markdown('<br>', unsafe_allow_html=True)  


options = st.multiselect(
    'Compare your portfolio correlation agaist SP500, Nasdaq, Bitcoin and Gold price',
     ('Portfolio','vs US index', 'vs Bitcoin', 'vs Gold'),
     default='Portfolio'
    )

US,BTC,GOLD = benchmarks()

if 'Portfolio' in options:
    corr = price.pct_change().apply(lambda x: np.log(1+x)).corr()
    div_index = dig(abs(corr.iloc[:,0].sum()-1)/(len(corr.index)))
    st.dataframe(corr.style.highlight_min(axis=0))
    st.write('Diversification Index = ',div_index)

if 'vs US index' in options:
    corrUS = toolbox.My_Corr(price,US)
    div_index_us = dig(abs(corrUS.iloc[:,0].sum()-1)/(len(corrUS.index)))
    st.dataframe(corrUS.style.highlight_min(axis=0))
    st.write('Diversification Index = ',div_index_us)

if 'vs Bitcoin' in options:
    corrBTC = toolbox.My_Corr(price,BTC)
    div_index_btc = dig(abs(corrBTC.iloc[:,0].sum()-1)/(len(corrBTC.index)))
    st.dataframe(corrBTC.style.highlight_min(axis=0))
    st.write('Diversification Index = ',div_index_btc)

if 'vs Gold' in options:
    corrGOLD = toolbox.My_Corr(price,GOLD)
    div_index_btc = dig(abs(corrGOLD.iloc[:,0].sum()-1)/(len(corrGOLD.index)))
    st.dataframe(corrGOLD.style.highlight_min(axis=0))
    st.write('Diversification Index = ',div_index_gold)


st.markdown('<br>', unsafe_allow_html=True)  


#Portfolio Optimization
st.title('Portfolio Optimization')

p_opt,or_p, mv_p = toolbox.Core_Calculations(portfolio,price)

with st.beta_expander('Description'):
        st.write('Modern Portfolio Theory, or also known as mean-variance analysis is a mathematical process which allows the user to maximize returns for a given risk level')
        st.write('This concept is also closely related to "risk-return" trade-off. Sharpe is the key metric of the risk-return of every asset or portfolio combination')

col1, col2, col3 = st.beta_columns(3)

with col1:
    st.header("Risk/Reward table")
    st.dataframe(p_opt.style.highlight_max(axis=0))

with col2:
    st.header("Weights")
    df = toolbox.GetWeights(or_p, mv_p)
    df = df.applymap(dig)
    st.dataframe(df.style.highlight_max(axis=0))

with col3:
    st.header("Risk/Reward chart")
    st.altair_chart(toolbox.Plot_P_Optimization(p_opt), use_container_width=True)
    

st.markdown('<br>', unsafe_allow_html=True)
#st.bokeh_chart(toolbox.Plot_Portfolio(p_opt),use_container_width=True)


st.markdown('<br>', unsafe_allow_html=True)



st.markdown('<br>', unsafe_allow_html=True)   
#st.title('Rankings')
#with st.beta_expander('Description'):
        #st.write('Clustering by industry clasification or Machine Learning')

#tupper = Info()
#bySharpe = tupper.sort_values(by='Sharpe', ascending = False).head(10)
#byYTD = tupper.sort_values(by='YTD', ascending = False).head(10)
#byA = tupper.sort_values(by='A', ascending = False).head(10)
#by6M = tupper.sort_values(by='6M', ascending = False).head(10)
#byM = tupper.sort_values(by='M', ascending = False).head(10)
#byW = tupper.sort_values(by='W', ascending = False).head(10)

#selectInfo = st.multiselect(
 #   'Get some insights ',
 #    ('by Sharpe','by Weekly Performance', 'by Monthly Performance', 'by 6M Performance' 'by Annual Performance','by Current Year to Date'),
 #    default='by Sharpe'
 #   )


#if 'by Sharpe' in selectInfo:
#   st.dataframe(bySharpe.style.highlight_max(axis=0))

#if 'by Annual Performance' in selectInfo:
#    st.dataframe(byA.style.highlight_max(axis=0))

#if 'by Current Year to Date' in selectInfo:
#    st.dataframe(byYTD.style.highlight_max(axis=0))

#if 'by Monthly Performance' in selectInfo:
#    st.dataframe(byM.style.highlight_max(axis=0))

#if 'by 6M Performance' in selectInfo:
#    st.dataframe(by6M.style.highlight_max(axis=0))

#if 'by Weekly Performance' in selectInfo:
#    st.dataframe(byW.style.highlight_max(axis=0))


st.markdown('<br>', unsafe_allow_html=True)   
Disclaimer = 'DISCLAIMER: Sharpe3 provides data regarding public stock market. It does NOT recommend or advice for any investment decission.\nData showed are a mathematical model based in historical public information.'
st.info(Disclaimer)