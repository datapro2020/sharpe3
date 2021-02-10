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



pct = lambda x: '{:.2%}'.format(x)
dig = lambda x: '{:.2f}'.format(x)
to_float = lambda x: float(x.strip('%'))/100

US_index = ['^GSPC','^IXIC']
gold = 'GC=F'
bitcoin = 'BTC-USD'

now = datetime.now()
now = now.strftime('%x - %X')

# Getting the main stock index as for benchmarking 
@st.cache(ttl=3600)
def benchmarks():
    US = toolbox.StockData(US_index)
    US.columns = ['SP500','Nasdaq']
    BTC = toolbox.StockData(bitcoin)
    BTC.columns = ['Bitcoin']
    #GOLD = toolbox.StockData(gold)
    #GOLD.columns = ['Gold SPOT']
    return US,BTC

@st.cache(ttl=3600)
def Tupper():
    return toolbox.GetTupper()

@st.cache(ttl=3600)
def My_Cluster(ann_mean, ann_std):
    return toolbox.Clustering(ann_mean,ann_std)





st.markdown('<h1>Data Driven Investing</h1>', unsafe_allow_html=True)
st.write('Collecting and analyzing data on ',now)


# Sidebar stuff
with st.sidebar:

    st.info("⚠️**BETA:** Ongoing Development")

    st.markdown('<h1>Sharpe 3</h1><br>', unsafe_allow_html=True)

    search_side_bar = st.text_input('eg. TSLA,PTON,AAPL',
    'TSLA,PTON,AAPL'
    )

    st.button('Run Analytics')

    progress_side_bar = st.progress(0)

    for percent_complete in range(100):
        time.sleep(0.1)
        progress_side_bar.progress(percent_complete + 1)


    st.markdown('<br><br><br>', unsafe_allow_html=True)
    
    st.markdown(
    '<b>How it works:</b><br><p>1 - Enter your portfolio. Just add the tickers with comman in the middle (see the example above) <br><br>2 - Click the button "Run Analytics". It makes 10000 mathematical calulations in less than 10seg.<p><br><br><br><br><br><br><br><br>',
    unsafe_allow_html=True
    )

    st.markdown('<p><a href="#about">About Sharpe 3</a></p>', unsafe_allow_html=True)
    st.markdown('<p><a href="#privacy">Privacy Policy</a></p>', unsafe_allow_html=True)
    st.markdown('<p><a href="#terms">Terms of Use</a></p>', unsafe_allow_html=True)




portfolio = search_side_bar.split(',')





st.markdown('<br>', unsafe_allow_html=True)  

# Performance
st.markdown('<h1>Performance</h1><br>', unsafe_allow_html=True)


with st.beta_expander('Description: click here 👉'):
        st.write('PMT')

st.markdown('<br>', unsafe_allow_html=True) 

price = toolbox.StockData(portfolio)
df = toolbox.Performance(price)
df = df.fillna(0)

col1, col2 = st.beta_columns(2)

with col1:
    st.markdown('<h3>Returns</h3>', unsafe_allow_html=True)
   
    st.dataframe(df.style.highlight_min(axis=0))

  
with col2:
    pic2 = toolbox.Plot_Performance2(df)
    st.altair_chart(pic2, use_container_width=True)
    

st.markdown('<h3>Evolution</h3>', unsafe_allow_html=True)
pic1 = toolbox.Plot_Performance1(price)
st.altair_chart(pic1, use_container_width=True)

st.markdown('<br><br>', unsafe_allow_html=True)  


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

US,BTC = benchmarks()

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

#if 'vs Gold' in options:
 #   corrGOLD = toolbox.My_Corr(price,GOLD)
  #  div_index_btc = dig(abs(corrGOLD.iloc[:,0].sum()-1)/(len(corrGOLD.index)))
   # st.dataframe(corrGOLD.style.highlight_min(axis=0))
    #st.write('Diversification Index = ',div_index_gold)


st.markdown('<br><br>', unsafe_allow_html=True)  


#Portfolio Optimization
st.title('Portfolio Optimization')

p_opt,or_p, mv_p, p_ret, p_vol = toolbox.Core_Calculations(portfolio,price)

with st.beta_expander('Description'):
        st.write('Modern Portfolio Theory, or also known as mean-variance analysis is a mathematical process which allows the user to maximize returns for a given risk level')
        st.write('This concept is also closely related to "risk-return" trade-off. Sharpe is the key metric of the risk-return of every asset or portfolio combination')

col1, col2, col3 = st.beta_columns(3)

with col1:
    st.markdown('<h3>Risk/Reward table</h3>', unsafe_allow_html=True)
    st.dataframe(p_opt.style.highlight_max(axis=0))

with col2:
    st.markdown('<h3>Weights</h3>', unsafe_allow_html=True)
    df = toolbox.GetWeights(or_p, mv_p)
    df = df.applymap(dig)
    st.dataframe(df.style.highlight_max(axis=0))

with col3:
    st.markdown('<h3>Risk/Reward chart</h3>', unsafe_allow_html=True)
    st.altair_chart(toolbox.Plot_P_Optimization(p_opt), use_container_width=True)
    





st.markdown('<br><br>', unsafe_allow_html=True)   


tupper_df = Tupper()
#tupper.iloc[:,0]=tupper.iloc[:,0].apply(to_float)
#tupper.iloc[:,1]=tupper.iloc[:,1].apply(to_float)


#Clustering
st.title('Clustering')
with st.beta_expander('Description'):
        st.write('In Machine Learning, data "unlabeled" can be automaticaly organized, known as “unsupervised learning”. The K-means clustering algorithm is a part of unsupervised learning, which a given unlabeled dataset will automatically grouped into coherent clusters ')

#ann_mean = tupper_df.loc[:,'Return']
#ann_mean = p_ret.append(ann_mean)
#ann_std =  tupper_df.loc[:,'Volatility']
#ann_std =  ann_std.append(ann_mean)

ann_mean = p_ret.combine_first(tupper_df.loc[:,'Return'])
ann_std = p_vol.combine_first(tupper_df.loc[:,'Volatility'])



#df1 = tupper.iloc[:,[0,1]]
#df2 = pd.concat([p_ret, p_vol],axis = 1)
#df = toolbox.Join_Df(df1,df2)
#k_means = toolbox.Clustering(df.iloc[:,0],df.iloc[:,1])
#k_means = toolbox.Clustering(ann_mean,ann_std)
k_means = My_Cluster(ann_mean,ann_std)
#k_means = pd.merge(k_means, tupper_df[['Name','Country','Sector','Industry','IPO Year','Market Cap']], left_index=True, right_index=False)
k_means = k_means.combine_first(tupper_df[['Name','Country','Sector','Industry','IPO Year','Market Cap']])
k_means.index.name = 'ticker'

k_means_pic = k_means.reset_index()




scales = alt.selection_interval(bind='scales')
pic = alt.Chart(k_means_pic).mark_point().encode(x='Volatility',y='Return',color='Cluster:N', tooltip=['ticker:N', 'Volatility:N','Return:N','Name:N','Country:N','Sector:N','Industry:N','IPO Year:N','Market Cap:N']).add_selection(
    scales
)


col1, col2 = st.beta_columns(2)

with col1:
    st.markdown('<h3>Clustering based in K-Means</h3>', unsafe_allow_html=True)
    #st.altair_chart(pic, use_container_width=True)
    st.write('Dataset of ',len(tupper_df.index.array),'public listed companies in the US over $2B market cap')

with col2:
    st.markdown('<h3>Portfolio clusters</h3>', unsafe_allow_html=True)
#    st.dataframe(p_cluster.style.highlight_max(axis=0))
    
    for i in portfolio:
        st.write(i+' is within the cluster ', k_means.loc[i,'Cluster'].astype(int))

#st.write(k_means[k_means['Clustering']==0].head(3))

st.altair_chart(pic, use_container_width=True)

tupper = pd.merge(k_means['Cluster'], tupper_df, left_index=True, right_index=True)
tupper = pd.DataFrame(tupper, columns=['Cluster','Name','Country','Sector','Industry','Market Cap','IPO Year','Return','Volatility','Sharpe','Min_Corr','Corr_value','D','W','M'])

cluster_view = st.multiselect(
    'Discover other stocks in each cluster and filter',
    ('by Maximum Return','by Maximum Sharpe', 'by Weekly Return', 'by Montly Return'),
    default='by Maximum Return'
    )


if 'by Maximum Return' in cluster_view:
    st.write(tupper[tupper['Cluster']==0].sort_values(by='Return', ascending = False).head(3))
    st.write(tupper[tupper['Cluster']==1].sort_values(by='Return', ascending = False).head(3))
    st.write(tupper[tupper['Cluster']==2].sort_values(by='Return', ascending = False).head(3))
    st.write(tupper[tupper['Cluster']==3].sort_values(by='Return', ascending = False).head(3))
    st.write(tupper[tupper['Cluster']==4].sort_values(by='Return', ascending = False).head(3))

if 'by Maximum Sharpe' in cluster_view:
    st.write(tupper[tupper['Cluster']==0].sort_values(by='Sharpe', ascending = False).head(3))
    st.write(tupper[tupper['Cluster']==1].sort_values(by='Sharpe', ascending = False).head(3))
    st.write(tupper[tupper['Cluster']==2].sort_values(by='Sharpe', ascending = False).head(3))
    st.write(tupper[tupper['Cluster']==3].sort_values(by='Sharpe', ascending = False).head(3))
    st.write(tupper[tupper['Cluster']==4].sort_values(by='Sharpe', ascending = False).head(3))

if 'by Weekly Return' in cluster_view:
    st.write(tupper[tupper['Cluster']==0].sort_values(by='W', ascending = False).head(3))
    st.write(tupper[tupper['Cluster']==1].sort_values(by='W', ascending = False).head(3))
    st.write(tupper[tupper['Cluster']==2].sort_values(by='W', ascending = False).head(3))
    st.write(tupper[tupper['Cluster']==3].sort_values(by='W', ascending = False).head(3))
    st.write(tupper[tupper['Cluster']==4].sort_values(by='W', ascending = False).head(3))

if 'by Montly Return' in cluster_view:
    st.write(tupper[tupper['Cluster']==0].sort_values(by='M', ascending = False).head(3))
    st.write(tupper[tupper['Cluster']==1].sort_values(by='M', ascending = False).head(3))
    st.write(tupper[tupper['Cluster']==2].sort_values(by='M', ascending = False).head(3))
    st.write(tupper[tupper['Cluster']==3].sort_values(by='M', ascending = False).head(3))
    st.write(tupper[tupper['Cluster']==4].sort_values(by='M', ascending = False).head(3))




st.markdown('<br><br>', unsafe_allow_html=True)  

#Forecasting
st.title('AI Forecasting')
with st.beta_expander('Description'):
        st.write('Forecasting based in historical data is no indication of whether a price will go up or down. However, modern AI technology can provide a forecasting framework for non-linear trends like the stock market. Sharpe 3 used the Prophet open source engine developed by Facebook to automate and adapt stock forecasting. Those charts represent trends for the next 12 months, with the best case (yhat_upper) and the worse case(yhat_lower) forecast line')

st.markdown('<br>', unsafe_allow_html=True)  

for i in portfolio:
    st.write('Forecast for ', i)
    pic = toolbox.Forecast(price[i],i)
    st.altair_chart(pic, use_container_width=True)



st.markdown('<br><br><br>', unsafe_allow_html=True)   
Disclaimer = '**DISCLAIMER:** Sharpe3 provides data regarding public stock market. It does NOT recommend or advice for any investment decission.\nData showed are based in a mathematical model calculated with historical public information.'
st.info(Disclaimer)

st.markdown('<hr/><br><br><br><br>', unsafe_allow_html=True)

st.markdown('<h2 id="about">_</h2>', unsafe_allow_html=True)
with st.beta_expander('About Us',expanded=True):     
    st.markdown('<p> Sharpe 3 provides clarity to simplify data driven investment deccisions. <br>Based in a Machine Leraning and data analytics technology, the Sharpe 3 platform can proccess thousands of calculations and data points to present a new perpective of stock assets. </p>', unsafe_allow_html=True)
    st.markdown('<p> Every stock deccision is in a permanend cross-road. In one direcction, there is a lot of "hype" and misinformation for trendy assets. In other direcctions, an universe of data based mathematical and statistical models, known as a Quantitative analysis (QA), complex to understand.</p>', unsafe_allow_html=True)
    st.markdown('<p>Sharpe 3 help individual investors to discover valuable data to complement every investment action.', unsafe_allow_html=True)

st.markdown('<h2 id="privacy">_</h2>', unsafe_allow_html=True)
with st.beta_expander('Privacy',expanded=False):     
    st.markdown('<p> Sharpe 3 provides clarity to simplify data driven investment deccisions. <br>Based in a Machine Leraning and data analytics technology, the Sharpe 3 platform can proccess thousands of calculations and data points to present a new perpective of stock assets. </p>', unsafe_allow_html=True)
    st.markdown('<p> Every stock deccision is in a permanend cross-road. In one direcction, there is a lot of "hype" and misinformation for trendy assets. In other direcctions, an universe of data based mathematical and statistical models, known as a Quantitative analysis (QA), complex to understand.</p>', unsafe_allow_html=True)
    st.markdown('<p>Sharpe 3 help individual investors to discover valuable data to complement every investment action', unsafe_allow_html=True)

st.markdown('<h2 id="terms">_</h2>', unsafe_allow_html=True)
with st.beta_expander('Terms of Use',expanded=False):     
    st.markdown('<p> Sharpe 3 provides clarity to simplify data driven investment deccisions. <br>Based in a Machine Leraning and data analytics technology, the Sharpe 3 platform can proccess thousands of calculations and data points to present a new perpective of stock assets. </p>', unsafe_allow_html=True)
    st.markdown('<p> Every stock deccision is in a permanend cross-road. In one direcction, there is a lot of "hype" and misinformation for trendy assets. In other direcctions, an universe of data based mathematical and statistical models, known as a Quantitative analysis (QA), complex to understand.</p>', unsafe_allow_html=True)
    st.markdown('<p>Sharpe 3 help individual investors to discover valuable data to complement every investment action', unsafe_allow_html=True)


