import numpy as np
import pandas as pd
import pandas_datareader as web
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.ticker as mtick

#%%
MELI = web.DataReader('MELI','yahoo',start="2015-01-01", end="2020-07-06")
NASDAQ = web.DataReader('^IXIC','yahoo',start="2015-01-01", end="2020-07-06")
SP500 = web.DataReader('SPY','yahoo',start="2015-01-01", end="2020-07-06")

#%%
MELI['Adj Close'].plot(title = "MeLi Closing Price")


#%%

decomp = seasonal_decompose(MELI['Open'], period=12)

fig = decomp.plot() 
fig.set_size_inches(15, 8)
fig.suptitle("MeLi Time Series Decomposition")
#%% Historic

MELI = MELI.resample(rule = "W").mean()
NASDAQ =  NASDAQ.resample(rule = "W").mean()
SP500 =  SP500.resample(rule = "W").mean()

MELI['simple_return'] = (MELI['Close'] / MELI['Close'].shift(1)) - 1
NASDAQ['simple_return'] = (NASDAQ['Close'] / NASDAQ['Close'].shift(1)) - 1
SP500['simple_return'] = (SP500['Close'] / SP500['Close'].shift(1)) - 1

MELI['simple_return'].plot()
NASDAQ['simple_return'].plot()
SP500['simple_return'].plot()


for df_stock in (MELI,SP500,NASDAQ):
    df_stock['Normalize Return'] = df_stock['Adj Close']/df_stock.iloc[0]['Adj Close']

portfolio_val = pd.concat([MELI['Normalize Return'],
                           NASDAQ['Normalize Return'],
                           SP500['Normalize Return']], axis = 1)
                           
portfolio_val.columns = ['MELI','Nasdaq','SP500']
portfolio_val = portfolio_val.reset_index()

#%%

x = portfolio_val.Date

fig2, ax2 = plt.subplots()  # Create a figure and an axes.
ax2.plot(x, portfolio_val.MELI , label='MELI')  # Plot some data on the axes.
ax2.plot(x, portfolio_val.Nasdaq, label='Nasdaq')  # Plot more data on the axes...
ax2.plot(x, portfolio_val.SP500, label='S&P500')  # ... and some more.
ax2.set_xlabel('Date')  # Add an x-label to the axes.
ax2.set_ylabel('Growth')  # Add a y-label to the axes.
ax2.set_title("Returns on Meli")  # Add a title to the axes.
ax2.legend()  # Add a legend.

#%% 2019 Onward

MELI_19 =  MELI.reset_index()
MELI_19 = MELI_19[MELI_19["Date"]>= "2019-01-01"].set_index("Date")

NASDAQ_19 =  NASDAQ.reset_index()
NASDAQ_19 = NASDAQ_19[NASDAQ_19["Date"]>= "2019-01-01"].set_index("Date")

SP500_19 =  SP500.reset_index()
SP500_19 = SP500_19[SP500_19["Date"]>= "2019-01-01"].set_index("Date")



MELI_19['simple_return'] = (MELI_19['Close'] / MELI_19['Close'].shift(1)) - 1
NASDAQ_19['simple_return'] = (NASDAQ_19['Close'] / NASDAQ_19['Close'].shift(1)) - 1
SP500_19['simple_return'] = (SP500_19['Close'] / SP500_19['Close'].shift(1)) - 1

for df_stock in (MELI_19,NASDAQ_19,SP500_19):
    df_stock['Normalize Return'] = (df_stock['Adj Close']/df_stock.iloc[0]['Adj Close']-1)*100

portfolio_19 = pd.concat([MELI_19['Normalize Return'],
                           NASDAQ_19['Normalize Return'],
                           SP500_19['Normalize Return']], axis = 1)
                           
portfolio_19.columns = ['MELI','NASDAQ','SP500']
portfolio_19 = portfolio_19.reset_index()


x = portfolio_19.Date

# Note that even in the OO-style, we use `.pyplot.figure` to create the figure.
fig3, ax3 = plt.subplots()  # Create a figure and an axes.
ax3.plot(x, portfolio_19.MELI , label='MELI')  # Plot some data on the axes.
ax3.plot(x, portfolio_19.NASDAQ, label='Nasdaq')  # Plot more data on the axes...
ax3.plot(x, portfolio_19.SP500, label='S&P500')  # ... and some more.
ax3.set_xlabel('Date', )  # Add an x-label to the axes.
ax3.set_ylabel('Growth')  # Add a y-label to the axes.
ax3.set_title("Returns")  # Add a title to the axes.
ax3.legend()  # Add a legend.
plt.xticks(rotation=45)
ax3.yaxis.set_major_formatter(mtick.PercentFormatter())

# %%

portfolio_hist = pd.concat([MELI_19['Adj Close'],
                           NASDAQ_19['Adj Close'],
                           SP500_19['Adj Close']], axis = 1)

portfolio_returns = np.log(portfolio_hist / portfolio_hist.shift(1))
portfolio_returns.columns = ['MELI','NASDAQ','SP500']

cov_matrix = portfolio_returns.cov()*250

corr_matrix = portfolio_returns.corr()



