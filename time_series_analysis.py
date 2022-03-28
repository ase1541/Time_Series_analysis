#LIBRARIES
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
from pandas_datareader import data as pdr
import yfinance as yf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from pmdarima.arima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.graphics import tsaplots
yf.pdr_override()
pd.set_option('display.float_format',  '{:,.2f}'.format)

### Hourly and Daily Time series-extraction  from yfinance
first_date = '2020-04-01' #It only allows 2years of hourly data
now=dt.datetime.today()
today = now.strftime('%Y-%m-%d')
companies=['ROVI.MC', '^IBEX']
#We get time series (Daily) for Ibex 35(it is an INDEX), and for the stock ROVI within this index
I35_dly=pdr.get_data_yahoo(companies[1], start=first_date, end=today,interval="1d")
ROVI_dly=pdr.get_data_yahoo(companies[0], start=first_date, end=today,interval="1d")
#We get time series (Hourly) for Ibex 35(it is an INDEX), and for the stock ROVI within this index
I35_hly=pdr.get_data_yahoo(companies[1], start=first_date, end=today,interval="1h")
ROVI_hly=pdr.get_data_yahoo(companies[0], start=first_date, end=today,interval="1h")
time_series={"ROVI_daily":ROVI_dly,"ROVI_hourly":ROVI_hly,"IBEX_daily":I35_dly,"IBEX_hourly":I35_hly}

### If we wanted we can plot the different time series
"""for key in time_series:
    Column="High"
    plt.figure(figsize=(10,6))
    plt.grid(True)
    plt.xlabel('Date')
    plt.ylabel(f'{Column} Prices')
    plt.plot(time_series[key][Column])
    plt.title(key,fontsize=18, fontweight='bold')
    plt.show()
"""
from functions import dickey_fuller

### Statistical anlysis of every time series
"""for key in time_series:
    column="Volume" #'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'
    fig, (ax1, ax2 ,ax3) = plt.subplots(1,3,figsize =(40,10))
    ax1.set_title('Kernel Density Estimate',fontsize=13)
    fig.suptitle(key, fontsize=24)
    time_series[key]["Open"].plot(kind='kde',ax=ax1)
    tsaplots.plot_acf (time_series[key][column], lags = 250, ax=ax2)
    tsaplots.plot_pacf(time_series[key][column], lags = 250, ax=ax3)
    dickey_fuller(time_series[key][column])
    plt.show ()
"""
### Decomposition of the Close price in Trend, Seasonal and Residual
decompose = seasonal_decompose(time_series["ROVI_daily"]["Close"], model='multiplicative', period=30)
fig = plt.figure()
fig.set_size_inches(16, 9)
decompose.plot()
plt.show()