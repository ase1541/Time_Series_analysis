#LIBRARIES
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
from pandas_datareader import data as pdr
from pmdarima.arima.utils import ndiffs
import yfinance as yf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
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
    plot_acf (time_series[key][column], lags = 250, ax=ax2)
    plot_pacf(time_series[key][column], lags = 250, ax=ax3)
    dickey_fuller(time_series[key][column])
    plt.show ()
"""
### Decomposition of the Close price in Trend, Seasonal and Residual
"""
decompose = seasonal_decompose(time_series["ROVI_daily"]["Close"], model='multiplicative', period=30)
fig = plt.figure()
fig.set_size_inches(16, 9)
decompose.plot()
plt.show()
"""

### Now we try to find p,d,q params for the ARIMA model
#First we start with the param d for differentiation
stock="IBEX_hourly"
serie="Close"
trial=time_series[stock][serie]
plt.rcParams.update({'figure.figsize':(9,7), 'figure.dpi':120})
# Original Series
fig, axes = plt.subplots(3, 2)
axes[0, 0].plot(trial)
axes[0, 0].set_title('Original Series')
plot_acf(trial, ax=axes[0, 1])

# 1st Differencing
axes[1, 0].plot(trial.diff())
axes[1, 0].set_title('1st Order Differencing')
plot_acf(trial.diff().dropna(), ax=axes[1, 1])

# 2nd Differencing
axes[2, 0].plot(trial.diff().diff())
axes[2, 0].set_title('2nd Order Differencing')
plot_acf(trial.diff().dropna(), ax=axes[2, 1])
fig.tight_layout()
plt.show()

print("Diff value with adf test: ",ndiffs(trial, test='adf'))
print("Diff value with kpss test: ",ndiffs(trial, test='kpss'))
print("Diff value with pp test: ",ndiffs(trial, test='pp'))




