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
 #It only allows 2years of hourly data
today=dt.datetime.today()
first_date = today - dt.timedelta(days=729)
companies=['ROVI.MC', '^IBEX']
#We get time series (Daily) for Ibex 35(it is an INDEX), and for the stock ROVI within this index
I35_dly=pdr.get_data_yahoo(companies[1], start=first_date, end=today,interval="1d")
ROVI_dly=pdr.get_data_yahoo(companies[0], start=first_date, end=today,interval="1d")
#We get time series (Hourly) for Ibex 35(it is an INDEX), and for the stock ROVI within this index
I35_hly=pdr.get_data_yahoo(companies[1], start=first_date, end=today,interval="1h")
ROVI_hly=pdr.get_data_yahoo(companies[0], start=first_date, end=today,interval="1h")
time_series={"ROVI_daily":ROVI_dly,"IBEX_daily":I35_dly ,"IBEX_hourly":I35_hly,"ROVI_hourly":ROVI_hly}#,"IBEX_hourly":I35_hly,"ROVI_hourly":ROVI_hly}

### If we wanted we can plot the different time series
"""for key in time_series:
    Column="Close"
    fig, axes = plt.subplots(1, 2, figsize=(25,7))
    axes[0].plot(time_series[key]["Close"])
    axes[0].set_title(f'{key} Closing prices')
    plt.xlabel('Date')

    axes[1].plot(time_series[key]["Volume"])
    axes[1].set_title(f'{key} Volume traded')


    plt.grid(True)
    plt.show()
"""
from functions import dickey_fuller

### Statistical anlysis of every time series
"""for key in time_series:
    column="Close" #'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'
    fig, (ax1, ax2 ,ax3) = plt.subplots(1,3,figsize =(40,10))
    ax1.set_title('Kernel Density Estimate',fontsize=13)
    fig.suptitle(f"{key} {column} ", fontsize=24)
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
#First we start with the param d for differentiation, for the p and d we find them through autoarima
#This plots the differentiation test for every data set and every time series within the dataset.
"""
for stock in time_series.keys():
    serie="Close"
    trial=time_series[stock][serie]
    plt.rcParams.update({'figure.figsize':(9,7), 'figure.dpi':120})
    # Original Series
    fig, axes = plt.subplots(3, 2)
    axes[0, 0].plot(trial)
    axes[0, 0].set_title(f'Original {stock} {serie} Series')
    plot_acf(trial, ax=axes[0, 1])

    # 1st Differencing
    axes[1, 0].plot(trial.diff())
    axes[1, 0].set_title(f'1st Order Differencing for {stock} {serie}')
    plot_acf(trial.diff().dropna(), ax=axes[1, 1])

    # 2nd Differencing
    axes[2, 0].plot(trial.diff().diff())
    axes[2, 0].set_title(f'2nd Order Differencing for {stock} {serie}')
    plot_acf(trial.diff().dropna(), ax=axes[2, 1])
    fig.tight_layout()
    plt.show()

    print(f"Diff value with adf test for {stock} {serie}: ",ndiffs(trial, test='adf'))
    print(f"Diff value with kpss test for {stock} {serie}: ",ndiffs(trial, test='kpss'))
    print(f"Diff value with pp test for {stock} {serie}: ",ndiffs(trial, test='pp'))
    print('\n')
    """




