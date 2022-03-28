#LIBRARIES
import pandas as pd
from statsmodels.tsa.stattools import adfuller

#Dickey Fuller Test
def dickey_fuller(timeseries):
    print("Results of dickey fuller test")
    adft = adfuller(timeseries,autolag='AIC')
    print('ADF Statistic: %f' % adft[0])
    print('p-value: %f' % adft[1])
    print('Critical Values:')
    output = pd.Series(adft[0:4],index=['ADF Test Statistics','p-value','No. of lags used','Number of observations used'])
    for key, value in adft[4].items():
        print('\t%s: %.3f' % (key, value))
    if adft[0] < adft[4]["5%"]:
        print ("Reject Ho - Time Series is Stationary")
    else:
        print ("Failed to Reject Ho - Time Series is Non-Stationary")