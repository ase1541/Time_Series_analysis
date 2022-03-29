#LIBRARIES
import pandas as pd
import numpy as np
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
        print ('Failed to Reject Ho - Time Series is Non-Stationary')


#Accuracy function information
# Accuracy metrics
def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
    me = np.mean(forecast - actual)             # ME
    mae = np.mean(np.abs(forecast - actual))    # MAE
    mpe = np.mean((forecast - actual)/actual)   # MPE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    corr = np.corrcoef(forecast, actual)[0,1]   # corr
    mins = np.amin(np.hstack([forecast[:,None],
                              actual[:,None]]), axis=1)
    maxs = np.amax(np.hstack([forecast[:,None],
                              actual[:,None]]), axis=1)
    minmax = 1 - np.mean(mins/maxs)             # minmax

    return({'mape':mape, 'me':me, 'mae': mae,
            'mpe': mpe, 'rmse':rmse,
            'corr':corr, 'minmax':minmax})
