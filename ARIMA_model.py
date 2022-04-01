##Libraries
from dataset_statistical_analysis import time_series
from functions import forecast_accuracy
import matplotlib.pyplot as plt
import pandas as pd
from pmdarima.arima import auto_arima


#Uncomment I we want to visualize the dataset
"""plt.figure(figsize=(10,6))
plt.grid(True)
plt.xlabel('Dates')
plt.ylabel('Closing Prices')
plt.plot(time_series["ROVI_daily"]["Close"], 'green', label='Train data')
plt.plot(test_data, 'blue', label='Test data')
plt.legend()
plt.show()"""

###We proceed to create the exogenous variables for the ARIMAX model.
stock="ROVI_hourly"
serie="Close"
trial=time_series[stock]
lag_features = ["Open", "High", "Low", "Adj Close"]
window1 = 3
window2 = 7
window3 = 30

#trial_rolled_3d = trial[lag_features].rolling(window=window1, min_periods=0)
#trial_rolled_7d = trial[lag_features].rolling(window=window2, min_periods=0)
trial_rolled_30d = trial[lag_features].rolling(window=window3, min_periods=0)

#trial_mean_3d = trial_rolled_3d.mean()
#trial_mean_7d = trial_rolled_7d.mean()
trial_mean_30d = trial_rolled_30d.mean()

#trial_std_3d = trial_rolled_3d.std()
#trial_std_7d = trial_rolled_7d.std()
trial_std_30d = trial_rolled_30d.std()

exogenous= pd.DataFrame(index=trial.index)

for feature in lag_features:
    #exogenous[f"{feature}_mean_lag{window1}"] = trial_mean_3d[feature]
    #exogenous[f"{feature}_mean_lag{window2}"] = trial_mean_7d[feature]
    exogenous[f"{feature}_mean_lag{window3}"] = trial_mean_30d[feature]

    #exogenous[f"{feature}_std_lag{window1}"] = trial_std_3d[feature]
    #exogenous[f"{feature}_std_lag{window2}"] = trial_std_7d[feature]
    exogenous[f"{feature}_std_lag{window3}"] = trial_std_30d[feature]
exogenous.fillna(exogenous.mean(), inplace=True)

train_data = trial[serie][:round(len(trial["Close"])*0.9)]
test_data = trial[serie][round(len(trial["Close"])*0.9):]
#exogenous=trial.drop(["Close"], axis=1)


train_data = time_series[stock]["Close"][:round(len(trial["Close"]) * 0.9)]
test_data = time_series[stock]["Close"][round(len(trial["Close"]) * 0.9):]
model_autoARIMA = auto_arima(train_data, start_p=0, start_q=0,exogenous=exogenous[:round(len(trial["Close"])*0.9)],
                             test='adf',       # use adftest to find optimal 'd'
                             max_p=2, max_q=2, # maximum p and q
                             m=1,              # frequency of series
                             d=1,           # let model determine 'd'
                             seasonal=False,   # No Seasonality
                             start_P=1,
                             D=1,
                             trace=True,
                             error_action='ignore',
                             suppress_warnings=True,
                             stepwise=True)
model_autoARIMA.fit(train_data, exogenous[:round(len(trial["Close"])*0.9)])
#print(model_autoARIMA.summary())
#model_autoARIMA.plot_diagnostics(figsize=(15,8))
#plt.show()

###Create predictions and plot them to see
prediction, confint = model_autoARIMA.predict(n_periods=len(test_data), return_conf_int=True, exogenous=exogenous[round(len(trial["Close"])*0.9):])

prediction_series = pd.Series(prediction, index=test_data.index) #Prediction series
cf= pd.DataFrame(confint) #Confidence intervals
plt.figure(figsize=(10,5), dpi=100)
#plt.plot(train_data, label='training data')
plt.plot(test_data, color = 'blue', label='Actual Stock Price')
plt.plot(prediction_series, color = 'orange',label='Predicted Stock Price')
plt.fill_between(prediction_series.index,
                 cf[0],
                 cf[1],color='grey',alpha=.3)

plt.title(f'{stock} {serie} Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend(loc='upper left', fontsize=8)
plt.show()

### We proceed to analyze the result of our predictions using the folllowing metrics
acc=forecast_accuracy(prediction, test_data)


print(f"""Accuracy metrics for {stock}
mape: {acc["mape"]}
me: {acc["me"]}
mae: {acc["mae"]}
mpe: {acc["mpe"]}
rmse: {acc["rmse"]}
corr: {acc["corr"]}
minmax: {acc["minmax"]}""")
