##Libraries
from dataset_statistical_analysis import trial, stock, serie
from functions import forecast_accuracy
import matplotlib.pyplot as plt
import pandas as pd
from pmdarima.arima import auto_arima

#split data into train (98%) and training set(2%)
train_data, test_data = trial[3:int(len(trial)*0.98)], trial[int(len(trial)*0.98):]
#Uncomment I we want to visualize the dataset
"""plt.figure(figsize=(10,6))
plt.grid(True)
plt.xlabel('Dates')
plt.ylabel('Closing Prices')
plt.plot(time_series["ROVI_daily"]["Close"], 'green', label='Train data')
plt.plot(test_data, 'blue', label='Test data')
plt.legend()
plt.show()"""
#find the proper model for ARIMA with d=1 and fit it
model_autoARIMA = auto_arima(train_data, start_p=0, start_q=0,
                      test='adf',       # use adftest to find optimal 'd'
                      max_p=5, max_q=5, # maximum p and q
                      m=1,              # frequency of series
                      d=1,           # let model determine 'd'
                      seasonal=False,   # No Seasonality
                      start_P=0,
                      D=0,
                      trace=True,
                      error_action='ignore',
                      suppress_warnings=True,
                      stepwise=True)
#print(model_autoARIMA.summary())
#model_autoARIMA.plot_diagnostics(figsize=(15,8))
#plt.show()

###Create predictions and plot them to see
prediction, confint = model_autoARIMA.predict(n_periods=len(test_data), return_conf_int=True)

prediction_series = pd.Series(prediction, index=test_data.index) #Prediction series
cf= pd.DataFrame(confint) #Confidence intervals
plt.figure(figsize=(10,5), dpi=100)
plt.plot(train_data, label='training data')
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

print(f"""mape:{acc["mape"]}
me:{acc["me"]}
mae:{acc["mae"]}
mpe:{acc["mpe"]} 
rmse:{acc["rmse"]}
corr:{acc["corr"]}
minmax:{acc["minmax"]}""")
