# Time_Series_analysis
Time series analysis for Rovi and Ibex35. In this code, I perform an ARIMA and ARIMAX modelling and benchmark a single stock model to its index performance. 
I perform several statistical tests to see the approximate parameters for the ARIMA model, and later, build ARIMA and ARIMAX bearing these parameters in mind 
when initiallizing the auto_arima method. 

There are three scripts:

•	ARIMA_model.py: In this script, I build the model, and plot the relevant figures of the ARIMA and ARIMAX figures. To switch from ARIMAX to ARIMA modelling, 
it is as easy as removing the exogenous variables from the method auto_arima, and the latter predict and fit methods as well.
•	dataset_statistical_analysis: Here I perform all of the statistical tests mentioned before as well as the dataset construction.
•	functions.py: In order to keep the code clean, I write the functions used along the other 2 scripts.


The results of the hourly model are very acceptable. It would perform better if the dataset was bigger.
