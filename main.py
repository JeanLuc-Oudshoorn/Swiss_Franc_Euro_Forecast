# Imports
import yfinance as yf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import pandas as pd

# Loading EUR/CHF exchange rate from Yahoo Finance
ticker = yf.Ticker('EURCHF=X')
eurchf_df = ticker.history(period='max')

# Reading in CPI data and turning into a dataframe
euro_CPI = pd.read_csv('Euro_CPI.csv')
swiss_CPI = pd.read_csv('Swiss_CPI.csv')

euro_CPI = pd.DataFrame(euro_CPI)
swiss_CPI = pd.DataFrame(swiss_CPI)

# Using date as an index and calculating inflation from CPI
euro_CPI['DATE'] = pd.to_datetime(euro_CPI['DATE'])
swiss_CPI['DATE'] = pd.to_datetime(swiss_CPI['DATE'])

euro_CPI['shift'] = euro_CPI['CP0000EZ19M086NEST'].shift(12)
swiss_CPI['shift'] = swiss_CPI['CHECPIALLMINMEI'].shift(12)

euro_CPI['inflation'] = euro_CPI['CP0000EZ19M086NEST'] / euro_CPI['shift']
swiss_CPI['inflation'] = swiss_CPI['CHECPIALLMINMEI'] / swiss_CPI['shift']

euro_CPI.set_index('DATE', inplace=True)
swiss_CPI.set_index('DATE', inplace=True)

# Creating a single dataframe and filling missing values
df_all = pd.DataFrame(eurchf_df['Close'])
df_all['swiss_inf'] = swiss_CPI['inflation']
df_all['euro_inf'] = euro_CPI['inflation']
df_all.ffill(axis=0, inplace=True)
df_all.bfill(axis=0, inplace=True)

# Creating exogenous variable: inflation ratio in Eurozone divided by Switzerland
df_all['ratio'] = df_all['euro_inf'] / df_all['swiss_inf']

# Checking for stationarity after one order of differencing
df_all = df_all['2015-02-01':]
plt.plot(df_all['Close'].diff().dropna())
plt.show()
adf = adfuller(df_all['Close'].diff().dropna())

# Creating an empty list for model results
order_aic_bic = []

# Fitting models in a loop to find the best solution
for p in range(3):
    for q in range(3):
        try:
            model = SARIMAX(df_all['Close'], exog = df_all['ratio'], order = (p, 1, q))
            results = model.fit()

            order_aic_bic.append((p, q, results.aic, results.bic))
        except:
            order_aic_bic.append((p, q, None, None))

order_df = pd.DataFrame(order_aic_bic,
                        columns=['p', 'q', 'AIC', 'BIC'])

print(order_df.sort_values('AIC'))
print(order_df.sort_values('BIC'))

# Fitting best model and checking summary statistics
model = SARIMAX(df_all['Close'], exog = df_all['ratio'], order = (2, 1, 0))
results = model.fit()

print(results.summary())
plt.clf()
results.plot_diagnostics()
plt.show()

# Generate forecast
dynamic_forecast = results.get_prediction(start=-60, dynamic=True)

mean_forecast = dynamic_forecast.predicted_mean

confidence_intervals = dynamic_forecast.conf_int()

lower_limits = confidence_intervals.loc[:,'lower Close']
upper_limits = confidence_intervals.loc[:,'upper Close']

# Plot forecast
plt.clf()
plt.plot(df_all.index, df_all['Close'], label='observed')
plt.plot(mean_forecast.index, mean_forecast, color='r', label='forecast')

plt.fill_between(lower_limits.index, lower_limits,
         upper_limits, color='pink')

plt.xlabel('Date')
plt.ylabel('EUR/CHF Exchange Rate (Daily)')
plt.legend()
plt.show()
