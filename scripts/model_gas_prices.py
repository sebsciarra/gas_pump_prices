


"""
Goal
1. Fit Toronto gas prices using 

1) Basic ETSL (base model for reference)
2) Innovates state space models (slightly more advanced reference)
3) 

Questions
1. Does a random-forest model of TS models (i.e., weak learners) produce less biased model with lower variance? 
2. How to handle patterns of multiple seasonality (weekly, seasons, holidays)? 
3. How to handle feedback effects of current gas prices on future ones? That is, when gas prices become too high
the rate of production slows down. 
4. 
"""

# region 0) Packages 
import numpy as np 
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')
from statsforecast import StatsForecast
from statsforecast.models import AutoETS
from statsmodels.tsa.seasonal import seasonal_decompose
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import rmse, mae, mape, mase

from plotnine import *

import calendar

if any(fname.startswith("model_gas_prices.py") for fname in os.listdir()):
    os.chdir("..")

# endregion 

# region 1) Load gas price data and apply pre-processing 

df_gas = pd.read_csv('data/raw/fueltypesall.csv')

# convert date to datetime object 
df_gas['Date'] = pd.to_datetime(df_gas['Date'])
df_gas.drop('Type de carburant', axis=1, inplace=True)

# convert to long format 
df_gas = pd.melt(frame=df_gas, id_vars=['Date', 'Fuel Type'], 
        var_name='city',value_name='price')
df_gas = df_gas.query(
    "city == 'Toronto East/Est' &  `Fuel Type` == 'Regular Unleaded Gasoline'")

# read in inflation data
df_inflation = pd.read_csv('data/raw/1810000401_databaseLoadingData.csv')
df_inflation.columns = df_inflation.columns.str.lower()
df_gas_inf = df_inflation.query("`products and product groups` == 'Gasoline'").copy()
                                   
# compute dollars in terms of 2025 dollars
# 1) Compute CPIs centered on 2025
value_25 = df_gas_inf.query('ref_date == "2025-06"')['value']
df_gas_inf['cpi_2025'] = value_25.values/df_gas_inf['value']

# merge in CPI value based on year-month and compute 
df_gas['year_month'] = df_gas["Date"].dt.strftime("%Y-%m")
df_gas_inf["ref_date"] = pd.to_datetime(df_gas_inf["ref_date"])
df_gas_inf['date_ymw'] = df_gas_inf["ref_date"].dt.strftime("%Y-%m")

cols = ['date_ymw', 'value', 'cpi_2025']
df_gas = pd.merge(left=df_gas, right=df_gas_inf[cols],
                  left_on='year_month', right_on='date_ymw', how='left')


# NOTE: replace 1990-01 with 1990-02 and 2025-07 with 2025-06
df_gas.loc[df_gas['year_month'].eq("1990-01"), 'cpi_2025'] = df_gas.loc[df_gas['year_month'].eq("1990-02"), 'cpi_2025'].iloc[0]
df_gas.loc[df_gas['year_month'].eq("2025-07"), 'cpi_2025'] = df_gas.loc[df_gas['year_month'].eq("2025-06"), 'cpi_2025'].iloc[0]

assert (df_gas['cpi_2025'].isna().sum() == 0), '''Check missing cpi_2025 values'''

# compute price adjusted metrics 
df_gas['price_2025'] = df_gas['price'] * df_gas['cpi_2025']

assert (df_gas['price_2025'].isna().sum() == 0), '''Check missing price_2025 values'''

# create toronto data
df_toronto = df_gas.query("city == 'Toronto East/Est'" )
df_toronto.to_csv('data/processed/data_toronto_proc.csv', index=False)


# endregion 

# region 2) Inspect gas price data 

(ggplot(data=df_toronto, mapping=aes(x='Date', y='price_2025')) + 
    geom_line() + 
    scale_x_datetime(date_breaks='5 year', date_labels='%Y') + 
    theme_classic())

# endregion 

# region 3) Modelling 

# create training and test sets
df_train = df_toronto.loc[df_toronto['Date'].lt('2021')]
df_test = df_toronto.loc[df_toronto['Date'].ge('2021')]

# region 3a) ETS modeling

"""
Advantages over classical decomposition, SEATS, and X-11 methods
1. Handle any type of seasonality (not just monthly and quarterly data)
2. Seasonal component can change over time 
3. Smoothness of trend cycle can be controlled by the user. 
4. Trend cycle can be made robust to outliers.
"""

# initial examination of seasonal components (group by year and examine monthly average )
df_toronto['year'] = df_toronto['Date'].dt.year
df_toronto['month'] = df_toronto['Date'].dt.month
df_toronto['month_name'] = df_toronto['Date'].dt.strftime('%b')

df_month_avg =  (df_toronto.groupby(['year', 'month_name', 'month'])['price_2025'].mean().reset_index())

month_labels = {i: calendar.month_abbr[i] for i in range(1, 13)}

# NOTE: toronto gas price data doesn't have strong seasonality. 
(ggplot(data=df_month_avg.loc[df_month_avg['year'].ge(2005)], 
        mapping=aes(x='month', y='price_2025', group='factor(year)', color='factor(year)')) +
    geom_line() +
    scale_x_continuous(breaks=list(month_labels.keys()),
                       labels=list(month_labels.values())) +
    theme_classic())

(df_toronto.groupby('month_name')['price_2025'].mean().reset_index())

"""
Compare following models using 10-fold cross-validation:

1) ETS(A,N,N); simple exponential smoothing model w/ additive error
2) ETS (A,A_d,N): Holt's linear method with damped trend (because gas prices 
actually remain constant when accounting for inflation)
3) ETS (A,A,A): Holt-Winters' additive method with seasonal component. The 
data don't appear to have strong seasonality 

Holt-Winters' model is the best reference model. 
"""

# ETS model (compare simple ETS(A,A,A) vs )
autoets = AutoETS(season_length=52)
autoets = autoets.fit(y=df_train["price_2025"].values)
autoets.model_["method"]

autoets.model_["fit"]

sf = StatsForecast(
    models=[AutoETS(season_length=52, model="ANN", alias="SES"), 
            AutoETS(season_length=52, model="AAN", alias="Holt"), 
            AutoETS(season_length=52, model="ANA", alias="error_season"), 
            AutoETS(season_length=52, model="ANA", alias="Holt_Winter")], freq="WE")

# 1. Use cross-validation  (10 sets with roling window size of 1 time period) 
cols = ['Fuel Type', 'Date', 'price_2025']
forecast_int = 1
df_cv = sf.cross_validation(h=forecast_int, step_size=12, n_windows=10, 
                            df=df_train[cols], target_col='price_2025', 
                            time_col='Date', id_col='Fuel Type')

print("CV performance: \n", 
      evaluate(df=df_cv, metrics=[rmse, mae, mape], 
               models=['SES', 'Holt', 'error_season', 'Holt_Winter'], 
               target_col='price_2025', time_col='Date', id_col='Fuel Type'))

