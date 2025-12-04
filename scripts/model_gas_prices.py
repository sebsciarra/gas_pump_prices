


"""
Goal
1. Fit Toronto gas prices using 

1) basic ETS/STL (base model for reference)
2) Innovates state space models (slightly more advanced reference)
3) 

Questions
1. Does a random-forest model of TS models (i.e., weak learners) produce less biased model with lower variance? 
2. How to handle patterns of multiple seasonality (weekly, seasons, holidays)? 
3. How to handle feedback effects of current gas prices on future ones? That is, when gas prices become too high
the rate of production slows down. 
4. 
"""

import numpy as np 
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from plotnine import *

if any(fname.startswith("model_gas_prices.py") for fname in os.listdir()):
    os.chdir("..")


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