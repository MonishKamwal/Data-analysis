import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime
import matplotlib.pyplot as plt 

# Thesis Timeframe
start_date = pd.to_datetime('2003-01-01')
end_date = pd.to_datetime('2014-05-31')

# Regime Periods
# Regime classification
regime_periods = {
    'Pre-Crisis': ('2003-01-01', '2007-11-30'),
    'Crisis': ('2007-12-01', '2009-06-30'),
    'Post-Crisis': ('2009-07-01', '2014-05-31')
}

# Files
files = {
    'Accruals' : 'Accruals.csv',
    'Assest Growth': 'AssetGrowth.csv',
    'BM': 'BM.csv',
    'Gross Profit': 'GP.csv',
    'Momentum': 'Mom12m.csv',
    'Leaverage Ret': 'Leverage_ret.csv',
}
#print()
data = pd.DataFrame()
for anomaly, files in files.items():
    df = pd.read_csv(files)
    columns = df.columns.to_list()
    df = df[['date', 'portLS']]
    df['date'] = pd.to_datetime(df['date'], format='mixed')
    df.rename(columns={'portLS':anomaly}, inplace=True)
    df = df.dropna(subset=[anomaly])
    if data.empty:
        data = df
    else:
        data = pd.merge(data, df, on='date', how='outer')
    df.info()
    df.head()
    print('\n \n')
data.dropna(subset = ['Accruals', 'Assest Growth', 'BM', 'Gross Profit', 'Momentum', 'Leaverage Ret'], inplace=True)
#print(data.head())
#print(data.tail())
#print(data.info())

# Extract data within Regime Periods
regime_data = data[(data['date'] >= start_date) & (data['date'] <= end_date)] 
#print(regime_data.head())
#print(regime_data.tail())
#print(regime_data.info())


