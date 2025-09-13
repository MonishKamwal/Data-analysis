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
print()
data = pd.DataFrame()

for anomaly, file in files.items():
    df = pd.read_csv(file)
    columns = df.columns.tolist()
    columns[0] = 'Date'
    df.columns = columns
    df.rename(columns={'portLS': anomaly}, inplace=True)
    df = df[['Date', anomaly]]
    df['Date'] = pd.to_datetime(df['Date'], format='mixed')
    df = df.dropna(subset=[anomaly])
    #df.info()
    #df.head()
    print('\n \n')
    if data.empty:
        data = df
    else:
        data = pd.merge(data, df, on='Date', how='outer')

    #data.info()
    #data.head()
    print('\n \n')

print(data.head())
print(data.tail())
data.info()

