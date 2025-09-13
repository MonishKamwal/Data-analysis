import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime
import matplotlib.pyplot as plt 

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
    df.info()
    df.head()
    print('\n \n')
    