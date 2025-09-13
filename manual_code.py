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
for anomaly, file in files.items():
    df = pd.read_csv(file)
    columns = df.columns.to_list()
    df = df[['date', 'portLS']]
    df['date'] = pd.to_datetime(df['date'], format='mixed')
    df.rename(columns={'portLS':anomaly}, inplace=True)
    df = df.dropna(subset=[anomaly])
    if data.empty:
        data = df
    else:
        data = pd.merge(data, df, on='date', how='outer')
    #df.info()
    #df.head()
    #print('\n \n')
columns_to_modify = list(files.keys())
data.dropna(subset = columns_to_modify, inplace=True)
data[columns_to_modify] = data[columns_to_modify]/ 100
#print(data.head())
#print(data.tail())
#print(data.info())

# Extract data within Regime Periods
regime_data = data[(data['date'] >= start_date) & (data['date'] <= end_date)] 
#print(regime_data.head())
#print(regime_data.tail())
#print(regime_data.info())

# Fama-French Factors
ff_factors = pd.read_csv('copy.csv')
col_ff = ff_factors.columns.to_list()
col_ff[0] = 'date'
ff_factors.columns = col_ff
ff_factors['date'] = pd.to_datetime(ff_factors['date'], format='%Y%m')
#print(ff_factors.head())
#print(ff_factors.tail())
#ff_factors.info()
# Extract data within Thesis Timeframe
modify_ff_cols = ['Mkt-RF', 'SMB', 'HML', 'RF']
ff_factors[modify_ff_cols] = ff_factors[modify_ff_cols].replace(-99.99, np.nan)
ff_factors.dropna(subset=modify_ff_cols, inplace=True)
ff_factors[modify_ff_cols] = ff_factors[modify_ff_cols] / 100
ff_factors = ff_factors[(ff_factors['date'] >= start_date) & (ff_factors['date'] <= end_date)] 
#print(ff_factors.head())
#print(ff_factors.tail())    
#ff_factors.info()

# Merge Regime Data with Fama-French Factors
# Clean and efficient method
def merge_by_month_year(df1, df2, date_col1='date', date_col2='date', how='inner'):
    # Create period columns
    df1 = df1.copy()
    df2 = df2.copy()
    
    df1['_merge_key'] = df1[date_col1].dt.to_period('M')
    df2['_merge_key'] = df2[date_col2].dt.to_period('M')
    df2 = df2.drop(date_col2, axis=1)  # Drop original date column to avoid duplication
    
    # Merge
    result = pd.merge(df1, df2, on='_merge_key', how=how)
    
    # Clean up
    result = result.drop('_merge_key', axis=1)
    
    return result

# Usage
regime_ff_merged = merge_by_month_year(regime_data, ff_factors, how='left')
print(regime_ff_merged.head())
print(regime_ff_merged.tail())
print(regime_ff_merged.info())

# Add 'Regime' column to regime_ff_merged
def add_regime_column(df, regime_periods):
    """
    Add Regime column using simple loop logic
    """
    df = df.copy()  # Avoid SettingWithCopyWarning
    df['Regime'] = None  # Default value
    
    for regime, (start_date, end_date) in regime_periods.items():
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        mask = (df['date'] >= start_dt) & (df['date'] <= end_dt)
        df.loc[mask, 'Regime'] = regime
    
    return df

# Apply the function
regime_ff_merged = add_regime_column(regime_ff_merged, regime_periods)
#print(regime_ff_merged.head(15))
#print(regime_ff_merged.tail(15))
#print(regime_ff_merged.info())

# Verify Regime Month Count 
counts = regime_ff_merged['Regime'].value_counts()
#print("Regime Month Counts:")
#print(counts)

# NOTE: Missing values in Sample Period already verified earlier

# NOTE: Date alignment between factors and anomalies already verified earlier

# Data Quality Assurance for Appendix
def data_quality_summary(df):
    """
    Quick data quality overview
    """
    print(f"Dataset Shape: {df.shape}")
    print(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print("\nData Types:")
    print(df.dtypes.value_counts())
    print("\nMissing Values:")
    missing = df.isnull().sum()
    print(missing[missing > 0].sort_values(ascending=False))
    print(f"\nDuplicate Rows: {df.duplicated().sum()}")
    print(f"Unique Values per Column:")
    print(df.nunique().sort_values(ascending=False))

#data_quality_summary(regime_ff_merged)