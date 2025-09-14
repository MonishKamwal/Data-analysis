import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime
import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches
import seaborn as sns
import itertools
import matplotlib.cm as cm
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
#print(regime_ff_merged.head())
#print(regime_ff_merged.tail())
#print(regime_ff_merged.info())

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

# Mean Monthly Return by Regime
anomaly_cols = list(files.keys())
monthly_mean = regime_ff_merged.groupby('Regime')[anomaly_cols].mean().round(4)
#print(monthly_mean) # NOTE: Multiply by 100 for % display

# Standard Deviation of Monthly Returns by Regime
monthly_std = regime_ff_merged.groupby('Regime')[anomaly_cols].std().round(4)
#print(monthly_std) # NOTE: Multiply by 100 for % display

# Number of Observations by Regime
monthly_count = regime_ff_merged.groupby('Regime')[anomaly_cols].count()
#print(monthly_count) # NOTE: Should be 56, 19, 58 for each regime respectively

# Calculate Excess Returns = anomaly Return - RF for Sharpe Ratio and Hit Rate
Excess_return_df = regime_ff_merged.copy()
excess_return_cols = []
for anomaly in list(files.keys()):
    excess_col = anomaly + '_Excess'
    excess_return_cols.append(excess_col)
    Excess_return_df[excess_col] = Excess_return_df[anomaly] - Excess_return_df['RF']
#print(excess_return_cols)
#print(Excess_return_df.head())
#print(Excess_return_df.tail())
#print(Excess_return_df.info())

# Calculate Sharpe Ratio for each anomaly
sharpe_ratios = {}
for anomaly in excess_return_cols:
    mean_excess = Excess_return_df[anomaly].mean()
    std_excess = Excess_return_df[anomaly].std()
    if std_excess == 0:
        sharpe_ratio = np.nan  # Avoid division by zero
    sharpe_ratio = (mean_excess / std_excess) * np.sqrt(12)  # Annualized Sharpe Ratio
    sharpe_ratios[anomaly] = round(sharpe_ratio, 4)

# Create a DataFrame for easy viewing and export
def create_sharpe_dataframe(sharpe_dict):
    """Convert Sharpe ratios to DataFrame for easy manipulation"""
    sharpe_df = pd.DataFrame(list(sharpe_dict.items()), columns=['Anomaly', 'Sharpe_Ratio'])
    sharpe_df = sharpe_df.sort_values('Sharpe_Ratio', ascending=False, na_position='last')
    return sharpe_df

sharpe_df = create_sharpe_dataframe(sharpe_ratios)
#print(sharpe_df)

# Sharpe Ratios by Regime and Anomaly
def calculate_regime_sharpe_ratios(df):
    """Calculate Sharpe ratios by regime and anomaly"""
    regime_sharpe = {}
    
    for regime in df['Regime'].unique():
        regime_data = df[df['Regime'] == regime]
        sharpe_ratios = {}
        
        for col in excess_return_cols:
            mean_return = regime_data[col].mean()
            std_return = regime_data[col].std()
            sharpe_ratio = mean_return  / std_return
            sharpe_ratios[col] = sharpe_ratio
        
        regime_sharpe[regime] = sharpe_ratios
    
    return pd.DataFrame(regime_sharpe).T

# Result format: 3×6 table (regimes × anomalies)
regime_sharpe_table = calculate_regime_sharpe_ratios(Excess_return_df)
#print("Sharpe Ratios by Regime and Anomaly:")
#print(regime_sharpe_table.round(4))


# Calculate Hit Rate for each anomaly
def quick_hit_percentage(df, column):
    """Quick calculation of hit percentage for a single column"""
    returns = df[column].dropna()
    return (returns > 0).mean() * 100

hit_rates_excess = {anomaly: quick_hit_percentage(Excess_return_df, anomaly) for anomaly in excess_return_cols}
hit_rate_excess_df = pd.DataFrame(list(hit_rates_excess.items()), columns=['Anomaly Excess', 'Hit_Rate'])
#hit_rate_excess_df.info()
#print("Hit Rates for Each Anomaly based on Excess Returns:")
#print(hit_rate_excess_df, end='\n\n')

hit_rates = {anomaly: quick_hit_percentage(Excess_return_df, anomaly) for anomaly in anomaly_cols}
hit_rate_df = pd.DataFrame(list(hit_rates.items()), columns=['Anomaly', 'Hit_Rate'])
#hit_rate_df.info()
#print("Hit Rates for Each Anomaly:")
#print(hit_rate_df)

# Newey-West HAC Standard Errors and t-statistics
def newey_west_variance(returns, lags=12):
    returns = np.array(returns)
    n = len(returns)
    # Demean the returns
    mean_return = np.mean(returns)
    demeaned_returns = returns - mean_return
    # Calculate gamma_0 (variance)
    gamma_0 = np.mean(demeaned_returns**2)
    # Calculate autocovariances gamma_j for j = 1, 2, ..., lags
    gamma_j_sum = 0
    for j in range(1, lags + 1):
        if j < n:
            # Calculate gamma_j
            gamma_j = np.mean(demeaned_returns[j:] * demeaned_returns[:-j])
            # Bartlett kernel weight: w_j = 1 - j/(lags+1)
            weight = 1 - j / (lags + 1)
            gamma_j_sum += 2 * weight * gamma_j
    # Newey-West variance estimator
    nw_variance = (gamma_0 + gamma_j_sum) / n
    return nw_variance

def calculate_newey_west_t_stat(returns, lags=12):
    returns = np.array(returns)
    returns = returns[~np.isnan(returns)]  # Remove NaN values
    n = len(returns)
    if n == 0:
        return {
            'mean': np.nan,
            't_statistic': np.nan,
            'p_value': np.nan,
            'nw_std_error': np.nan,
            'observations': 0,
            'lags_used': lags
        }
    # Calculate sample mean
    mean_return = np.mean(returns)
    # Calculate Newey-West variance
    nw_variance = newey_west_variance(returns, lags)
    nw_std_error = np.sqrt(nw_variance)
    # Calculate t-statistic
    t_statistic = mean_return / nw_std_error if nw_std_error != 0 else np.nan
    # Calculate p-value (two-tailed test)
    # Use t-distribution with n-1 degrees of freedom
    if not np.isnan(t_statistic):
        p_value = 2 * (1 - stats.t.cdf(abs(t_statistic), df=n-1))
    else:
        p_value = np.nan
    return {
        'mean': mean_return,
        't_statistic': t_statistic,
        'p_value': p_value,
        'nw_std_error': nw_std_error,
        'observations': n,
        'lags_used': lags
    }

def calculate_t_stats_for_strategies(df, columns, lags=12):
    results = {}
    for column in columns:
        results[column] = calculate_newey_west_t_stat(df[column], lags)
    return results

def display_t_statistics(results_dict):
    """Display t-statistics in a formatted table"""
    print("=" * 80)
    print("NEWEY-WEST HAC t-STATISTICS (H0: Mean = 0)")
    print("=" * 80)
    print(f"{'Strategy':<25} {'Mean':<10} {'t-stat':<10} {'p-value':<12} {'Significant':<12}")
    print("-" * 80)
    # Sort by absolute t-statistic (descending)
    sorted_strategies = sorted(results_dict.items(), 
                             key=lambda x: abs(x[1]['t_statistic']) if not np.isnan(x[1]['t_statistic']) else -1, 
                             reverse=True)
    
    for strategy, results in sorted_strategies:
        strategy_name = strategy.replace('_Excess', '')  # Clean name for display
        mean_val = results['mean']
        t_stat = results['t_statistic']
        p_val = results['p_value']
        
        # Determine significance
        if not np.isnan(p_val):
            if p_val < 0.01:
                significance = "***"
            elif p_val < 0.05:
                significance = "**"
            elif p_val < 0.10:
                significance = "*"
            else:
                significance = ""
        else:
            significance = "N/A"
        
        if not any(np.isnan([mean_val, t_stat, p_val])):
            print(f"{strategy_name:<25} {mean_val:<10.4f} {t_stat:<10.3f} {p_val:<12.4f} {significance:<12}")
        else:
            print(f"{strategy_name:<25} {'N/A':<10} {'N/A':<10} {'N/A':<12} {'N/A':<12}")
    
    print("-" * 80)
    print("Significance levels: *** p<0.01, ** p<0.05, * p<0.10")
    print(f"Newey-West lags used: {list(results_dict.values())[0]['lags_used']}")
    print("=" * 80)

def create_t_statistics_dataframe(results_dict):
    """Convert t-statistics results to DataFrame"""
    data = []
    for strategy, results in results_dict.items():
        strategy_name = strategy.replace('_Excess', '')
        
        # Determine significance level
        p_val = results['p_value']
        if not np.isnan(p_val):
            if p_val < 0.01:
                sig_level = "***"
            elif p_val < 0.05:
                sig_level = "**"
            elif p_val < 0.10:
                sig_level = "*"
            else:
                sig_level = ""
        else:
            sig_level = "N/A"
        
        data.append({
            'Strategy': strategy_name,
            'Mean': results['mean'],
            'T_Statistic': results['t_statistic'],
            'P_Value': results['p_value'],
            'NW_Std_Error': results['nw_std_error'],
            'Observations': results['observations'],
            'Significance': sig_level
        })
    
    t_stats_df = pd.DataFrame(data)
    t_stats_df = t_stats_df.sort_values('T_Statistic', key=abs, ascending=False, na_position='last')
    return t_stats_df


# Calculate Newey-West t-statistics
t_stats_results = calculate_t_stats_for_strategies(Excess_return_df, excess_return_cols, lags=12)

# Display results
#display_t_statistics(t_stats_results)

# Create DataFrame for further analysis
t_stats_df = create_t_statistics_dataframe(t_stats_results)
#print("\nDataFrame format:")
#print(t_stats_df)
# NOTE: Cross checked the results of the t-statistics with Claude. It was suspicious at first but after reveiwing the data it said it might actually be correct
# Save to CSV if needed
#t_stats_df.to_csv('newey_west_t_statistics.csv', index=False)


# Regime Comparison Test - Welch's T-test
pre_crisis_returns = Excess_return_df[Excess_return_df['Regime'] == 'Pre-Crisis']
crisis_returns = Excess_return_df[Excess_return_df['Regime'] == 'Crisis']
post_crisis_returns = Excess_return_df[Excess_return_df['Regime'] == 'Post-Crisis']

#pre_crisis_returns.info()
#crisis_returns.info()
#post_crisis_returns.info()

#print(pre_crisis_returns.head())
#print(crisis_returns.head())
#print(post_crisis_returns.head())   

t_test_results_pre_crisis_vs_crisis = {}
t_test_results_crisis_vs_post_crisis = {}

def perform_welchs_t_test(group1, group2, columns):
    results = {}
    for col in columns:
        t_stat, p_val = stats.ttest_ind(group1[col], group2[col], equal_var=False, nan_policy='omit')
        results[col] = {
            't_statistic': t_stat,
            'p_value': p_val,
            'mean_group1': group1[col].mean(),
            'mean_group2': group2[col].mean(),
        }
    return results

t_test_results_pre_crisis_vs_crisis = perform_welchs_t_test(pre_crisis_returns, crisis_returns, excess_return_cols)
t_test_results_crisis_vs_post_crisis = perform_welchs_t_test(crisis_returns, post_crisis_returns, excess_return_cols)
# Display results
def display_welchs_t_test_results(results_dict, group1_name, group2_name):
    print(f"\nWelch's T-test Results: {group1_name} vs {group2_name}")
    print("=" * 80)
    print(f"{'Strategy':<25} {'Mean_'+group1_name:<15} {'Mean_'+group2_name:<15} {'t-stat':<10} {'p-value':<12} {'Significant':<12}")
    print("-" * 80)
    
    for strategy, results in results_dict.items():
        mean1 = results['mean_group1']
        mean2 = results['mean_group2']
        t_stat = results['t_statistic']
        p_val = results['p_value']
        
        # Determine significance
        if not np.isnan(p_val):
            if p_val < 0.01:
                significance = "***"
            elif p_val < 0.05:
                significance = "**"
            elif p_val < 0.10:
                significance = "*"
            else:
                significance = ""
        else:
            significance = "N/A"
        
        if not any(np.isnan([mean1, mean2, t_stat, p_val])):
            print(f"{strategy:<25} {mean1:<15.4f} {mean2:<15.4f} {t_stat:<10.3f} {p_val:<12.4f} {significance:<12}")
        else:
            print(f"{strategy:<25} {'N/A':<15} {'N/A':<15} {'N/A':<10} {'N/A':<12} {'N/A':<12}")
    
    print("-" * 80)
    print("Significance levels: *** p<0.01, ** p<0.05, * p<0.10")
    print("=" * 80)
#display_welchs_t_test_results(t_test_results_pre_crisis_vs_crisis, 'Pre-Crisis', 'Crisis')
#display_welchs_t_test_results(t_test_results_crisis_vs_post_crisis, 'Crisis', 'Post-Crisis')

# Calculate and print 'Crisis Drop' = Mean(Crisis) - Mean(Pre-Crisis)
print("\nCrisis Drop (Mean Crisis - Mean Pre-Crisis):")
drop_results = {}
for strategy, results in t_test_results_pre_crisis_vs_crisis.items():
    drop = results['mean_group2'] - results['mean_group1']
    drop_results[strategy] = drop
# print drop_resutls in a formatted way
print(f"{'Strategy':<25} {'Crisis Drop':<15}")
print("-" * 40)
for strategy, drop in drop_results.items():
    print(f"{strategy:<25} {drop:<15.4f}")
print("-" * 40) 

# Calculate Recovery Rate = (Post-Crisis Mean - Crisis Mean)* 100 / (Pre-Crisis Mean - Crisis Mean) (From drop_results)
print("\nRecovery Rate (%):")
recovery_results = {}
for strategy, results in t_test_results_pre_crisis_vs_crisis.items():
    pre_crisis_mean = results['mean_group1']
    crisis_mean = results['mean_group2']
    post_crisis_mean = t_test_results_crisis_vs_post_crisis[strategy]['mean_group2']
    
    if (pre_crisis_mean - crisis_mean) != 0:
        recovery_rate = ((post_crisis_mean - crisis_mean) * 100) / (pre_crisis_mean - crisis_mean)
    else:
        recovery_rate = np.nan  # Avoid division by zero
    
    recovery_results[strategy] = recovery_rate
# print recovery_results in a formatted way
"""
print(f"{'Strategy':<25} {'Recovery Rate (%)':<20}")
print("-" * 45)
for strategy, rate in recovery_results.items():
    print(f"{strategy:<25} {rate:<20.2f}")
print("-" * 45) 
"""

# Visualizations
# Visualization for each anomaly with regime shading

regime_colors = {
    'Pre-Crisis': '#e0e0e0',
    'Crisis': '#ffcccc',
    'Post-Crisis': '#ccffcc'
}

# Plot all anomalies as base 100 index on one graph

fig, ax = plt.subplots(figsize=(14, 6))
color_cycle = itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
anomaly_colors = {anomaly: next(color_cycle) for anomaly in anomaly_cols}

for anomaly in anomaly_cols:
    returns = Excess_return_df[anomaly].fillna(0)
    index = 100 * (1 + returns).cumprod()
    ax.plot(Excess_return_df['date'], index, label=anomaly, color=anomaly_colors[anomaly])

# Shade regimes
for regime, (start, end) in regime_periods.items():
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    ax.axvspan(start_dt, end_dt, color=regime_colors[regime], alpha=0.2, label=regime)

ax.set_title('Long-Short Returns Over Regimes (Base 100 Index, All Anomalies)')
ax.set_xlabel('Date')
ax.set_ylabel('Index (Base 100)')
ax.grid(True)

anomaly_handles = [plt.Line2D([], [], color=anomaly_colors[a], label=a) for a in anomaly_cols]
regime_handles = [mpatches.Patch(color=regime_colors[r], label=r) for r in regime_colors]
handles = anomaly_handles + regime_handles
ax.legend(handles=handles, loc='upper left', fontsize='medium')

plt.tight_layout()
#plt.show()

# Distribution comparisons: Overlaid histograms by regime for each anomaly
for anomaly in anomaly_cols:
    plt.figure(figsize=(10, 6))
    for regime, color in regime_colors.items():
        subset = Excess_return_df[Excess_return_df['Regime'] == regime][anomaly].dropna()
        sns.histplot(subset, kde=False, color=color, label=regime, stat='density', alpha=0.5, bins=20)
    plt.title(f'Distribution of {anomaly} Returns by Regime')
    plt.xlabel('Return')
    plt.ylabel('Density')
    plt.legend(title='Regime')
    plt.tight_layout()
    #plt.show()

# Summary visualization: Bar chart of mean returns by regime

# Enhanced summary bar chart of mean returns by regime

mean_returns = Excess_return_df.groupby('Regime')[anomaly_cols].mean().T
fig, ax = plt.subplots(figsize=(13, 7))

# Use pastel colors for bars
bar_colors = [regime_colors.get(r, cm.Pastel1(i)) for i, r in enumerate(mean_returns.columns)]
bars = mean_returns.plot(kind='bar', ax=ax, color=bar_colors, edgecolor='black', width=0.75)

plt.title('Mean Returns by Regime for Each Anomaly', fontsize=18, fontweight='bold', pad=20)
plt.xlabel('Anomaly', fontsize=14)
plt.ylabel('Mean Return', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.5)

# Add value labels to each bar
for container in ax.containers:
    ax.bar_label(container, fmt='%.3f', label_type='edge', fontsize=12, padding=2)

# Custom legend with larger font and better placement
plt.legend(title='Regime', fontsize=13, title_fontsize=14, loc='lower right', frameon=True)
plt.xticks(rotation=25, ha='right', fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()