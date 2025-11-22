# Load all anomaly data into a single DataFrame
import pandas as pd
import numpy as np
from scipy import stats

def load_data(files):
    data = pd.DataFrame()
    for anomaly, file in files.items():
        df = pd.read_csv(file)
        #columns = df.columns.to_list()
        df = df[['date', 'portLS']]
        df['date'] = pd.to_datetime(df['date'], format='mixed')
        df.rename(columns={'portLS':anomaly}, inplace=True)
        df = df.dropna(subset=[anomaly])
        if data.empty:
            data = df
        else:
            data = pd.merge(data, df, on='date', how='outer')
    columns_to_modify = list(files.keys())
    data.dropna(subset = columns_to_modify, inplace=True)
    data[columns_to_modify] = data[columns_to_modify]/ 100
    return data

# Add Fama-French Factors to data
def load_fama_french_factors_to_data(file, data, start_date, end_date):
    # Fama-French Factors
    ff_factors = pd.read_csv(file)
    col_ff = ff_factors.columns.to_list()
    col_ff[0] = 'date'
    ff_factors.columns = col_ff
    ff_factors['date'] = pd.to_datetime(ff_factors['date'], format='%Y%m')

    # Extract data within Thesis Timeframe
    modify_ff_cols = ['Mkt-RF', 'SMB', 'HML', 'RF']
    ff_factors[modify_ff_cols] = ff_factors[modify_ff_cols].replace(-99.99, np.nan)
    ff_factors.dropna(subset=modify_ff_cols, inplace=True)
    ff_factors[modify_ff_cols] = ff_factors[modify_ff_cols] / 100
    ff_factors = ff_factors[(ff_factors['date'] >= start_date) & (ff_factors['date'] <= end_date)] 

    # Merge with anomaly data
    data['_merge_key'] = data['date'].dt.to_period('M')
    ff_factors['_merge_key'] = ff_factors['date'].dt.to_period('M')
    merged_data = pd.merge(data, ff_factors, on='_merge_key', how='inner')
    merged_data.drop('_merge_key', axis=1, inplace=True)
    merged_data.drop('date_y', axis=1, inplace=True)
    merged_data.rename(columns={'date_x':'date'}, inplace=True)
    return merged_data

# Add 'Regime' column
def add_regime_column(df, regime_periods):
    df = df.copy()  # Avoid SettingWithCopyWarning
    df['Regime'] = None  # Default value
    for regime, (start_date, end_date) in regime_periods.items():
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        mask = (df['date'] >= start_dt) & (df['date'] <= end_dt)
        df.loc[mask, 'Regime'] = regime
    return df

def calculate_excess_returns(df, anomaly_cols):
    excess_returns = pd.DataFrame()
    for col in anomaly_cols:
        excess_returns[col] = df[col] - df['RF']
    excess_returns['date'] = df['date']
    excess_returns['Regime'] = df['Regime']
    excess_returns.set_index('date', inplace=True)
    return excess_returns

# Calculate Hit Rate for each anomaly
def quick_hit_percentage(df, column):
    #Quick calculation of hit percentage for a single column
    returns = df[column].dropna()
    return (returns > 0).mean() * 100

def calculate_hit_precentage_by_regime(df, anomaly_cols):
    #Calculate hit percentage by regime for excess returns
    hit_rates = {}
    for regime in df['Regime'].unique():
        regime_data = df[df['Regime'] == regime]
        regime_hit_rates = {}
        for col in anomaly_cols:
            hit_rate = quick_hit_percentage(regime_data, col)
            regime_hit_rates[col] = hit_rate
        hit_rates[regime] = regime_hit_rates
    return pd.DataFrame(hit_rates).T

# Sharpe Ratios by Regime and Anomaly
def calculate_regime_sharpe_ratios(df, anomaly_cols):
    #Calculate Sharpe ratios by regime and anomaly
    regime_sharpe = {}
    for regime in df['Regime'].unique():
        regime_data = df[df['Regime'] == regime]
        sharpe_ratios = {}
        
        for col in anomaly_cols:
            mean_return = regime_data[col].mean()
            std_return = regime_data[col].std()
            if std_return == 0:
                sharpe_ratio = np.nan  # Avoid division by zero
            else:
                sharpe_ratio = (mean_return  / std_return) * np.sqrt(12)  # Annualized Sharpe Ratio
            sharpe_ratios[col] = round(sharpe_ratio, 4)
        
        regime_sharpe[regime] = sharpe_ratios
    
    return pd.DataFrame(regime_sharpe).T

# HELPER FUNCTION: Newey-West HAC Standard Errors and t-statistics
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

# HELPER FUNCTION: Calculate Newey-West t-statistic for each anomaly
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

def calculate_t_stats_for_strategies(df, anomaly_cols, lags=12):
    results_dict = {}
    for column in anomaly_cols:
        results_dict[column] = calculate_newey_west_t_stat(df[column], lags)
    #Convert t-statistics results to DataFrame
    data = []
    for strategy, results in results_dict.items():
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
            'Strategy': strategy,
            'Mean': results['mean'],
            'T_Statistic': results['t_statistic'],
            'P_Value': results['p_value'],
            'NW_Std_Error': results['nw_std_error'],
            'Observations': results['observations'],
            'Significance': sig_level
        })
    t_stats_df = pd.DataFrame(data)
    return t_stats_df

# Perform Welch's t-test between two groups for specified columns
def perform_welchs_t_test(group1, group2, columns, group1_name, group2_name):
    results = {}
    for col in columns:
        t_stat, p_val = stats.ttest_ind(group1[col], group2[col], equal_var=False, nan_policy='omit')
        results[col] = {
            group1_name: group1[col].mean(),
            group2_name: group2[col].mean(),
            'Difference': group1[col].mean() - group2[col].mean(),
            'T_Statistic': t_stat,
            'P_Value': p_val
        }
    # convert to DataFrame
    results_df = pd.DataFrame(results).T
    return results_df

def calculate_recovery_rate_and_crisis_drop(means):
    df = pd.DataFrame(['Strategy', 'Crisis Drop', 'Recovery Rate', 'Full Recovery?'])
    pre_crisis_means = means.T['Pre-Crisis']
    crisis_means = means.T['Crisis']
    post_crisis_means = means.T['Post-Crisis']
    crisis_drop = pre_crisis_means - crisis_means
    recovery_rate = ((post_crisis_means - crisis_means)*100) / crisis_drop
    full_recovery = recovery_rate >= 100
    recovery_df = pd.DataFrame({
        'Crisis Drop': crisis_drop,
        'Recovery Rate (%)': recovery_rate,
        'Full Recovery?': full_recovery
    })
    return recovery_df
    
def latex_descriptive_statistics_data_prep(regime, anomaly_cols, mean, std, count, skew, kurt, t_stats, hit_rate, sharpe_ratios):
    data = pd.DataFrame({
        "Mean": mean.T[regime],
        "Std Dev": std.T[regime],
        "T-Stat": t_stats['T_Statistic'],
        "Sharpe": sharpe_ratios.T[regime],
        "Skewness": skew.T[regime],
        "Kurtosis": kurt.T[regime],
        "Hit %": hit_rate.T[regime],
        "Observations": count.T[regime]
    })
    return data 
    # print("\n\n=== Descriptive Statistics for " + regime + " ===\n")
    # print(round(data,3))
    # latex = data.to_latex(float_format="%.3f", index=False)
    # return latex
