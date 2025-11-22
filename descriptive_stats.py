import pandas as pd
from functions import (
    load_data,
    load_fama_french_factors_to_data,
    add_regime_column,
    calculate_excess_returns,
    calculate_hit_precentage_by_regime,
    calculate_regime_sharpe_ratios,
    calculate_t_stats_for_strategies,
    perform_welchs_t_test,
    latex_descriptive_statistics_data_prep
)

#Thesis Timeframe
start_date = pd.to_datetime('2003-01-01')
end_date = pd.to_datetime('2014-05-31')

#Regime classification 
regime_periods = {
    'Pre-Crisis': ('2003-01-01', '2007-11-30'),
    'Crisis': ('2007-12-01', '2009-06-30'),
    'Post-Crisis': ('2009-07-01', '2014-05-31')
}

#Files to Load
files_to_load = {
    'Accruals' : './Descriptive Stats Data/Accruals.csv',
    'Assest Growth': './Descriptive Stats Data/AssetGrowth.csv',
    'BM': './Descriptive Stats Data/BM.csv',
    'Gross Profit': './Descriptive Stats Data/GP.csv',
    'Momentum': './Descriptive Stats Data/Mom12m.csv',
    'Leaverage Ret': './Descriptive Stats Data/Leverage_ret.csv',
}
anomaly_cols = list(files_to_load.keys())

ff_factors_file = './Descriptive Stats Data/FF_Factors_clean.csv'

# Load Data
data = load_data(files_to_load)
# print("Initial Data Loaded:")
# print(round(data,3))

# Extract data within Regime Periods
data = data[(data['date'] >= start_date) & (data['date'] <= end_date)] 
# print("Data after filtering by Thesis Timeframe:")
# print(round(data,3))

# Load Fama-French Factors
data = load_fama_french_factors_to_data(ff_factors_file, data, start_date, end_date)
# print("Data after loading anomalies and Fama-French factors:")
# print(round(data,3))

# Add Regime Column
data = add_regime_column(data, regime_periods)
# print("Data with Regime Column Added:")
# print(round(data,3))

# Export Fama-French Factors with Regimes for regression analysis
# print("FF Factors for Regression:")
# ff_export = data[['date', 'Mkt-RF', 'SMB', 'HML', 'RF', 'Regime']]
# # print(ff_export.head())
# # print(ff_export.info())
# ff_export.to_excel('./Regression Data/fama_french_factors.xlsx')

# # Verify Regime Counts
# print("\n\n=== Regime Counts ===\n")
# print(data['Regime'].value_counts())

# Calculate Excess Returns
excess_returns = calculate_excess_returns(data, anomaly_cols)
# print("Excess Returns DataFrame:")
# print(round(excess_returns,3))
# excess_returns.to_excel('./Regression Data/excess_returns.xlsx')


# Mean Monthly Return by Regime
monthly_mean = excess_returns.groupby('Regime')[anomaly_cols].mean().round(4)
# print("\n\n=== Mean Monthly Excess Returns by Regime ===\n")
# print(round(monthly_mean,3)) 

# Standard Deviation by Regime
monthly_std = excess_returns.groupby('Regime')[anomaly_cols].std().round(4)
# print("\n\n=== Standard Deviation of Excess Returns ===\n")
# print(round(monthly_std,3))

# Number of Observations by Regime
monthly_count = excess_returns.groupby('Regime')[anomaly_cols].count()
# print("\n\n=== Observations for Regimes ===\n")
# print(monthly_count)

# Skewness by Regime
monthly_skew = excess_returns.groupby('Regime')[anomaly_cols].skew().round(4)
# print("\n\n=== Skewness of Excess Returns ===\n")
# print(round(monthly_skew,3))

# Kurtosis by Regime
monthly_kurt = excess_returns.groupby('Regime')[anomaly_cols].apply(lambda x: x.kurtosis()).round(4)
# print("\n\n=== Fischer's Kurtosis of Excess Returns ===\n")
# print(round(monthly_kurt,3))

#Hit Percentage by Regime
hit_precentage = calculate_hit_precentage_by_regime(excess_returns, anomaly_cols)
# print("\n\n=== Hit Percentage by Regime ===\n")
# print(round(hit_precentage,3))

# Sharpe Ratios by Anomaly
sharpe_ratio_by_regime = calculate_regime_sharpe_ratios(excess_returns, anomaly_cols)
# print("\n\n=== Sharpe Ratios by Regime ===\n")
# print(round(sharpe_ratio_by_regime,3))

# # Newey-West t-statistics for each anomaly OVERALL
# t_stats_df = calculate_t_stats_for_strategies(excess_returns, anomaly_cols, lags=12)
# print("\n\n=== Newey-West t-statistics for Each Anomaly ===\n")
# print(t_stats_df)

# Prepare data subsets for regime-wise Newey-West t-statistics
pre_crisis_returns = excess_returns[excess_returns['Regime'] == 'Pre-Crisis']
crisis_returns = excess_returns[excess_returns['Regime'] == 'Crisis']
post_crisis_returns = excess_returns[excess_returns['Regime'] == 'Post-Crisis']

# Newey-West t-statistics for each anomaly by Regime
pre_crisis_t_stats = calculate_t_stats_for_strategies(pre_crisis_returns, anomaly_cols, lags=12)
crisis_t_stats = calculate_t_stats_for_strategies(crisis_returns, anomaly_cols, lags=12)
post_crisis_t_stats = calculate_t_stats_for_strategies(post_crisis_returns, anomaly_cols, lags=12)
# print("\n\n=== Newey-West t-statistics for Pre-Crisis ===\n")
# print(round(pre_crisis_t_stats,3))
# print("\n\n=== Newey-West t-statistics for Crisis ===\n")
# print(round(crisis_t_stats,3))
# print("\n\n=== Newey-West t-statistics for Post-Crisis ===\n")
# print(round(post_crisis_t_stats,3))

# Welch's t-test between Regimes
pre_vs_crisis_results = perform_welchs_t_test(pre_crisis_returns, crisis_returns, anomaly_cols, 'Pre-Crisis', 'Crisis')
crisis_vs_post_results = perform_welchs_t_test(crisis_returns, post_crisis_returns, anomaly_cols, 'Crisis', 'Post-Crisis')
# Export Comparison Results to Excel
with pd.ExcelWriter('./Descriptive Stats Results/welchs_t_test_results.xlsx') as writer:
    pre_vs_crisis_results.to_excel(writer, sheet_name='Pre-Crisis vs Crisis')
    crisis_vs_post_results.to_excel(writer, sheet_name='Crisis vs Post-Crisis')


print("\n\n=== Welch's t-test: Pre-Crisis vs Crisis ===\n")
print(round(pre_vs_crisis_results, 3))
print("\n\n=== Welch's t-test: Crisis vs Post-Crisis ===\n")
print(round(crisis_vs_post_results, 3))

# Latex Table Generation for Descriptive Statistics 
pre_crisis_descriptive_stats = latex_descriptive_statistics_data_prep(
    'Pre-Crisis',
    anomaly_cols,
    monthly_mean,
    monthly_std,
    monthly_count,
    monthly_skew,
    monthly_kurt,
    pre_crisis_t_stats.set_index('Strategy'),
    hit_precentage,
    sharpe_ratio_by_regime
)

crisis_descriptive_stats = latex_descriptive_statistics_data_prep(
    'Crisis',
    anomaly_cols,
    monthly_mean,
    monthly_std,
    monthly_count,
    monthly_skew,
    monthly_kurt,
    crisis_t_stats.set_index('Strategy'),
    hit_precentage,
    sharpe_ratio_by_regime
)

post_crisis_descriptive_stats = latex_descriptive_statistics_data_prep(
    'Post-Crisis',
    anomaly_cols,
    monthly_mean,
    monthly_std,
    monthly_count,
    monthly_skew,
    monthly_kurt,
    post_crisis_t_stats.set_index('Strategy'),
    hit_precentage,
    sharpe_ratio_by_regime
)   
# Export Descriptive Statistics to excel
with pd.ExcelWriter('./Descriptive Stats Results/descriptive_statistics_tables.xlsx') as writer:
    pre_crisis_descriptive_stats.to_excel(writer, sheet_name='Pre-Crisis')
    crisis_descriptive_stats.to_excel(writer, sheet_name='Crisis')
    post_crisis_descriptive_stats.to_excel(writer, sheet_name='Post-Crisis')

#latex_string = "\n\n".join([pre_crisis_descriptive_stats, crisis_descriptive_stats, post_crisis_descriptive_stats, pre_crisis_descriptive_stats])
# with open('descriptive_statistics_tables.tex', 'w') as f:
#     f.write(latex_string)

# #Print descriptive statistics 
# print("\n\n=== Pre-Crisis Descriptive Statistics Tables ===\n")
# print(pre_crisis_descriptive_stats)
# print("\n\n=== Crisis Descriptive Statistics Tables ===\n")
# print(crisis_descriptive_stats)
# print("\n\n=== Post-Crisis Descriptive Statistics Tables ===\n")
# print(post_crisis_descriptive_stats)

# Calculate Recovery Rate and Crisis Drop
# recovery_metric = calculate_recovery_rate_and_crisis_drop(monthly_mean)
# print("\n\n=== Recovery Rate and Crisis Drop ===\n")
# recovery_metric = calculate_recovery_rate_and_crisis_drop(monthly_mean)
# print(round(recovery_metric,3))

# Correlation Matrix on Excess Returns of Anomalies for Each Regime
pre_crisis_corr = pre_crisis_returns[anomaly_cols].corr().round(2)
crisis_corr = crisis_returns[anomaly_cols].corr().round(2)
post_crisis_corr = post_crisis_returns[anomaly_cols].corr().round(2)
print("\n\n=== Correlation Matrix: Pre-Crisis ===\n")
print(pre_crisis_corr)
print("\n\n=== Correlation Matrix: Crisis ===\n")
print(crisis_corr)
print("\n\n=== Correlation Matrix: Post-Crisis ===\n")
print(post_crisis_corr)
# Export Correlation Matrices to Excel
with pd.ExcelWriter('./Descriptive Stats Results/correlation_matrices.xlsx') as writer:
    pre_crisis_corr.to_excel(writer, sheet_name='Pre-Crisis')
    crisis_corr.to_excel(writer, sheet_name='Crisis')
    post_crisis_corr.to_excel(writer, sheet_name='Post-Crisis')