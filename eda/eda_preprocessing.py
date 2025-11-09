"""
Comprehensive EDA and Preprocessing Pipeline
For Transaction Load Forecasting Project
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 8)

print("=" * 80)
print("TRANSACTION DATA - EXPLORATORY DATA ANALYSIS & PREPROCESSING")
print("=" * 80)

# ============================================
# PART 1: RAW DATA EXPLORATION
# ============================================
print("\n" + "="*80)
print("PART 1: RAW TRANSACTION DATA EXPLORATION")
print("="*80)

print("\n[1.1] Loading raw transaction data...")
df_raw = pd.read_csv('data/synthetic_fraud_data.csv')

print(f"\n📊 Dataset Overview:")
print(f"  • Total records: {len(df_raw):,}")
print(f"  • Columns: {list(df_raw.columns)}")
print(f"  • Memory usage: {df_raw.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

print(f"\n📋 First 10 rows:")
print(df_raw.head(10))

print(f"\n📋 Last 10 rows:")
print(df_raw.tail(10))

print(f"\n🔍 Data Types:")
print(df_raw.dtypes)

print(f"\n📊 Basic Statistics:")
print(df_raw.describe())

print(f"\n❓ Missing Values:")
missing = df_raw.isnull().sum()
print(missing)
if missing.sum() == 0:
    print("  ✓ No missing values found!")

print(f"\n🔢 Unique Values per Column:")
for col in df_raw.columns:
    n_unique = df_raw[col].nunique()
    print(f"  • {col}: {n_unique:,} unique values")

# Check for duplicates
duplicates = df_raw.duplicated().sum()
print(f"\n🔄 Duplicate Rows: {duplicates}")
if duplicates > 0:
    print(f"  ⚠️  Found {duplicates} duplicate transactions")
    print(f"  • Keeping: first occurrence")
    df_raw = df_raw.drop_duplicates()

# ============================================
# PART 2: TEMPORAL ANALYSIS
# ============================================
print("\n" + "="*80)
print("PART 2: TEMPORAL ANALYSIS")
print("="*80)

print("\n[2.1] Converting timestamps...")
df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp'], format='mixed')

# Remove timezone if present
if df_raw['timestamp'].dt.tz is not None:
    print("  • Removing timezone information...")
    df_raw['timestamp'] = df_raw['timestamp'].dt.tz_localize(None)

# Sort by timestamp
df_raw = df_raw.sort_values('timestamp').reset_index(drop=True)

print(f"\n⏰ Time Range:")
print(f"  • Start:    {df_raw['timestamp'].min()}")
print(f"  • End:      {df_raw['timestamp'].max()}")
duration = df_raw['timestamp'].max() - df_raw['timestamp'].min()
print(f"  • Duration: {duration.days} days, {duration.seconds//3600} hours")

print(f"\n📅 Transactions by Date:")
daily_counts = df_raw.groupby(df_raw['timestamp'].dt.date).size()
print(f"  • Average per day: {daily_counts.mean():.0f} transactions")
print(f"  • Min per day:     {daily_counts.min():.0f} transactions")
print(f"  • Max per day:     {daily_counts.max():.0f} transactions")
print(f"  • Std Dev:         {daily_counts.std():.0f} transactions")

# Time gaps
print(f"\n⏱️  Transaction Spacing Analysis:")
df_raw['time_diff'] = df_raw['timestamp'].diff()
print(f"  • Mean gap:   {df_raw['time_diff'].mean()}")
print(f"  • Median gap: {df_raw['time_diff'].median()}")
print(f"  • Min gap:    {df_raw['time_diff'].min()}")
print(f"  • Max gap:    {df_raw['time_diff'].max()}")

# Large gaps (potential issues)
large_gaps = df_raw[df_raw['time_diff'] > pd.Timedelta(minutes=10)]
if len(large_gaps) > 0:
    print(f"\n  ⚠️  Found {len(large_gaps)} gaps > 10 minutes:")
    print(large_gaps[['timestamp', 'time_diff']].head())

# ============================================
# PART 3: AGGREGATE TO TIME SERIES (TPS)
# ============================================
print("\n" + "="*80)
print("PART 3: TIME SERIES AGGREGATION")
print("="*80)

print("\n[3.1] Aggregating to per-minute TPS...")
df_tps = df_raw.set_index('timestamp').resample('1min').size().reset_index(name='tps')
df_tps['tps'] = df_tps['tps'].fillna(0)

print(f"\n✓ Aggregated Results:")
print(f"  • Total time points: {len(df_tps):,} minutes")
print(f"  • Date range: {df_tps['timestamp'].min()} to {df_tps['timestamp'].max()}")

# Check for missing minutes
expected_minutes = (df_tps['timestamp'].max() - df_tps['timestamp'].min()).total_seconds() / 60 + 1
actual_minutes = len(df_tps)
print(f"  • Expected minutes: {expected_minutes:.0f}")
print(f"  • Actual minutes:   {actual_minutes}")
if expected_minutes != actual_minutes:
    print(f"  ⚠️  Missing {expected_minutes - actual_minutes:.0f} minutes!")

print(f"\n📊 TPS Statistics:")
print(f"  • Mean:   {df_tps['tps'].mean():.2f} txn/min")
print(f"  • Median: {df_tps['tps'].median():.2f} txn/min")
print(f"  • Std:    {df_tps['tps'].std():.2f} txn/min")
print(f"  • Min:    {df_tps['tps'].min():.0f} txn/min")
print(f"  • Max:    {df_tps['tps'].max():.0f} txn/min")
print(f"  • CV:     {df_tps['tps'].std() / df_tps['tps'].mean():.2f} (Coefficient of Variation)")

# Zeros analysis
zero_minutes = (df_tps['tps'] == 0).sum()
print(f"\n📉 Zero Transaction Minutes:")
print(f"  • Count: {zero_minutes} ({zero_minutes/len(df_tps)*100:.2f}%)")
if zero_minutes > len(df_tps) * 0.05:
    print(f"  ⚠️  High percentage of zeros - consider implications for modeling")

# Percentiles
print(f"\n📈 TPS Distribution (Percentiles):")
percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
for p in percentiles:
    value = df_tps['tps'].quantile(p/100)
    print(f"  • {p:2d}th percentile: {value:6.1f} txn/min")

# ============================================
# PART 4: TEMPORAL PATTERNS
# ============================================
print("\n" + "="*80)
print("PART 4: TEMPORAL PATTERNS ANALYSIS")
print("="*80)

# Extract time features
df_tps['hour'] = df_tps['timestamp'].dt.hour
df_tps['day_of_week'] = df_tps['timestamp'].dt.dayofweek
df_tps['day_name'] = df_tps['timestamp'].dt.day_name()
df_tps['date'] = df_tps['timestamp'].dt.date
df_tps['is_weekend'] = df_tps['day_of_week'].isin([5, 6])

print("\n[4.1] Hourly Pattern Analysis:")
hourly_stats = df_tps.groupby('hour')['tps'].agg(['mean', 'std', 'min', 'max'])
print(hourly_stats.round(2))

peak_hour = hourly_stats['mean'].idxmax()
low_hour = hourly_stats['mean'].idxmin()
hourly_range = hourly_stats['mean'].max() - hourly_stats['mean'].min()
print(f"\n  🔥 Peak hour: {peak_hour}:00 ({hourly_stats.loc[peak_hour, 'mean']:.2f} txn/min)")
print(f"  💤 Low hour:  {low_hour}:00 ({hourly_stats.loc[low_hour, 'mean']:.2f} txn/min)")
print(f"  📊 Hourly variation: {hourly_range:.2f} txn/min ({hourly_range/hourly_stats['mean'].mean()*100:.1f}%)")

print("\n[4.2] Daily Pattern Analysis:")
daily_stats = df_tps.groupby('day_name')['tps'].agg(['mean', 'std'])
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
daily_stats = daily_stats.reindex(day_order)
print(daily_stats.round(2))

busiest_day = daily_stats['mean'].idxmax()
quietest_day = daily_stats['mean'].idxmin()
daily_range = daily_stats['mean'].max() - daily_stats['mean'].min()
print(f"\n  🔥 Busiest day:  {busiest_day} ({daily_stats.loc[busiest_day, 'mean']:.2f} txn/min)")
print(f"  💤 Quietest day: {quietest_day} ({daily_stats.loc[quietest_day, 'mean']:.2f} txn/min)")
print(f"  📊 Weekly variation: {daily_range:.2f} txn/min ({daily_range/daily_stats['mean'].mean()*100:.1f}%)")

print("\n[4.3] Weekend vs Weekday:")
weekend_mean = df_tps[df_tps['is_weekend']]['tps'].mean()
weekday_mean = df_tps[~df_tps['is_weekend']]['tps'].mean()
print(f"  • Weekday average: {weekday_mean:.2f} txn/min")
print(f"  • Weekend average: {weekend_mean:.2f} txn/min")
print(f"  • Difference:      {abs(weekend_mean - weekday_mean):.2f} ({abs(weekend_mean - weekday_mean)/weekday_mean*100:.1f}%)")

# ============================================
# PART 5: STATIONARITY TESTING
# ============================================
print("\n" + "="*80)
print("PART 5: STATIONARITY ANALYSIS")
print("="*80)

print("\n[5.1] Augmented Dickey-Fuller Test:")
adf_result = adfuller(df_tps['tps'].dropna())
print(f"  • ADF Statistic:  {adf_result[0]:.4f}")
print(f"  • p-value:        {adf_result[1]:.4f}")
print(f"  • Critical Values:")
print(f"  Debug - ADF result type: {type(adf_result)}")
print(f"  Debug - ADF result length: {len(adf_result)}")
print(f"  Debug - ADF critical values type: {type(adf_result[4])}")
try:
    for key, value in adf_result[4].items():
        print(f"    - {key}: {value:.4f}")
except Exception as e:
    print(f"  Debug - Error: {str(e)}")
    print(f"  Debug - Critical values content: {adf_result[4]}")

if adf_result[1] < 0.05:
    print(f"\n  ✓ Data is STATIONARY (p < 0.05)")
    print(f"    → Can use models like ARIMA directly")
else:
    print(f"\n  ⚠️  Data is NON-STATIONARY (p >= 0.05)")
    print(f"    → May need differencing for ARIMA")
    print(f"    → Neural networks (RNN/LSTM) can handle non-stationary data")

print("\n[5.2] Rolling Statistics:")
window = 1440  # 1 day
df_tps['rolling_mean'] = df_tps['tps'].rolling(window=window).mean()
df_tps['rolling_std'] = df_tps['tps'].rolling(window=window).std()

first_mean = df_tps['rolling_mean'].iloc[window:len(df_tps)//2].mean()
second_mean = df_tps['rolling_mean'].iloc[len(df_tps)//2:].mean()
first_std = df_tps['rolling_std'].iloc[window:len(df_tps)//2].mean()
second_std = df_tps['rolling_std'].iloc[len(df_tps)//2:].mean()

print(f"  • First half mean:  {first_mean:.2f}")
print(f"  • Second half mean: {second_mean:.2f}")
print(f"  • Mean change:      {abs(second_mean - first_mean):.2f} ({abs(second_mean - first_mean)/first_mean*100:.1f}%)")
print(f"  • First half std:   {first_std:.2f}")
print(f"  • Second half std:  {second_std:.2f}")
print(f"  • Std change:       {abs(second_std - first_std):.2f} ({abs(second_std - first_std)/first_std*100:.1f}%)")

if abs(second_mean - first_mean)/first_mean < 0.1 and abs(second_std - first_std)/first_std < 0.2:
    print(f"\n  ✓ Mean and variance are relatively stable")
else:
    print(f"\n  ⚠️  Significant changes in mean or variance detected")

# ============================================
# PART 6: AUTOCORRELATION ANALYSIS
# ============================================
print("\n" + "="*80)
print("PART 6: AUTOCORRELATION ANALYSIS")
print("="*80)

print("\n[6.1] Autocorrelation at Key Lags:")
key_lags = [1, 5, 10, 15, 30, 60, 120, 1440]
acf_values = acf(df_tps['tps'].dropna(), nlags=max(key_lags))

for lag in key_lags:
    if lag < len(acf_values):
        lag_name = f"{lag} min"
        if lag >= 60:
            lag_name = f"{lag//60} hr"
        if lag >= 1440:
            lag_name = f"{lag//1440} day"
        print(f"  • Lag {lag_name:8s}: {acf_values[lag]:.4f}")

# Determine optimal lookback
significant_lags = [i for i, val in enumerate(acf_values[:100]) if abs(val) > 0.2]
if significant_lags:
    print(f"\n  📊 Significant autocorrelation (|ACF| > 0.2) found up to lag {max(significant_lags)}")
    print(f"  💡 Recommended lookback window: {max(significant_lags)} to {max(significant_lags)*2} minutes")
else:
    print(f"\n  ⚠️  Low autocorrelation - data may be random/noisy")

# ============================================
# PART 7: OUTLIER DETECTION
# ============================================
print("\n" + "="*80)
print("PART 7: OUTLIER & ANOMALY DETECTION")
print("="*80)

print("\n[7.1] Statistical Outliers (IQR Method):")
Q1 = df_tps['tps'].quantile(0.25)
Q3 = df_tps['tps'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 3 * IQR
upper_bound = Q3 + 3 * IQR

outliers = df_tps[(df_tps['tps'] < lower_bound) | (df_tps['tps'] > upper_bound)]

print(f"  • Q1 (25th percentile): {Q1:.2f}")
print(f"  • Q3 (75th percentile): {Q3:.2f}")
print(f"  • IQR: {IQR:.2f}")
print(f"  • Lower bound: {lower_bound:.2f}")
print(f"  • Upper bound: {upper_bound:.2f}")
print(f"  • Outliers found: {len(outliers)} ({len(outliers)/len(df_tps)*100:.2f}%)")

if len(outliers) > 0:
    print(f"\n  Outlier Statistics:")
    print(f"  • Min outlier: {outliers['tps'].min():.0f}")
    print(f"  • Max outlier: {outliers['tps'].max():.0f}")
    print(f"  • Mean outlier: {outliers['tps'].mean():.0f}")

print("\n[7.2] Z-Score Method:")
z_scores = np.abs(stats.zscore(df_tps['tps']))
z_outliers = df_tps[z_scores > 3]
print(f"  • Outliers (|z| > 3): {len(z_outliers)} ({len(z_outliers)/len(df_tps)*100:.2f}%)")

# ============================================
# PART 8: DISTRIBUTION ANALYSIS
# ============================================
print("\n" + "="*80)
print("PART 8: DISTRIBUTION ANALYSIS")
print("="*80)

print("\n[8.1] Distribution Shape:")
print(f"  • Skewness: {df_tps['tps'].skew():.4f}")
if abs(df_tps['tps'].skew()) < 0.5:
    print(f"    → Distribution is fairly symmetric")
elif df_tps['tps'].skew() > 0:
    print(f"    → Distribution is right-skewed (long tail on right)")
else:
    print(f"    → Distribution is left-skewed (long tail on left)")

print(f"  • Kurtosis: {df_tps['tps'].kurtosis():.4f}")
if abs(df_tps['tps'].kurtosis()) < 0.5:
    print(f"    → Distribution is mesokurtic (normal-like)")
elif df_tps['tps'].kurtosis() > 0:
    print(f"    → Distribution is leptokurtic (heavy-tailed, peaked)")
else:
    print(f"    → Distribution is platykurtic (light-tailed, flat)")

print("\n[8.2] Normality Test (Shapiro-Wilk):")
# Sample 5000 points for computational efficiency
sample = df_tps['tps'].sample(min(5000, len(df_tps)), random_state=42)
stat, p_value = stats.shapiro(sample)
print(f"  • Test statistic: {stat:.4f}")
print(f"  • p-value: {p_value:.6f}")
if p_value > 0.05:
    print(f"  ✓ Data appears normally distributed (p > 0.05)")
else:
    print(f"  ⚠️  Data is not normally distributed (p < 0.05)")
    print(f"    → This is common for real-world data")
    print(f"    → Neural networks don't require normality")

# ============================================
# PART 9: PREPROCESSING RECOMMENDATIONS
# ============================================
print("\n" + "="*80)
print("PART 9: PREPROCESSING RECOMMENDATIONS")
print("="*80)

print("\n📋 Based on the analysis above, here are recommendations:\n")

# Recommendation 1: Normalization
if df_tps['tps'].std() / df_tps['tps'].mean() > 0.3:
    print("✅ 1. NORMALIZATION:")
    print("   → High variance detected (CV > 0.3)")
    print("   → Recommend: MinMaxScaler or StandardScaler")
    print("   → Benefit: Faster neural network convergence")
    print("   → CRITICAL: Fit on train only, transform test!")
else:
    print("ℹ️  1. NORMALIZATION:")
    print("   → Low variance (CV < 0.3)")
    print("   → Optional: May not significantly improve results")

# Recommendation 2: Differencing
if adf_result[1] >= 0.05:
    print("\n✅ 2. DIFFERENCING:")
    print("   → Data is non-stationary (ADF p-value >= 0.05)")
    print("   → Recommend: First-order differencing for ARIMA")
    print("   → Note: RNN/LSTM can handle non-stationary data directly")
else:
    print("\nℹ️  2. DIFFERENCING:")
    print("   → Data is stationary")
    print("   → Not required for ARIMA")

# Recommendation 3: Outlier Handling
outlier_pct = len(outliers) / len(df_tps) * 100
if outlier_pct > 2:
    print(f"\n⚠️  3. OUTLIER HANDLING:")
    print(f"   → {outlier_pct:.1f}% outliers detected")
    print(f"   → Options:")
    print(f"     a) Remove outliers (if data errors)")
    print(f"     b) Cap at percentiles (e.g., 1st/99th)")
    print(f"     c) Use robust models (tree-based)")
    print(f"     d) Keep if valid (neural networks handle outliers)")
else:
    print(f"\n✅ 3. OUTLIER HANDLING:")
    print(f"   → Low outlier percentage ({outlier_pct:.1f}%)")
    print(f"   → No special handling needed")

# Recommendation 4: Feature Engineering
if hourly_range / hourly_stats['mean'].mean() > 0.2 or daily_range / daily_stats['mean'].mean() > 0.1:
    print(f"\n✅ 4. FEATURE ENGINEERING:")
    print(f"   → Strong temporal patterns detected")
    print(f"   → Recommend adding:")
    print(f"     • Hour of day (cyclical encoding: sin/cos)")
    print(f"     • Day of week (cyclical encoding: sin/cos)")
    print(f"     • Is weekend (binary flag)")
    print(f"     • Rolling statistics (mean, std)")
else:
    print(f"\nℹ️  4. FEATURE ENGINEERING:")
    print(f"   → Weak temporal patterns")
    print(f"   → May not significantly improve performance")

# Recommendation 5: Lookback Window
print(f"\n✅ 5. LOOKBACK WINDOW:")
if significant_lags:
    recommended_lookback = max(significant_lags)
    print(f"   → Based on autocorrelation analysis")
    print(f"   → Minimum: {recommended_lookback} minutes")
    print(f"   → Recommended: {recommended_lookback} to {recommended_lookback*2} minutes")
    print(f"   → Maximum: {min(recommended_lookback*3, 1440)} minutes (don't overfit)")
else:
    print(f"   → Low autocorrelation detected")
    print(f"   → Start with: 10-30 minutes")
    print(f"   → Experiment to find optimal")

# Recommendation 6: Train/Test Split
print(f"\n✅ 6. TRAIN/TEST SPLIT:")
print(f"   → Time series requires temporal split")
print(f"   → Recommend: 80/20 or 90/10 split")
print(f"   → NEVER shuffle data!")
print(f"   → Validate on recent data (walk-forward validation)")

# ============================================
# PART 10: SAVE PROCESSED DATA
# ============================================
print("\n" + "="*80)
print("PART 10: SAVE PROCESSED DATA")
print("="*80)

# Save cleaned TPS data
output_path = 'data/processed_tps_data.csv'
df_tps.to_csv(output_path, index=False)
print(f"\n✓ Saved processed data to: {output_path}")
print(f"  • Includes: timestamp, tps, hour, day_of_week, day_name, is_weekend")

# Save summary statistics
summary_stats = {
    'total_records': len(df_raw),
    'total_minutes': len(df_tps),
    'mean_tps': df_tps['tps'].mean(),
    'std_tps': df_tps['tps'].std(),
    'min_tps': df_tps['tps'].min(),
    'max_tps': df_tps['tps'].max(),
    'zero_minutes': zero_minutes,
    'outliers_pct': outlier_pct,
    'adf_pvalue': adf_result[1],
    'is_stationary': adf_result[1] < 0.05,
    'peak_hour': peak_hour,
    'low_hour': low_hour,
    'recommended_lookback': max(significant_lags) if significant_lags else 10
}

summary_df = pd.DataFrame([summary_stats])
summary_df.to_csv('data/eda_summary.csv', index=False)
print(f"✓ Saved EDA summary to: data/eda_summary.csv")

print("\n" + "="*80)
print("✓ EDA COMPLETE!")
print("="*80)
print("\nNext steps:")
print("1. Review the analysis above")
print("2. Run visualizations (see eda_visualizations.py)")
print("3. Apply preprocessing based on recommendations")
print("4. Train models with informed hyperparameters")
print("="*80)