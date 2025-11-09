"""
EDA Visualizations for Transaction Data
Creates comprehensive plots to understand data patterns
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
sns.set_palette("husl")

print("=" * 80)
print("GENERATING EDA VISUALIZATIONS")
print("=" * 80)

# Load processed data
df_tps = pd.read_csv('data/processed_tps_data.csv')
df_tps['timestamp'] = pd.to_datetime(df_tps['timestamp'])

print(f"\n✓ Loaded {len(df_tps):,} time points")

# Create comprehensive figure
fig = plt.figure(figsize=(20, 24))
fig.suptitle('Transaction Load Forecasting - Exploratory Data Analysis', 
             fontsize=20, fontweight='bold', y=0.995)

# ============================================
# 1. Time Series Overview
# ============================================
ax1 = plt.subplot(6, 3, 1)
sample_ratio = max(1, len(df_tps) // 10000)  # Sample for visibility
df_sample = df_tps.iloc[::sample_ratio]
ax1.plot(df_sample['timestamp'], df_sample['tps'], alpha=0.7, linewidth=0.5)
ax1.set_title('1. TPS Over Time (Full Dataset)', fontweight='bold')
ax1.set_xlabel('Time')
ax1.set_ylabel('Transactions per Minute')
ax1.grid(True, alpha=0.3)

# ============================================
# 2. Distribution Histogram
# ============================================
ax2 = plt.subplot(6, 3, 2)
ax2.hist(df_tps['tps'], bins=50, edgecolor='black', alpha=0.7)
ax2.axvline(df_tps['tps'].mean(), color='red', linestyle='--', linewidth=2, 
           label=f'Mean: {df_tps["tps"].mean():.1f}')
ax2.axvline(df_tps['tps'].median(), color='green', linestyle='--', linewidth=2,
           label=f'Median: {df_tps["tps"].median():.1f}')
ax2.set_title('2. TPS Distribution', fontweight='bold')
ax2.set_xlabel('Transactions per Minute')
ax2.set_ylabel('Frequency')
ax2.legend()
ax2.grid(True, alpha=0.3)

# ============================================
# 3. Box Plot
# ============================================
ax3 = plt.subplot(6, 3, 3)
box_data = ax3.boxplot(df_tps['tps'], vert=True, patch_artist=True)
box_data['boxes'][0].set_facecolor('lightblue')
ax3.set_title('3. TPS Box Plot', fontweight='bold')
ax3.set_ylabel('Transactions per Minute')
ax3.grid(True, alpha=0.3)

# ============================================
# 4. Hourly Pattern (Bar Chart)
# ============================================
ax4 = plt.subplot(6, 3, 4)
hourly_avg = df_tps.groupby('hour')['tps'].mean()
bars = ax4.bar(hourly_avg.index, hourly_avg.values, color='skyblue', edgecolor='black')
# Highlight peak and low hours
peak_idx = hourly_avg.idxmax()
low_idx = hourly_avg.idxmin()
bars[peak_idx].set_color('red')
bars[low_idx].set_color('green')
ax4.set_title('4. Average TPS by Hour', fontweight='bold')
ax4.set_xlabel('Hour of Day')
ax4.set_ylabel('Average TPS')
ax4.set_xticks(range(0, 24, 2))
ax4.grid(True, alpha=0.3, axis='y')

# ============================================
# 5. Hourly Pattern (Line with Std)
# ============================================
ax5 = plt.subplot(6, 3, 5)
hourly_stats = df_tps.groupby('hour')['tps'].agg(['mean', 'std'])
ax5.plot(hourly_stats.index, hourly_stats['mean'], marker='o', linewidth=2, label='Mean')
ax5.fill_between(hourly_stats.index, 
                 hourly_stats['mean'] - hourly_stats['std'],
                 hourly_stats['mean'] + hourly_stats['std'],
                 alpha=0.3, label='±1 Std Dev')
ax5.set_title('5. Hourly Pattern with Variability', fontweight='bold')
ax5.set_xlabel('Hour of Day')
ax5.set_ylabel('TPS')
ax5.legend()
ax5.grid(True, alpha=0.3)

# ============================================
# 6. Daily Pattern
# ============================================
ax6 = plt.subplot(6, 3, 6)
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
daily_avg = df_tps.groupby('day_name')['tps'].mean().reindex(day_order)
colors = ['lightcoral' if day in ['Saturday', 'Sunday'] else 'lightblue' for day in day_order]
ax6.bar(range(len(day_order)), daily_avg.values, color=colors, edgecolor='black')
ax6.set_title('6. Average TPS by Day of Week', fontweight='bold')
ax6.set_xlabel('Day')
ax6.set_ylabel('Average TPS')
ax6.set_xticks(range(len(day_order)))
ax6.set_xticklabels([d[:3] for d in day_order], rotation=45)
ax6.grid(True, alpha=0.3, axis='y')

# ============================================
# 7. Heatmap: Hour x Day
# ============================================
ax7 = plt.subplot(6, 3, 7)
pivot_table = df_tps.pivot_table(values='tps', index='hour', columns='day_of_week', aggfunc='mean')
pivot_table.columns = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
sns.heatmap(pivot_table, cmap='YlOrRd', ax=ax7, cbar_kws={'label': 'Avg TPS'}, 
           fmt='.0f', annot=False)
ax7.set_title('7. TPS Heatmap: Hour × Day', fontweight='bold')
ax7.set_xlabel('Day of Week')
ax7.set_ylabel('Hour of Day')

# ============================================
# 8. Rolling Mean & Std (1 day window)
# ============================================
ax8 = plt.subplot(6, 3, 8)
window = 1440  # 1 day
df_tps['rolling_mean'] = df_tps['tps'].rolling(window=window).mean()
df_tps['rolling_std'] = df_tps['tps'].rolling(window=window).std()
sample = df_tps.iloc[::sample_ratio]
ax8.plot(sample['timestamp'], sample['tps'], alpha=0.3, linewidth=0.5, label='TPS')
ax8.plot(sample['timestamp'], sample['rolling_mean'], color='red', 
        linewidth=2, label=f'{window}min Rolling Mean')
ax8.fill_between(sample['timestamp'],
                sample['rolling_mean'] - sample['rolling_std'],
                sample['rolling_mean'] + sample['rolling_std'],
                alpha=0.2, color='red')
ax8.set_title('8. TPS with Rolling Statistics', fontweight='bold')
ax8.set_xlabel('Time')
ax8.set_ylabel('TPS')
ax8.legend()
ax8.grid(True, alpha=0.3)

# ============================================
# 9. Decomposition (Trend + Seasonality)
# ============================================
ax9 = plt.subplot(6, 3, 9)
# Simple moving average as trend
df_tps['trend'] = df_tps['tps'].rolling(window=1440, center=True).mean()
df_tps['detrended'] = df_tps['tps'] - df_tps['trend']
sample = df_tps.iloc[::sample_ratio]
ax9.plot(sample['timestamp'], sample['tps'], alpha=0.5, linewidth=0.5, label='Original')
ax9.plot(sample['timestamp'], sample['trend'], color='red', linewidth=2, label='Trend')
ax9.set_title('9. Trend Decomposition', fontweight='bold')
ax9.set_xlabel('Time')
ax9.set_ylabel('TPS')
ax9.legend()
ax9.grid(True, alpha=0.3)

# ============================================
# 10. ACF Plot
# ============================================
ax10 = plt.subplot(6, 3, 10)
plot_acf(df_tps['tps'].dropna(), lags=100, ax=ax10, alpha=0.05)
ax10.set_title('10. Autocorrelation Function (ACF)', fontweight='bold')
ax10.set_xlabel('Lag (minutes)')
ax10.grid(True, alpha=0.3)

# ============================================
# 11. PACF Plot
# ============================================
ax11 = plt.subplot(6, 3, 11)
plot_pacf(df_tps['tps'].dropna(), lags=50, ax=ax11, alpha=0.05, method='ywm')
ax11.set_title('11. Partial Autocorrelation (PACF)', fontweight='bold')
ax11.set_xlabel('Lag (minutes)')
ax11.grid(True, alpha=0.3)

# ============================================
# 12. Lag Plot (t vs t-1)
# ============================================
ax12 = plt.subplot(6, 3, 12)
sample_size = min(5000, len(df_tps))
sample_indices = np.random.choice(len(df_tps)-1, sample_size, replace=False)
tps_t = df_tps['tps'].iloc[sample_indices].values
tps_t1 = df_tps['tps'].iloc[sample_indices + 1].values
ax12.scatter(tps_t, tps_t1, alpha=0.3, s=1)
ax12.plot([tps_t.min(), tps_t.max()], [tps_t.min(), tps_t.max()], 
         'r--', linewidth=2, label='y=x')
ax12.set_title('12. Lag Plot (t vs t+1)', fontweight='bold')
ax12.set_xlabel('TPS at time t')
ax12.set_ylabel('TPS at time t+1')
ax12.legend()
ax12.grid(True, alpha=0.3)

# ============================================
# 13. Q-Q Plot (Normality Check)
# ============================================
ax13 = plt.subplot(6, 3, 13)
stats.probplot(df_tps['tps'].sample(min(5000, len(df_tps)), random_state=42), 
              dist="norm", plot=ax13)
ax13.set_title('13. Q-Q Plot (Normality Check)', fontweight='bold')
ax13.grid(True, alpha=0.3)

# ============================================
# 14. Outliers Visualization
# ============================================
ax14 = plt.subplot(6, 3, 14)
Q1 = df_tps['tps'].quantile(0.25)
Q3 = df_tps['tps'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 3 * IQR
upper_bound = Q3 + 3 * IQR
outliers = df_tps[(df_tps['tps'] < lower_bound) | (df_tps['tps'] > upper_bound)]
sample = df_tps.iloc[::sample_ratio]
ax14.scatter(sample['timestamp'], sample['tps'], alpha=0.3, s=1, label='Normal', color='blue')
if len(outliers) > 0:
    ax14.scatter(outliers['timestamp'], outliers['tps'], alpha=0.8, s=20, 
                label='Outliers', color='red', marker='x')
ax14.axhline(y=upper_bound, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax14.axhline(y=lower_bound, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax14.set_title('14. Outlier Detection (IQR Method)', fontweight='bold')
ax14.set_xlabel('Time')
ax14.set_ylabel('TPS')
ax14.legend()
ax14.grid(True, alpha=0.3)

# ============================================
# 15. Weekday vs Weekend Distribution
# ============================================
ax15 = plt.subplot(6, 3, 15)
weekday_tps = df_tps[~df_tps['is_weekend']]['tps']
weekend_tps = df_tps[df_tps['is_weekend']]['tps']
ax15.hist([weekday_tps, weekend_tps], bins=30, label=['Weekday', 'Weekend'], 
         alpha=0.7, edgecolor='black')
ax15.set_title('15. Weekday vs Weekend Distribution', fontweight='bold')
ax15.set_xlabel('TPS')
ax15.set_ylabel('Frequency')
ax15.legend()
ax15.grid(True, alpha=0.3)

# ============================================
# 16. Cumulative Distribution
# ============================================
ax16 = plt.subplot(6, 3, 16)
sorted_tps = np.sort(df_tps['tps'])
cumulative = np.arange(1, len(sorted_tps) + 1) / len(sorted_tps)
ax16.plot(sorted_tps, cumulative, linewidth=2)
ax16.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Median')
ax16.axhline(y=0.95, color='g', linestyle='--', alpha=0.5, label='95th percentile')
ax16.set_title('16. Cumulative Distribution Function', fontweight='bold')
ax16.set_xlabel('TPS')
ax16.set_ylabel('Cumulative Probability')
ax16.legend()
ax16.grid(True, alpha=0.3)

# ============================================
# 17. Variance Over Time
# ============================================
ax17 = plt.subplot(6, 3, 17)
df_tps['rolling_var'] = df_tps['tps'].rolling(window=1440).var()
sample = df_tps.iloc[::sample_ratio]
ax17.plot(sample['timestamp'], sample['rolling_var'], linewidth=1)
ax17.set_title('17. Rolling Variance (1 day window)', fontweight='bold')
ax17.set_xlabel('Time')
ax17.set_ylabel('Variance')
ax17.grid(True, alpha=0.3)

# ============================================
# 18. Periodogram (Frequency Analysis)
# ============================================
ax18 = plt.subplot(6, 3, 18)
from scipy.signal import periodogram
frequencies, power = periodogram(df_tps['tps'].dropna(), fs=1)
# Show only meaningful frequencies
mask = (frequencies > 0) & (frequencies < 0.1)
ax18.semilogy(frequencies[mask], power[mask])
ax18.set_title('18. Periodogram (Frequency Analysis)', fontweight='bold')
ax18.set_xlabel('Frequency (cycles per minute)')
ax18.set_ylabel('Power Spectral Density')
ax18.grid(True, alpha=0.3)

plt.tight_layout()

# Save figure
output_file = 'eda_comprehensive_analysis.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\n✓ Saved comprehensive EDA visualization to: {output_file}")

# Show plot
plt.show()

print("\n" + "=" * 80)
print("✓ VISUALIZATION COMPLETE!")
print("=" * 80)
print("\nGenerated plots:")
print("1. Time series overview")
print("2. Distribution analysis")
print("3. Box plot")
print("4-7. Temporal patterns (hourly, daily, heatmap)")
print("8-9. Trend and rolling statistics")
print("10-12. Autocorrelation analysis")
print("13. Normality check")
print("14. Outlier detection")
print("15. Weekday vs weekend")
print("16. Cumulative distribution")
print("17. Variance over time")
print("18. Frequency analysis")
print("=" * 80)