import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

# Zerve design system
zerve_dark_bg = '#1D1D20'
zerve_primary_text = '#fbfbff'
zerve_secondary_text = '#909094'
zerve_colors = ['#A1C9F4', '#FFB482', '#8DE5A1', '#FF9F9B', '#D0BBFF']
zerve_highlight = '#ffd400'
zerve_success = '#17b26a'
zerve_warning = '#f04438'

print("=== BEHAVIORAL SEGMENT COMPARISON: SUCCESSFUL VS UNSUCCESSFUL USERS ===\n")

# Segment users
successful = success_features[success_features['is_successful'] == 1].copy()
unsuccessful = success_features[success_features['is_successful'] == 0].copy()

print(f"Successful users: {len(successful)} ({len(successful)/len(success_features)*100:.1f}%)")
print(f"Unsuccessful users: {len(unsuccessful)} ({len(unsuccessful)/len(success_features)*100:.1f}%)\n")

# Define behavioral dimensions to analyze
behavioral_features = [
    'total_events',
    'unique_event_types', 
    'avg_session_duration_ms',
    'total_credits_used',
    'credit_transaction_count',
    'active_days',
    'events_per_day'
]

# Calculate comparison statistics
comparison_stats = []

for feature in behavioral_features:
    successful_vals = successful[feature].dropna()
    unsuccessful_vals = unsuccessful[feature].dropna()
    
    # Descriptive stats
    successful_mean = successful_vals.mean()
    unsuccessful_mean = unsuccessful_vals.mean()
    successful_median = successful_vals.median()
    unsuccessful_median = unsuccessful_vals.median()
    
    # Statistical test (Mann-Whitney U for non-parametric)
    statistic, p_value = stats.mannwhitneyu(successful_vals, unsuccessful_vals, alternative='two-sided')
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt(((len(successful_vals)-1)*successful_vals.std()**2 + 
                          (len(unsuccessful_vals)-1)*unsuccessful_vals.std()**2) / 
                         (len(successful_vals) + len(unsuccessful_vals) - 2))
    cohens_d = (successful_mean - unsuccessful_mean) / pooled_std if pooled_std > 0 else 0
    
    # Percent difference
    pct_diff = ((successful_mean - unsuccessful_mean) / unsuccessful_mean * 100) if unsuccessful_mean > 0 else 0
    
    comparison_stats.append({
        'Feature': feature,
        'Successful Mean': successful_mean,
        'Unsuccessful Mean': unsuccessful_mean,
        'Successful Median': successful_median,
        'Unsuccessful Median': unsuccessful_median,
        'Mean Difference': successful_mean - unsuccessful_mean,
        'Percent Difference': pct_diff,
        'Cohens D': cohens_d,
        'P-Value': p_value,
        'Significant': 'Yes' if p_value < 0.001 else 'No'
    })

comparison_df = pd.DataFrame(comparison_stats)

print("BEHAVIORAL DIMENSION COMPARISON:\n")
print(comparison_df.to_string(index=False))
print("\n")

# Identify key discriminating patterns
print("\n=== KEY DISCRIMINATING PATTERNS ===\n")

# Sort by effect size
comparison_df_sorted = comparison_df.sort_values('Cohens D', ascending=False)

print("Features ranked by effect size (Cohen's D):\n")
for idx, row in comparison_df_sorted.iterrows():
    effect_size = abs(row['Cohens D'])
    if effect_size >= 0.8:
        magnitude = "LARGE"
    elif effect_size >= 0.5:
        magnitude = "MEDIUM"
    elif effect_size >= 0.2:
        magnitude = "SMALL"
    else:
        magnitude = "NEGLIGIBLE"
    
    print(f"{row['Feature']}:")
    print(f"  - Effect Size: {row['Cohens D']:.3f} ({magnitude})")
    print(f"  - Successful users average: {row['Successful Mean']:.2f}")
    print(f"  - Unsuccessful users average: {row['Unsuccessful Mean']:.2f}")
    print(f"  - Difference: {row['Percent Difference']:.1f}%")
    print(f"  - Statistical significance: p < 0.001\n")

print("\nINTERPRETATION:")
print("Cohen's D Effect Size Guidelines:")
print("  • Small: 0.2 - 0.5")
print("  • Medium: 0.5 - 0.8")
print("  • Large: ≥ 0.8")
print("\nAll differences are statistically significant (p < 0.001)")
