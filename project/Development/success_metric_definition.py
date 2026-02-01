import pandas as pd
import numpy as np

# Define "long-term success" based on multiple indicators
print("=== DEFINING LONG-TERM SUCCESS METRICS ===\n")

success_features = engineered_features.copy()

# PRIMARY SUCCESS METRIC: Binary classification based on multiple criteria
# A successful user demonstrates SUSTAINED engagement (not just one-time use)

print("Success Criteria Components:")
print("1. Retention: Active for 2+ days (shows return behavior)")
print("2. Engagement Depth: Used 3+ different event types (broad feature adoption)")
print("3. Activity Volume: 5+ total events (meaningful interaction)")
print("4. Session Quality: Average session > 2 minutes (engaged sessions)")
print("\n")

# Calculate each success indicator
success_features['has_retention'] = (success_features['active_days'] >= 2).astype(int)
success_features['has_engagement_depth'] = (success_features['unique_event_types'] >= 3).astype(int)
success_features['has_activity_volume'] = (success_features['total_events'] >= 5).astype(int)
success_features['has_session_quality'] = (success_features['avg_session_duration_ms'] >= 120000).astype(int)  # 2 min in ms

# BINARY SUCCESS: User meets at least 3 out of 4 criteria
success_features['success_score'] = (
    success_features['has_retention'] +
    success_features['has_engagement_depth'] +
    success_features['has_activity_volume'] +
    success_features['has_session_quality']
)

success_features['is_successful'] = (success_features['success_score'] >= 3).astype(int)

print("Binary Success Metric Distribution:")
print(success_features['is_successful'].value_counts())
print(f"\nSuccess rate: {success_features['is_successful'].mean() * 100:.2f}%")

# CONTINUOUS SUCCESS METRIC: Weighted score (0-100 scale)
# This provides more granular measurement of success

# Normalize each component to 0-1 scale
def normalize_score(series, cap_percentile=95):
    """Normalize to 0-1, capping outliers at 95th percentile"""
    cap_value = series.quantile(cap_percentile / 100)
    capped = series.clip(upper=cap_value)
    return capped / cap_value if cap_value > 0 else capped

# Component scores (0-1 scale)
success_features['retention_score'] = normalize_score(success_features['active_days'])
success_features['engagement_score'] = normalize_score(success_features['unique_event_types'])
success_features['activity_score'] = normalize_score(success_features['total_events'])
success_features['session_score'] = normalize_score(success_features['avg_session_duration_ms'])
success_features['credit_score'] = normalize_score(success_features['total_credits_used'])

# Weighted composite success score (0-100)
# Higher weights for retention and engagement as they indicate long-term value
success_features['success_score_continuous'] = (
    success_features['retention_score'] * 0.30 +      # 30% - Most important
    success_features['engagement_score'] * 0.25 +     # 25% - Feature breadth
    success_features['activity_score'] * 0.20 +       # 20% - Volume
    success_features['session_score'] * 0.15 +        # 15% - Quality
    success_features['credit_score'] * 0.10            # 10% - Value creation
) * 100

print("\n\nContinuous Success Score Distribution:")
print(success_features['success_score_continuous'].describe())

# Success tiers based on continuous score
success_features['success_tier'] = pd.cut(
    success_features['success_score_continuous'],
    bins=[0, 20, 40, 60, 80, 100],
    labels=['At Risk', 'Low', 'Medium', 'High', 'Very High']
)

print("\nSuccess Tier Distribution:")
print(success_features['success_tier'].value_counts().sort_index())
print(f"\nPercentage breakdown:")
print((success_features['success_tier'].value_counts() / len(success_features) * 100).round(2).sort_index())

# Summary statistics by success status
print("\n\n=== SUCCESS METRIC VALIDATION ===\n")
print("Average metrics by binary success status:\n")
comparison = success_features.groupby('is_successful')[
    ['active_days', 'unique_event_types', 'total_events', 
     'avg_session_duration_ms', 'total_credits_used']
].mean()
comparison.columns = ['Active Days', 'Unique Events', 'Total Events', 'Avg Session (ms)', 'Credits Used']
print(comparison.round(2))

print(f"\n\nFinal Dataset: {len(success_features)} users with success metrics defined")
print(f"Success columns added: is_successful (binary), success_score_continuous (0-100)")
