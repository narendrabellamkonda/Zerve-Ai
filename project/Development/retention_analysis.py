import pandas as pd
import numpy as np

# Analyze retention and activity persistence patterns
print("=== RETENTION & PERSISTENCE ANALYSIS ===\n")

# 1. Activity Persistence: Users who come back multiple days
retention_metrics = engineered_features.copy()

# Categorize retention based on active days
retention_metrics['retention_category'] = pd.cut(
    retention_metrics['active_days'],
    bins=[0, 1, 3, 7, 30, float('inf')],
    labels=['1-day', '2-3 days', '4-7 days', '1-4 weeks', '30+ days']
)

print("User Distribution by Retention Category:")
print(retention_metrics['retention_category'].value_counts().sort_index())
print(f"\nPercentage breakdown:")
print((retention_metrics['retention_category'].value_counts() / len(retention_metrics) * 100).round(2).sort_index())

# 2. Engagement Depth: Event diversity and volume
print("\n\n=== ENGAGEMENT DEPTH ===\n")

# High engagement: multiple event types + consistent activity
retention_metrics['engagement_depth'] = pd.cut(
    retention_metrics['unique_event_types'],
    bins=[0, 2, 5, 10, float('inf')],
    labels=['Low (1-2)', 'Medium (3-5)', 'High (6-10)', 'Very High (10+)']
)

print("User Distribution by Engagement Depth:")
print(retention_metrics['engagement_depth'].value_counts().sort_index())
print(f"\nPercentage breakdown:")
print((retention_metrics['engagement_depth'].value_counts() / len(retention_metrics) * 100).round(2).sort_index())

# 3. Value Creation: Credits usage as proxy for value creation
print("\n\n=== VALUE CREATION INDICATORS ===\n")

credit_users = retention_metrics[retention_metrics['total_credits_used'] > 0]
print(f"Users who used credits: {len(credit_users)} ({len(credit_users)/len(retention_metrics)*100:.2f}%)")
print(f"Average credits used by active credit users: {credit_users['total_credits_used'].mean():.2f}")
print(f"Median credits used by active credit users: {credit_users['total_credits_used'].median():.2f}")
print(f"Max credits used: {credit_users['total_credits_used'].max():.2f}")

# 4. Session quality indicators
print("\n\n=== SESSION QUALITY ===\n")

# Convert milliseconds to minutes
retention_metrics['avg_session_duration_min'] = retention_metrics['avg_session_duration_ms'] / 60000

print("Average session duration stats (in minutes):")
print(retention_metrics['avg_session_duration_min'].describe())

# Categorize session engagement
retention_metrics['session_quality'] = pd.cut(
    retention_metrics['avg_session_duration_min'],
    bins=[0, 1, 5, 15, float('inf')],
    labels=['Brief (<1 min)', 'Short (1-5 min)', 'Medium (5-15 min)', 'Long (15+ min)']
)

print("\nSession Quality Distribution:")
print(retention_metrics['session_quality'].value_counts().sort_index())

# 5. Activity intensity
print("\n\n=== ACTIVITY INTENSITY ===\n")

print("Events per day statistics:")
print(retention_metrics['events_per_day'].describe())

# Categorize activity intensity
retention_metrics['activity_intensity'] = pd.cut(
    retention_metrics['events_per_day'],
    bins=[0, 2, 5, 10, 20, float('inf')],
    labels=['Very Low (1-2)', 'Low (3-5)', 'Medium (6-10)', 'High (11-20)', 'Very High (20+)']
)

print("\nActivity Intensity Distribution:")
print(retention_metrics['activity_intensity'].value_counts().sort_index())
