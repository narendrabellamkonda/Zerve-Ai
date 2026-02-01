import pandas as pd

# Load the data
df = pd.read_csv("zerve_hackathon_for_reviewc8fa7c7.csv")

# Convert timestamp to datetime using ISO8601 format
df['timestamp'] = pd.to_datetime(df['timestamp'], format='ISO8601')
df['created_at'] = pd.to_datetime(df['created_at'], format='ISO8601')

# Feature 1: Event count per user
event_counts = df.groupby('person_id').size().reset_index(name='total_events')

# Feature 2: Unique event types per user
unique_events = df.groupby('person_id')['event'].nunique().reset_index(name='unique_event_types')

# Feature 3: Session engagement - average session duration per user
session_duration = df.groupby(['person_id', 'prop_$session_id'])['prop_$sdk_debug_current_session_duration'].max().reset_index()
avg_session_duration = session_duration.groupby('person_id')['prop_$sdk_debug_current_session_duration'].mean().reset_index(name='avg_session_duration_ms')

# Feature 4: Credits usage metrics
credits_used = df.groupby('person_id')['prop_credits_used'].sum().reset_index(name='total_credits_used')
credit_transactions = df[df['prop_credits_used'].notna()].groupby('person_id').size().reset_index(name='credit_transaction_count')

# Feature 5: Active days per user
df['date'] = df['timestamp'].dt.date
active_days = df.groupby('person_id')['date'].nunique().reset_index(name='active_days')

# Feature 6: Events per active day
events_per_day = event_counts.merge(active_days, on='person_id')
events_per_day['events_per_day'] = events_per_day['total_events'] / events_per_day['active_days']

# Merge all features together
engineered_features = event_counts.merge(unique_events, on='person_id', how='left')
engineered_features = engineered_features.merge(avg_session_duration, on='person_id', how='left')
engineered_features = engineered_features.merge(credits_used, on='person_id', how='left')
engineered_features = engineered_features.merge(credit_transactions, on='person_id', how='left')
engineered_features = engineered_features.merge(active_days, on='person_id', how='left')
engineered_features = engineered_features.merge(events_per_day[['person_id', 'events_per_day']], on='person_id', how='left')

# Fill NaN values with 0
engineered_features = engineered_features.fillna(0)

print(f"Engineered features for {len(engineered_features)} unique users")
print(f"\nFeature columns: {list(engineered_features.columns)}")
print(f"\nSummary statistics:")
print(engineered_features.describe())
print(f"\nSample of engineered features:")
print(engineered_features.head(10))
