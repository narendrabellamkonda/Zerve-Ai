import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Zerve design system
zerve_dark_bg = '#1D1D20'
zerve_primary_text = '#fbfbff'
zerve_secondary_text = '#909094'
zerve_colors = ['#A1C9F4', '#FFB482', '#8DE5A1', '#FF9F9B', '#D0BBFF']
zerve_highlight = '#ffd400'
zerve_success = '#17b26a'
zerve_warning = '#f04438'

print("=== PREDICTIVE MODEL: IDENTIFYING BEHAVIORS PREDICTIVE OF LONG-TERM SUCCESS ===\n")

# Prepare data for modeling
model_data = success_features.copy()

# Feature set: behavioral metrics (exclude target and derived success metrics)
feature_cols = [
    'total_events',
    'unique_event_types', 
    'avg_session_duration_ms',
    'total_credits_used',
    'credit_transaction_count',
    'active_days',
    'events_per_day'
]

X = model_data[feature_cols].fillna(0)
y = model_data['is_successful']

print(f"Dataset: {len(X)} users")
print(f"Features: {len(feature_cols)}")
print(f"Target distribution: {y.value_counts().to_dict()}")
print(f"Class balance: {y.mean()*100:.1f}% successful\n")

# Standardize features for better model performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols, index=X.index)

# Cross-validation setup
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("=== MODEL 1: GRADIENT BOOSTING CLASSIFIER ===\n")

# Gradient Boosting with tuned hyperparameters
gb_model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    min_samples_split=20,
    min_samples_leaf=10,
    subsample=0.8,
    random_state=42
)

# Cross-validation
scoring = ['roc_auc', 'accuracy', 'precision', 'recall', 'f1']
gb_cv_results = cross_validate(gb_model, X_scaled, y, cv=cv, scoring=scoring, n_jobs=-1)

print("Cross-Validation Results (5-fold):")
print(f"  ROC AUC: {gb_cv_results['test_roc_auc'].mean():.4f} (± {gb_cv_results['test_roc_auc'].std():.4f})")
print(f"  Accuracy: {gb_cv_results['test_accuracy'].mean():.4f} (± {gb_cv_results['test_accuracy'].std():.4f})")
print(f"  Precision: {gb_cv_results['test_precision'].mean():.4f} (± {gb_cv_results['test_precision'].std():.4f})")
print(f"  Recall: {gb_cv_results['test_recall'].mean():.4f} (± {gb_cv_results['test_recall'].std():.4f})")
print(f"  F1 Score: {gb_cv_results['test_f1'].mean():.4f} (± {gb_cv_results['test_f1'].std():.4f})")

# Train on full dataset for feature importance
gb_model.fit(X_scaled, y)

# Feature importance
gb_feature_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': gb_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nFeature Importance (Gradient Boosting):")
print(gb_feature_importance.to_string(index=False))

print("\n" + "="*70)
print("=== MODEL 2: RANDOM FOREST CLASSIFIER ===\n")

# Random Forest with tuned hyperparameters
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=20,
    min_samples_leaf=10,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)

# Cross-validation
rf_cv_results = cross_validate(rf_model, X_scaled, y, cv=cv, scoring=scoring, n_jobs=-1)

print("Cross-Validation Results (5-fold):")
print(f"  ROC AUC: {rf_cv_results['test_roc_auc'].mean():.4f} (± {rf_cv_results['test_roc_auc'].std():.4f})")
print(f"  Accuracy: {rf_cv_results['test_accuracy'].mean():.4f} (± {rf_cv_results['test_accuracy'].std():.4f})")
print(f"  Precision: {rf_cv_results['test_precision'].mean():.4f} (± {rf_cv_results['test_precision'].std():.4f})")
print(f"  Recall: {rf_cv_results['test_recall'].mean():.4f} (± {rf_cv_results['test_recall'].std():.4f})")
print(f"  F1 Score: {rf_cv_results['test_f1'].mean():.4f} (± {rf_cv_results['test_f1'].std():.4f})")

# Train on full dataset for feature importance
rf_model.fit(X_scaled, y)

# Feature importance
rf_feature_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nFeature Importance (Random Forest):")
print(rf_feature_importance.to_string(index=False))

# Compare models
print("\n" + "="*70)
print("=== MODEL COMPARISON ===\n")

comparison_metrics = pd.DataFrame({
    'Metric': ['ROC AUC', 'Accuracy', 'Precision', 'Recall', 'F1'],
    'Gradient Boosting': [
        gb_cv_results['test_roc_auc'].mean(),
        gb_cv_results['test_accuracy'].mean(),
        gb_cv_results['test_precision'].mean(),
        gb_cv_results['test_recall'].mean(),
        gb_cv_results['test_f1'].mean()
    ],
    'Random Forest': [
        rf_cv_results['test_roc_auc'].mean(),
        rf_cv_results['test_accuracy'].mean(),
        rf_cv_results['test_precision'].mean(),
        rf_cv_results['test_recall'].mean(),
        rf_cv_results['test_f1'].mean()
    ]
})

print(comparison_metrics.to_string(index=False))

# Determine best model
best_model_name = 'Gradient Boosting' if gb_cv_results['test_roc_auc'].mean() > rf_cv_results['test_roc_auc'].mean() else 'Random Forest'
best_model = gb_model if best_model_name == 'Gradient Boosting' else rf_model
best_auc = max(gb_cv_results['test_roc_auc'].mean(), rf_cv_results['test_roc_auc'].mean())

print(f"\n✓ Best Model: {best_model_name} (ROC AUC: {best_auc:.4f})")
print(f"✓ Success Criteria Met: AUC = {best_auc:.4f} {'> 0.70 ✓' if best_auc > 0.70 else '< 0.70 ✗'}")

print("\n" + "="*70)
print("=== INTERPRETABILITY: TOP PREDICTIVE BEHAVIORS ===\n")

# Combine feature importances from both models
combined_importance = pd.DataFrame({
    'Feature': feature_cols,
    'GB_Importance': gb_model.feature_importances_,
    'RF_Importance': rf_model.feature_importances_,
    'Average_Importance': (gb_model.feature_importances_ + rf_model.feature_importances_) / 2
}).sort_values('Average_Importance', ascending=False)

print("Ranked Predictive Features (Average across both models):")
print(combined_importance.to_string(index=False))

print("\n" + "="*70)
print("=== FEATURE IMPACT DIRECTIONS ===\n")

# Calculate correlations to understand direction of impact
impact_directions = []
for feature in feature_cols:
    corr = model_data[feature].corr(model_data['is_successful'])
    direction = '↑ Positive' if corr > 0 else '↓ Negative'
    impact_directions.append({
        'Feature': feature,
        'Correlation': corr,
        'Direction': direction,
        'Interpretation': f"Higher {feature} → {'Higher' if corr > 0 else 'Lower'} success probability"
    })

impact_df = pd.DataFrame(impact_directions).sort_values('Correlation', ascending=False, key=abs)

print("Feature Impact Directions:")
for _, row in impact_df.iterrows():
    print(f"  {row['Feature']}:")
    print(f"    - Correlation: {row['Correlation']:.4f}")
    print(f"    - {row['Interpretation']}\n")
