import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix

# Zerve design system
zerve_dark_bg = '#1D1D20'
zerve_primary_text = '#fbfbff'
zerve_secondary_text = '#909094'
zerve_colors = ['#A1C9F4', '#FFB482', '#8DE5A1', '#FF9F9B', '#D0BBFF']
zerve_highlight = '#ffd400'
zerve_success = '#17b26a'
zerve_warning = '#f04438'

print("=== MODEL VISUALIZATION & PERFORMANCE SUMMARY ===\n")

# 1. Feature Importance Comparison Chart
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
fig1.patch.set_facecolor(zerve_dark_bg)

# GB feature importance
ax1.set_facecolor(zerve_dark_bg)
bars1 = ax1.barh(range(len(gb_feature_importance)), 
                  gb_feature_importance['Importance'].values, 
                  color=zerve_colors[0])
ax1.set_yticks(range(len(gb_feature_importance)))
ax1.set_yticklabels(gb_feature_importance['Feature'].values, color=zerve_primary_text)
ax1.set_xlabel('Importance', color=zerve_primary_text, fontsize=12)
ax1.set_title('Gradient Boosting Feature Importance', color=zerve_primary_text, fontsize=14, fontweight='bold', pad=20)
ax1.tick_params(colors=zerve_primary_text)
ax1.spines['bottom'].set_color(zerve_secondary_text)
ax1.spines['left'].set_color(zerve_secondary_text)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.invert_yaxis()

# Add value labels
for i, (bar, val) in enumerate(zip(bars1, gb_feature_importance['Importance'].values)):
    ax1.text(val + 0.01, bar.get_y() + bar.get_height()/2, 
             f'{val:.3f}', va='center', color=zerve_primary_text, fontsize=10)

# RF feature importance
ax2.set_facecolor(zerve_dark_bg)
bars2 = ax2.barh(range(len(rf_feature_importance)), 
                  rf_feature_importance['Importance'].values, 
                  color=zerve_colors[1])
ax2.set_yticks(range(len(rf_feature_importance)))
ax2.set_yticklabels(rf_feature_importance['Feature'].values, color=zerve_primary_text)
ax2.set_xlabel('Importance', color=zerve_primary_text, fontsize=12)
ax2.set_title('Random Forest Feature Importance', color=zerve_primary_text, fontsize=14, fontweight='bold', pad=20)
ax2.tick_params(colors=zerve_primary_text)
ax2.spines['bottom'].set_color(zerve_secondary_text)
ax2.spines['left'].set_color(zerve_secondary_text)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.invert_yaxis()

# Add value labels
for i, (bar, val) in enumerate(zip(bars2, rf_feature_importance['Importance'].values)):
    ax2.text(val + 0.005, bar.get_y() + bar.get_height()/2, 
             f'{val:.3f}', va='center', color=zerve_primary_text, fontsize=10)

plt.tight_layout()
print("✓ Created feature importance comparison charts")

# 2. Composite Feature Ranking
fig2, ax3 = plt.subplots(figsize=(14, 7))
fig2.patch.set_facecolor(zerve_dark_bg)
ax3.set_facecolor(zerve_dark_bg)

# Use the final ranking from previous block
features_sorted = final_ranking.sort_values('Composite_Score', ascending=True)

colors = [zerve_success if corr > 0 else zerve_warning for corr in features_sorted['Correlation']]
bars3 = ax3.barh(range(len(features_sorted)), 
                 features_sorted['Composite_Score'].values, 
                 color=colors, alpha=0.8)

ax3.set_yticks(range(len(features_sorted)))
ax3.set_yticklabels(features_sorted['Feature'].values, color=zerve_primary_text, fontsize=12)
ax3.set_xlabel('Composite Importance Score', color=zerve_primary_text, fontsize=13)
ax3.set_title('Top Predictive Behaviors for Long-Term Success\n(Composite Score: GB + RF + Permutation)', 
              color=zerve_primary_text, fontsize=15, fontweight='bold', pad=20)
ax3.tick_params(colors=zerve_primary_text)
ax3.spines['bottom'].set_color(zerve_secondary_text)
ax3.spines['left'].set_color(zerve_secondary_text)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

# Add value labels with direction arrows
for i, (bar, row) in enumerate(zip(bars3, features_sorted.itertuples())):
    val = row.Composite_Score
    arrow = '↑' if row.Correlation > 0 else '↓'
    ax3.text(val + 0.01, bar.get_y() + bar.get_height()/2, 
             f'{val:.3f} {arrow}', va='center', color=zerve_primary_text, fontsize=11, fontweight='bold')

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=zerve_success, alpha=0.8, label='Positive Impact (↑)'),
    Patch(facecolor=zerve_warning, alpha=0.8, label='Negative Impact (↓)')
]
ax3.legend(handles=legend_elements, loc='lower right', 
           frameon=True, facecolor=zerve_dark_bg, edgecolor=zerve_secondary_text,
           labelcolor=zerve_primary_text, fontsize=11)

plt.tight_layout()
print("✓ Created composite feature ranking chart")

# 3. ROC Curve comparison
fig3, ax4 = plt.subplots(figsize=(10, 8))
fig3.patch.set_facecolor(zerve_dark_bg)
ax4.set_facecolor(zerve_dark_bg)

# Get predictions for both models
gb_pred_proba = gb_model.predict_proba(X_scaled)[:, 1]
rf_pred_proba = rf_model.predict_proba(X_scaled)[:, 1]

# Calculate ROC curves
gb_fpr, gb_tpr, _ = roc_curve(y, gb_pred_proba)
rf_fpr, rf_tpr, _ = roc_curve(y, rf_pred_proba)
gb_auc = auc(gb_fpr, gb_tpr)
rf_auc = auc(rf_fpr, rf_tpr)

# Plot ROC curves
ax4.plot(gb_fpr, gb_tpr, color=zerve_colors[0], linewidth=3, 
         label=f'Gradient Boosting (AUC = {gb_auc:.4f})')
ax4.plot(rf_fpr, rf_tpr, color=zerve_colors[1], linewidth=3, 
         label=f'Random Forest (AUC = {rf_auc:.4f})')
ax4.plot([0, 1], [0, 1], color=zerve_secondary_text, linestyle='--', linewidth=2, 
         label='Random Classifier (AUC = 0.50)')

ax4.set_xlabel('False Positive Rate', color=zerve_primary_text, fontsize=13)
ax4.set_ylabel('True Positive Rate', color=zerve_primary_text, fontsize=13)
ax4.set_title('ROC Curve: Model Performance Comparison', 
              color=zerve_primary_text, fontsize=15, fontweight='bold', pad=20)
ax4.legend(loc='lower right', frameon=True, facecolor=zerve_dark_bg, 
           edgecolor=zerve_secondary_text, labelcolor=zerve_primary_text, fontsize=12)
ax4.tick_params(colors=zerve_primary_text)
ax4.spines['bottom'].set_color(zerve_secondary_text)
ax4.spines['left'].set_color(zerve_secondary_text)
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
ax4.grid(True, alpha=0.2, color=zerve_secondary_text)

plt.tight_layout()
print("✓ Created ROC curve comparison")

# 4. Model Performance Summary
fig4, ax5 = plt.subplots(figsize=(12, 7))
fig4.patch.set_facecolor(zerve_dark_bg)
ax5.set_facecolor(zerve_dark_bg)

metrics = ['ROC AUC', 'Accuracy', 'Precision', 'Recall', 'F1']
gb_scores = [
    gb_cv_results['test_roc_auc'].mean(),
    gb_cv_results['test_accuracy'].mean(),
    gb_cv_results['test_precision'].mean(),
    gb_cv_results['test_recall'].mean(),
    gb_cv_results['test_f1'].mean()
]
rf_scores = [
    rf_cv_results['test_roc_auc'].mean(),
    rf_cv_results['test_accuracy'].mean(),
    rf_cv_results['test_precision'].mean(),
    rf_cv_results['test_recall'].mean(),
    rf_cv_results['test_f1'].mean()
]

x = np.arange(len(metrics))
width = 0.35

bars_gb = ax5.bar(x - width/2, gb_scores, width, label='Gradient Boosting', 
                   color=zerve_colors[0], alpha=0.9)
bars_rf = ax5.bar(x + width/2, rf_scores, width, label='Random Forest', 
                   color=zerve_colors[1], alpha=0.9)

ax5.set_ylabel('Score', color=zerve_primary_text, fontsize=13)
ax5.set_title('Model Performance Comparison (5-Fold Cross-Validation)', 
              color=zerve_primary_text, fontsize=15, fontweight='bold', pad=20)
ax5.set_xticks(x)
ax5.set_xticklabels(metrics, color=zerve_primary_text, fontsize=12)
ax5.tick_params(colors=zerve_primary_text)
ax5.legend(frameon=True, facecolor=zerve_dark_bg, edgecolor=zerve_secondary_text,
           labelcolor=zerve_primary_text, fontsize=12)
ax5.spines['bottom'].set_color(zerve_secondary_text)
ax5.spines['left'].set_color(zerve_secondary_text)
ax5.spines['top'].set_visible(False)
ax5.spines['right'].set_visible(False)
ax5.set_ylim([0.95, 1.01])

# Add value labels
for bars in [bars_gb, bars_rf]:
    for bar in bars:
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', 
                color=zerve_primary_text, fontsize=10)

plt.tight_layout()
print("✓ Created model performance comparison chart")

print("\n" + "="*70)
print("VISUALIZATION SUMMARY")
print("="*70)
print("\n✓ 4 comprehensive visualizations created:")
print("  1. Feature Importance: GB vs RF comparison")
print("  2. Composite Feature Ranking: Top predictive behaviors with impact directions")
print("  3. ROC Curves: Model discrimination performance")
print("  4. Cross-Validation Metrics: Comprehensive performance comparison")
print(f"\n✓ Best Model: {best_model_name} with AUC = {best_auc:.4f}")
print(f"✓ Success Criteria: {'✓ MET' if best_auc > 0.70 else '✗ NOT MET'} (threshold: 0.70)")
