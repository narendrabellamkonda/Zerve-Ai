import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define Zerve color palette
zerve_dark_bg = '#1D1D20'
zerve_primary_text = '#fbfbff'
zerve_secondary_text = '#909094'
zerve_colors = ['#A1C9F4', '#FFB482', '#8DE5A1', '#FF9F9B', '#D0BBFF']
zerve_highlight = '#ffd400'
zerve_success = '#17b26a'
zerve_warning = '#f04438'

# Success Rate Comparison
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig1.patch.set_facecolor(zerve_dark_bg)

# Binary success distribution
success_counts = success_features['is_successful'].value_counts()
labels = ['Not Successful', 'Successful']
ax1.bar(labels, success_counts.values, color=[zerve_warning, zerve_success], alpha=0.8, edgecolor=zerve_primary_text)
ax1.set_ylabel('Number of Users', color=zerve_primary_text, fontsize=11)
ax1.set_title('Binary Success Distribution', color=zerve_primary_text, fontsize=13, fontweight='bold', pad=15)
ax1.set_facecolor(zerve_dark_bg)
ax1.tick_params(colors=zerve_primary_text)
ax1.spines['bottom'].set_color(zerve_secondary_text)
ax1.spines['left'].set_color(zerve_secondary_text)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
for i, v in enumerate(success_counts.values):
    ax1.text(i, v + 50, f'{v}\n({v/len(success_features)*100:.1f}%)', 
             ha='center', va='bottom', color=zerve_primary_text, fontsize=10)

# Success tier distribution
tier_counts = success_features['success_tier'].value_counts().sort_index()
colors_tier = [zerve_warning, '#FF6B6B', zerve_colors[1], zerve_colors[2], zerve_success]
ax2.barh(range(len(tier_counts)), tier_counts.values, color=colors_tier, alpha=0.8, edgecolor=zerve_primary_text)
ax2.set_yticks(range(len(tier_counts)))
ax2.set_yticklabels(tier_counts.index, color=zerve_primary_text)
ax2.set_xlabel('Number of Users', color=zerve_primary_text, fontsize=11)
ax2.set_title('Success Tier Distribution', color=zerve_primary_text, fontsize=13, fontweight='bold', pad=15)
ax2.set_facecolor(zerve_dark_bg)
ax2.tick_params(colors=zerve_primary_text)
ax2.spines['bottom'].set_color(zerve_secondary_text)
ax2.spines['left'].set_color(zerve_secondary_text)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
for i, v in enumerate(tier_counts.values):
    ax2.text(v + 30, i, f'{v}', va='center', color=zerve_primary_text, fontsize=9)

plt.tight_layout()
print("Figure 1: Success Distribution Metrics")
plt.show()

# Continuous success score distribution
fig2, ax = plt.subplots(figsize=(12, 5))
fig2.patch.set_facecolor(zerve_dark_bg)
ax.hist(success_features['success_score_continuous'], bins=50, color=zerve_colors[0], 
        alpha=0.7, edgecolor=zerve_primary_text, linewidth=0.5)
ax.axvline(success_features['success_score_continuous'].median(), color=zerve_highlight, 
           linestyle='--', linewidth=2, label=f'Median: {success_features["success_score_continuous"].median():.1f}')
ax.axvline(success_features['success_score_continuous'].mean(), color=zerve_success, 
           linestyle='--', linewidth=2, label=f'Mean: {success_features["success_score_continuous"].mean():.1f}')
ax.set_xlabel('Continuous Success Score (0-100)', color=zerve_primary_text, fontsize=11)
ax.set_ylabel('Number of Users', color=zerve_primary_text, fontsize=11)
ax.set_title('Distribution of Continuous Success Scores', color=zerve_primary_text, 
             fontsize=13, fontweight='bold', pad=15)
ax.set_facecolor(zerve_dark_bg)
ax.tick_params(colors=zerve_primary_text)
ax.spines['bottom'].set_color(zerve_secondary_text)
ax.spines['left'].set_color(zerve_secondary_text)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.legend(loc='upper right', facecolor=zerve_dark_bg, edgecolor=zerve_secondary_text, 
          labelcolor=zerve_primary_text, fontsize=10)
plt.tight_layout()
print("\nFigure 2: Continuous Success Score Distribution")
plt.show()

# Success criteria breakdown
fig3, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
fig3.patch.set_facecolor(zerve_dark_bg)

criteria = ['has_retention', 'has_engagement_depth', 'has_activity_volume', 'has_session_quality']
criteria_labels = ['Retention\n(2+ days)', 'Engagement\n(3+ event types)', 
                   'Activity\n(5+ events)', 'Session Quality\n(2+ min avg)']
axes = [ax1, ax2, ax3, ax4]

for i, (criterion, label, ax) in enumerate(zip(criteria, criteria_labels, axes)):
    counts = success_features[criterion].value_counts()
    colors = [zerve_warning, zerve_success]
    ax.bar(['Not Met', 'Met'], [counts.get(0, 0), counts.get(1, 0)], 
           color=colors, alpha=0.8, edgecolor=zerve_primary_text)
    ax.set_title(label, color=zerve_primary_text, fontsize=12, fontweight='bold', pad=10)
    ax.set_ylabel('Users', color=zerve_primary_text, fontsize=10)
    ax.set_facecolor(zerve_dark_bg)
    ax.tick_params(colors=zerve_primary_text)
    ax.spines['bottom'].set_color(zerve_secondary_text)
    ax.spines['left'].set_color(zerve_secondary_text)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    met_pct = counts.get(1, 0) / len(success_features) * 100
    ax.text(1, counts.get(1, 0) + 50, f'{met_pct:.1f}%', 
            ha='center', va='bottom', color=zerve_primary_text, fontsize=10, fontweight='bold')

fig3.suptitle('Success Criteria Component Analysis', color=zerve_primary_text, 
              fontsize=15, fontweight='bold', y=0.995)
plt.tight_layout()
print("\nFigure 3: Individual Success Criteria Breakdown")
plt.show()

print("\n✓ Success metrics visualized across 3 figures")
