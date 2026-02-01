import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Zerve design system
zerve_dark_bg = '#1D1D20'
zerve_primary_text = '#fbfbff'
zerve_secondary_text = '#909094'
zerve_colors = ['#A1C9F4', '#FFB482', '#8DE5A1', '#FF9F9B', '#D0BBFF']
zerve_success = '#17b26a'
zerve_warning = '#f04438'

print("=== GENERATING BEHAVIORAL COMPARISON VISUALIZATIONS ===\n")

# Top discriminating features by effect size
top_features = [
    ('unique_event_types', 'Unique Event Types'),
    ('avg_session_duration_ms', 'Avg Session Duration (min)'),
    ('events_per_day', 'Events Per Day'),
    ('active_days', 'Active Days'),
    ('total_events', 'Total Events')
]

# Create comparison visualizations
fig1, axes = plt.subplots(2, 3, figsize=(16, 10))
fig1.patch.set_facecolor(zerve_dark_bg)
axes = axes.flatten()

for idx, (feature, label) in enumerate(top_features):
    ax = axes[idx]
    ax.set_facecolor(zerve_dark_bg)
    
    # Prepare data
    successful_vals = successful[feature].dropna()
    unsuccessful_vals = unsuccessful[feature].dropna()
    
    # Convert session duration to minutes for readability
    if feature == 'avg_session_duration_ms':
        successful_vals = successful_vals / 60000
        unsuccessful_vals = unsuccessful_vals / 60000
    
    # Box plot comparison
    positions = [1, 2]
    bp = ax.boxplot([successful_vals, unsuccessful_vals], 
                     positions=positions,
                     widths=0.6,
                     patch_artist=True,
                     showfliers=False,
                     medianprops=dict(color=zerve_primary_text, linewidth=2),
                     whiskerprops=dict(color=zerve_secondary_text),
                     capprops=dict(color=zerve_secondary_text))
    
    # Color boxes
    bp['boxes'][0].set_facecolor(zerve_success)
    bp['boxes'][0].set_alpha(0.7)
    bp['boxes'][1].set_facecolor(zerve_warning)
    bp['boxes'][1].set_alpha(0.7)
    
    ax.set_xticks(positions)
    ax.set_xticklabels(['Successful', 'Unsuccessful'], color=zerve_primary_text, fontsize=10)
    ax.set_ylabel(label, color=zerve_primary_text, fontsize=10)
    ax.tick_params(colors=zerve_secondary_text, labelsize=9)
    ax.spines['bottom'].set_color(zerve_secondary_text)
    ax.spines['left'].set_color(zerve_secondary_text)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.2, color=zerve_secondary_text)
    
    # Add mean annotations
    mean_successful = successful_vals.mean()
    mean_unsuccessful = unsuccessful_vals.mean()
    ax.text(1, ax.get_ylim()[1] * 0.95, f'μ={mean_successful:.1f}', 
            ha='center', va='top', color=zerve_primary_text, fontsize=8,
            bbox=dict(boxstyle='round,pad=0.3', facecolor=zerve_dark_bg, edgecolor=zerve_success, alpha=0.8))
    ax.text(2, ax.get_ylim()[1] * 0.95, f'μ={mean_unsuccessful:.1f}', 
            ha='center', va='top', color=zerve_primary_text, fontsize=8,
            bbox=dict(boxstyle='round,pad=0.3', facecolor=zerve_dark_bg, edgecolor=zerve_warning, alpha=0.8))

# Remove extra subplot
axes[-1].remove()

plt.suptitle('Behavioral Patterns: Successful vs Unsuccessful Users', 
             color=zerve_primary_text, fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.97])
fig_comparison = fig1

print("✓ Box plot comparison created")

# Effect size visualization
fig2, ax = plt.subplots(figsize=(12, 7))
fig2.patch.set_facecolor(zerve_dark_bg)
ax.set_facecolor(zerve_dark_bg)

effect_sizes = comparison_df_sorted[['Feature', 'Cohens D']].copy()
effect_sizes['Abs_Effect'] = effect_sizes['Cohens D'].abs()
effect_sizes = effect_sizes.sort_values('Abs_Effect')

colors_effect = [zerve_warning if d < 0 else zerve_success for d in effect_sizes['Cohens D']]

bars = ax.barh(range(len(effect_sizes)), effect_sizes['Cohens D'], color=colors_effect, alpha=0.8)

# Add magnitude thresholds
ax.axvline(x=0.2, color=zerve_secondary_text, linestyle='--', alpha=0.4, linewidth=1)
ax.axvline(x=0.5, color=zerve_secondary_text, linestyle='--', alpha=0.4, linewidth=1)
ax.axvline(x=0.8, color=zerve_secondary_text, linestyle='--', alpha=0.4, linewidth=1)
ax.axvline(x=-0.2, color=zerve_secondary_text, linestyle='--', alpha=0.4, linewidth=1)
ax.axvline(x=-0.5, color=zerve_secondary_text, linestyle='--', alpha=0.4, linewidth=1)
ax.axvline(x=-0.8, color=zerve_secondary_text, linestyle='--', alpha=0.4, linewidth=1)

# Labels
feature_labels = [f.replace('_', ' ').title() for f in effect_sizes['Feature']]
ax.set_yticks(range(len(effect_sizes)))
ax.set_yticklabels(feature_labels, color=zerve_primary_text, fontsize=10)
ax.set_xlabel("Cohen's D (Effect Size)", color=zerve_primary_text, fontsize=11, fontweight='bold')
ax.set_title("Effect Sizes: Key Behavioral Discriminators", 
             color=zerve_primary_text, fontsize=13, fontweight='bold', pad=15)

ax.tick_params(colors=zerve_secondary_text, labelsize=10)
ax.spines['bottom'].set_color(zerve_secondary_text)
ax.spines['left'].set_color(zerve_secondary_text)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='x', alpha=0.2, color=zerve_secondary_text)

# Add value labels
for idx, (bar, val) in enumerate(zip(bars, effect_sizes['Cohens D'])):
    x_pos = val + (0.05 if val > 0 else -0.05)
    ha = 'left' if val > 0 else 'right'
    ax.text(x_pos, idx, f'{val:.2f}', va='center', ha=ha, 
            color=zerve_primary_text, fontsize=9, fontweight='bold')

plt.tight_layout()
fig_effect_sizes = fig2

print("✓ Effect size chart created")

# Distribution comparison for top feature (unique_event_types)
fig3, ax = plt.subplots(figsize=(12, 6))
fig3.patch.set_facecolor(zerve_dark_bg)
ax.set_facecolor(zerve_dark_bg)

# Histogram
bins_range = np.arange(0, 31, 1)
ax.hist(successful['unique_event_types'], bins=bins_range, alpha=0.6, 
        color=zerve_success, label='Successful Users', edgecolor='none')
ax.hist(unsuccessful['unique_event_types'], bins=bins_range, alpha=0.6, 
        color=zerve_warning, label='Unsuccessful Users', edgecolor='none')

ax.set_xlabel('Unique Event Types', color=zerve_primary_text, fontsize=11, fontweight='bold')
ax.set_ylabel('Number of Users', color=zerve_primary_text, fontsize=11, fontweight='bold')
ax.set_title('Feature Adoption: Unique Event Types Distribution', 
             color=zerve_primary_text, fontsize=13, fontweight='bold', pad=15)

ax.tick_params(colors=zerve_secondary_text, labelsize=10)
ax.spines['bottom'].set_color(zerve_secondary_text)
ax.spines['left'].set_color(zerve_secondary_text)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.2, color=zerve_secondary_text)

legend = ax.legend(loc='upper right', framealpha=0.9, fontsize=10)
legend.get_frame().set_facecolor(zerve_dark_bg)
legend.get_frame().set_edgecolor(zerve_secondary_text)
for text in legend.get_texts():
    text.set_color(zerve_primary_text)

plt.tight_layout()
fig_distribution = fig3

print("✓ Distribution comparison created")
print("\nAll behavioral comparison visualizations generated successfully!")
