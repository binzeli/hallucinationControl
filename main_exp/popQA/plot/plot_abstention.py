import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Data extracted from the spreadsheet - B w/ norm v2 removed, Pure Eval added (one bar only), B control and A baseline removed
data = {
    'Setting': [
        'Pure Eval', 'A', 'B', 'B w/ norms',
        'B', 'B w/ norms', 'B', 'B w/ norms',
        'B', 'B w/ norms', 'A', 'B', 'B w/ norms', 'B', 'B w/ norms',
        'B', 'B w/ norms', 'B', 'B w/ norms'
    ],
    'False Penalty': [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        -1, -1, -1, -1, -1, -1, -1, -1, -1
    ],
    'IDK Reward': [
        '-', '-', 0.2, 0.2, 0.4, 0.4, 0.6, 0.6, 0.8, 0.8, 
        '-', 0.2, 0.2, 0.4, 0.4, 0.6, 0.6, 0.8  , 0.8
    ],
    'Abstention Ratio': [
        0.06, 0.213, 0.386, 0.528, 0.385, 0.514, 0.421,
        0.597, 0.418, 0.538, 
        0.262, 0.5, 0.643, 0.499,
        0.656, 0.5, 0.622, 0.551, 0.62
    ]
}

df = pd.DataFrame(data)

# Define colors for each scheme
colors = {
    'Pure Eval': '#2ca02c',
    'A': '#aec7e8',
    'B': '#ffbb78',
    'B w/ norms': '#d62728'
}

# Separate data by penalty level
penalty_0 = df[df['False Penalty'] == 0].reset_index(drop=True)
penalty_minus1 = df[df['False Penalty'] == -1].reset_index(drop=True)

# Create subplots for cleaner visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: False Penalty = 0
x_pos = np.arange(len(penalty_0))
for i, row in penalty_0.iterrows():
    ax1.bar(i, row['Abstention Ratio'], 
           color=colors.get(row['Setting'], '#cccccc'),
           edgecolor='black', linewidth=1.5, alpha=0.8)
    # Add IDK reward label
    idk_label = str(row['IDK Reward'])
    ax1.text(i, row['Abstention Ratio'] + 0.01, idk_label, 
            ha='center', va='bottom', fontsize=12, fontweight='bold')

ax1.set_xlabel('Configuration (IDK Reward values)', fontsize=16, fontweight='bold')
ax1.set_ylabel('Abstention over Incorrect Ratio', fontsize=16, fontweight='bold')
ax1.set_title('Reward for incorrect answers = 0', fontsize=16, fontweight='bold')
ax1.set_xticks(x_pos)
ax1.set_xticklabels([s.replace(' ', '\n') for s in penalty_0['Setting']], fontsize=14)
ax1.set_ylim(0, 0.7)
ax1.grid(axis='y', alpha=0.3, linestyle='--')
ax1.set_axisbelow(True)
ax1.tick_params(axis='both', which='major', labelsize=14)

# Plot 2: False Penalty = -1
x_pos = np.arange(len(penalty_minus1))
for i, row in penalty_minus1.iterrows():
    ax2.bar(i, row['Abstention Ratio'], 
           color=colors.get(row['Setting'], '#cccccc'),
           edgecolor='black', linewidth=1.5, alpha=0.8)
    # Add IDK reward label
    idk_label = str(row['IDK Reward'])
    ax2.text(i, row['Abstention Ratio'] + 0.01, idk_label, 
            ha='center', va='bottom', fontsize=12, fontweight='bold')

ax2.set_xlabel('Configuration (IDK Reward values)', fontsize=16, fontweight='bold')
ax2.set_ylabel('Abstention over Incorrect Ratio', fontsize=16, fontweight='bold')
ax2.set_title('Reward for incorrect answers = -1', fontsize=16, fontweight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels([s.replace(' ', '\n') for s in penalty_minus1['Setting']], fontsize=14)
ax2.set_ylim(0, 0.7)
ax2.grid(axis='y', alpha=0.3, linestyle='--')
ax2.set_axisbelow(True)
ax2.tick_params(axis='both', which='major', labelsize=14)

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=color, edgecolor='black', label=scheme, alpha=0.8) 
                   for scheme, color in colors.items()]
fig.legend(handles=legend_elements, loc='upper center', ncol=3, 
          bbox_to_anchor=(0.5, -0.02), fontsize=13, frameon=True)

plt.tight_layout()
plt.subplots_adjust(bottom=0.15)

# Create alternative visualization: Single plot with Pure Eval separated
fig2, ax3 = plt.subplots(figsize=(18, 6))

# Separate Pure Eval from the rest
pure_eval_data = df[df['Setting'] == 'Pure Eval'].reset_index(drop=True)
other_data = df[df['Setting'] != 'Pure Eval'].reset_index(drop=True)

# Create x positions and labels
x_positions = []
x_labels = []
bar_colors = []
bar_values = []

x_offset = 0

# Add Pure Eval bar first (no IDK reward label)
for i, row in pure_eval_data.iterrows():
    x_positions.append(x_offset + i)
    x_labels.append(f"{row['Setting']}")
    bar_colors.append(colors.get(row['Setting'], '#cccccc'))
    bar_values.append(row['Abstention Ratio'])

x_offset += len(pure_eval_data) + 2

# Add the penalty groups
for penalty_val in [0, -1]:
    group = other_data[other_data['False Penalty'] == penalty_val].reset_index(drop=True)
    
    for i, row in group.iterrows():
        x_positions.append(x_offset + i)
        # Don't show IDK reward for scheme A
        if row['Setting'] == 'A':
            x_labels.append(f"{row['Setting']}")
        else:
            x_labels.append(f"{row['Setting']}\n(+{row['IDK Reward']})")
        bar_colors.append(colors.get(row['Setting'], '#cccccc'))
        bar_values.append(row['Abstention Ratio'])
    
    x_offset += len(group) + 1.5

# Create bars
bars = ax3.bar(x_positions, bar_values, color=bar_colors, edgecolor='black', linewidth=1.5, alpha=0.85, width=0.8)

# Numeric value labels on each bar removed per request

# Add vertical separator lines
pure_eval_count = len(pure_eval_data)
penalty_0_count = len(other_data[other_data['False Penalty'] == 0])

separator_x1 = pure_eval_count + 0.75
separator_x2 = pure_eval_count + penalty_0_count + 2.75

ax3.axvline(separator_x1, color='gray', linestyle='--', linewidth=2.5, alpha=0.6)
ax3.axvline(separator_x2, color='gray', linestyle='--', linewidth=2.5, alpha=0.6)

ax3.text(separator_x1 + penalty_0_count/2, 0.7, 'penalty for incorrect answers = 0', 
    fontsize=14, fontweight='bold', ha='center', 
    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6, edgecolor='black', linewidth=1.5))
ax3.text(separator_x2 + penalty_0_count/2, 0.7, 'penalty for incorrect answers = -1', 
    fontsize=14, fontweight='bold', ha='center',
    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.6, edgecolor='black', linewidth=1.5))

ax3.set_xlabel('Settings & Reward Configurations', fontsize=17, fontweight='bold')
ax3.set_ylabel('Abstention over Error Ratio', fontsize=17, fontweight='bold')
ax3.set_xticks(x_positions)
ax3.set_xticklabels(x_labels, fontsize=13, rotation=45, ha='center')
ax3.set_ylim(0, 0.75)
ax3.grid(axis='y', alpha=0.3, linestyle='--')
ax3.set_axisbelow(True)
ax3.tick_params(axis='both', which='major', labelsize=14)

# Add legend
legend_elements = [Patch(facecolor=color, edgecolor='black', label=scheme, alpha=0.85) 
                   for scheme, color in colors.items()]
ax3.legend(handles=legend_elements, loc='upper left', fontsize=13, frameon=True, edgecolor='black', fancybox=True)

plt.tight_layout()
plt.savefig('abstention_ratio_comprehensive.png', dpi=300, bbox_inches='tight')
print("Comprehensive plot saved to abstention_ratio_comprehensive.png")

plt.close('all')
print("\nAll plots generated successfully!")