import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib import rcParams

rcParams['font.family'] = 'DejaVu Sans'
rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False

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
        '-', 0.2, 0.2, 0.4, 0.4, 0.6, 0.6, 0.8, 0.8
    ],
    'Abstention Ratio': [
        0.058, 0.213, 0.386, 0.528, 0.385, 0.514, 0.421,
        0.597, 0.418, 0.538,
        0.262, 0.5, 0.643, 0.499,
        0.656, 0.5, 0.622, 0.551, 0.62
    ]
}

df = pd.DataFrame(data)

colors = {
    'Pure Eval': '#1A3A4A',
    'A':         '#2E7BB4',
    'B':         '#64B4CF',
    'B w/ norms':'#B8DDE9'
}

BG = '#FFFFFF'
GRID_COLOR = '#E0E0E0'
SPINE_COLOR = '#AAAAAA'
TEXT_COLOR = '#1A1A2E'
ANNOT_COLOR = '#444444'

fig2, ax3 = plt.subplots(figsize=(15, 6))
fig2.patch.set_facecolor(BG)
ax3.set_facecolor(BG)

pure_eval_data = df[df['Setting'] == 'Pure Eval'].reset_index(drop=True)
other_data = df[df['Setting'] != 'Pure Eval'].reset_index(drop=True)

x_positions = []
x_labels = []
bar_colors = []
bar_values = []

x_offset = 0

for i, row in pure_eval_data.iterrows():
    x_positions.append(x_offset + i)
    x_labels.append(f"{row['Setting']}")
    bar_colors.append(colors.get(row['Setting'], '#cccccc'))
    bar_values.append(row['Abstention Ratio'])

x_offset += len(pure_eval_data) + 1.8

for penalty_val in [0, -1]:
    group = other_data[other_data['False Penalty'] == penalty_val].reset_index(drop=True)
    for i, row in group.iterrows():
        x_positions.append(x_offset + i)
        if row['Setting'] == 'A':
            x_labels.append(f"{row['Setting']}")
        else:
            x_labels.append(f"{row['Setting']}\n(+{row['IDK Reward']})")
        bar_colors.append(colors.get(row['Setting'], '#cccccc'))
        bar_values.append(row['Abstention Ratio'])
    x_offset += len(group) + 1.2

for xp, val, bc in zip(x_positions, bar_values, bar_colors):
    ax3.bar(xp + 0.04, val, color='#CCCCCC', width=0.72, zorder=1, alpha=0.5)
    ax3.bar(xp, val, color=bc, width=0.72, zorder=2, linewidth=0, alpha=0.92)
    ax3.plot([xp - 0.36, xp + 0.36], [val, val], color='white', linewidth=1.5, zorder=3, alpha=0.6)

pure_eval_count = len(pure_eval_data)
penalty_0_count = len(other_data[other_data['False Penalty'] == 0])

sep1_x = pure_eval_count + 0.55
sep2_x = pure_eval_count + penalty_0_count + 1.95

for sx in [sep1_x, sep2_x]:
    ax3.axvline(sx, color='#BBBBBB', linestyle=':', linewidth=2, alpha=0.9, zorder=0)

def section_banner(ax, x_center, y, label, facecolor, edgecolor):
    ax.text(x_center, y, label,
            fontsize=14, fontweight='bold', ha='center', va='center', color=TEXT_COLOR,
            bbox=dict(boxstyle='round,pad=0.45', facecolor=facecolor,
                      edgecolor=edgecolor, linewidth=1.4, alpha=0.88))

p0_center = sep1_x + penalty_0_count / 2
p1_center = sep2_x + (len(other_data[other_data['False Penalty'] == -1])) / 2

section_banner(ax3, p0_center, 0.73, 'Penalty for incorrect answers = 0', '#FFF3CD', '#D4A017')
section_banner(ax3, p1_center, 0.73, 'Penalty for incorrect answers = −1', '#D6EAF8', '#2E7BB4')

ax3.set_xlim(-0.7, max(x_positions) + 0.8)
ax3.set_ylim(0, 0.80)
ax3.set_xticks(x_positions)
ax3.set_xticklabels(x_labels, fontsize=15, rotation=45, ha='center', color=TEXT_COLOR)
ax3.set_ylabel('Abstention over Error Ratio', fontsize=18, fontweight='bold', color=TEXT_COLOR, labelpad=10)
ax3.set_xlabel('Settings & Reward Configurations', fontsize=22, fontweight='bold', color=TEXT_COLOR, labelpad=12)

ax3.yaxis.set_tick_params(labelsize=17, colors=ANNOT_COLOR)
ax3.xaxis.set_tick_params(colors=TEXT_COLOR)
ax3.spines['left'].set_color(SPINE_COLOR)
ax3.spines['bottom'].set_color(SPINE_COLOR)
ax3.spines['left'].set_linewidth(1.2)
ax3.spines['bottom'].set_linewidth(1.2)
ax3.yaxis.grid(True, color=GRID_COLOR, linestyle='--', linewidth=0.9, zorder=0)
ax3.set_axisbelow(True)

legend_elements = [mpatches.Patch(facecolor=color, edgecolor='#666666', linewidth=0.8,
                                   label=scheme, alpha=0.92)
                   for scheme, color in colors.items()]
legend = ax3.legend(handles=legend_elements, loc='upper left', fontsize=14,
                    frameon=True, edgecolor='#DDDDDD', fancybox=True,
                    framealpha=0.95, title='Scheme', title_fontsize=16)
legend.get_frame().set_linewidth(1.2)

plt.tight_layout(pad=2.0)
plt.savefig('abstention_ratio.png', dpi=180, bbox_inches='tight', facecolor=BG)
plt.savefig('abstention_ratio.pdf', bbox_inches='tight', facecolor=BG)
print("Saved to abstention_ratio_comprehensive.png")