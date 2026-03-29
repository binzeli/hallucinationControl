import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import rcParams
from matplotlib.patheffects import withSimplePatchShadow
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator

# -------------------- 
# Configuration
# -------------------- 
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
rcParams['figure.facecolor'] = '#ffffff'
rcParams['axes.edgecolor'] = '#1c1c1c'
rcParams['axes.linewidth'] = 1.8

# -------------------- 
# Data
# -------------------- 
# Setup 1: Penalty = -1
setups_1 = [
    "A baseline", "A", "B control", "B (+0.2)", "B w/ norms (+0.2)", 
    "B (+0.4)", "B w/ norms (+0.4)", "B (+0.6)", "B w/ norms (+0.6)", 
    "B (+0.8)", "B w/ norms (+0.8)",
]
answered_1 = [13829, 11982, 10737, 9704, 8253, 9683, 7888, 9633, 8282, 9124, 8510]
far1_1 = [0.524, 0.482, 0.444, 0.410, 0.355, 0.41, 0.342, 0.412, 0.365, 0.392, 0.37]

# Setup 2: Penalty = 0
setups_2 = [
    "A baseline", "A", "B control", "B (+0.2)", "B w/ norms (+0.2)", 
    "B (+0.4)", "B w/ norms (+0.4)", "B (+0.6)", "B w/ norms (+0.6)", 
    "B (+0.8)", "B w/ norms (+0.8)",
]
answered_2 = [13829, 12403, 11165, 10830, 9280, 10788, 9501, 10377, 8774, 10515, 9428]
far1_2 = [0.522, 0.491, 0.457, 0.444, 0.399, 0.445, 0.404, 0.431, 0.379, 0.435, 0.404]

# Convert to proportions, then to abstention rate (1 - proportion answered)
total_questions = 14267
abstention_1 = [1 - x / total_questions for x in answered_1]
abstention_2 = [1 - x / total_questions for x in answered_2]

# Add `Pure eval` data point
pure_eval_answered = 13768
pure_eval_far1 = 0.523
pure_eval_abstention = 1 - pure_eval_answered / total_questions

# Exclude certain labels from the plot (remove A baseline and B control)
exclude_labels = {"A baseline", "B control"}
kept_indices_1 = [i for i, lbl in enumerate(setups_1) if lbl not in exclude_labels]
kept_indices_2 = [i for i, lbl in enumerate(setups_2) if lbl not in exclude_labels]

filtered_setups_1 = [setups_1[i] for i in kept_indices_1]
filtered_abstention_1 = [abstention_1[i] for i in kept_indices_1]
filtered_far1_1 = [far1_1[i] for i in kept_indices_1]

filtered_setups_2 = [setups_2[i] for i in kept_indices_2]
filtered_abstention_2 = [abstention_2[i] for i in kept_indices_2]
filtered_far1_2 = [far1_2[i] for i in kept_indices_2]

# Combine all points for Pareto frontier (include pure eval)
abstention_all = filtered_abstention_1 + filtered_abstention_2 + [pure_eval_abstention]
far1_all = filtered_far1_1 + filtered_far1_2 + [pure_eval_far1]

# -------------------- 
# Compute Pareto frontier
# Higher abstention = fewer answers. We want lower FAR1 with lower abstention.
# Pareto: for increasing abstention, FAR1 should be decreasing.
# -------------------- 
points = sorted(zip(abstention_all, far1_all), key=lambda x: x[0])
pareto_x = []
pareto_y = []
best_far1 = float("inf")
for x, y in points:
    if y < best_far1:
        pareto_x.append(x)
        pareto_y.append(y)
        best_far1 = y

pareto_x, pareto_y = zip(*sorted(zip(pareto_x, pareto_y)))

# -------------------- 
# Create figure with extra space for labels
# -------------------- 
fig = plt.figure(figsize=(11, 6), dpi=300)
ax = fig.add_subplot(111)

# Color palette
color_neg1 = '#0F2F4F'       # Deep navy (Setup 1)
color_0 = '#D95428'          # Burnt sienna (Setup 2)
color_pareto = '#888EA0'     # Burgundy
color_bg_light = '#faf8f5'
color_text = '#1c1c1c'

# Background
ax.set_facecolor(color_bg_light)
fig.patch.set_facecolor('#ffffff')

# Grid
ax.grid(True, alpha=0.10, linestyle=':', linewidth=0.3, 
       color='#e5e1db', zorder=0, which='minor')
ax.set_axisbelow(True)

# ==================== SCATTER POINTS ====================
# Setup 1: Penalty = -1 (circles)
ax.scatter(filtered_abstention_1, filtered_far1_1, marker='o', s=130, alpha=0.82, 
          color=color_neg1, edgecolors='#ffffff', linewidth=2.2,
          label='Penalty for incorrect answers = −1', zorder=5, rasterized=True,
          path_effects=[withSimplePatchShadow(offset=(0.5, -0.5), 
                                              shadow_rgbFace='black', 
                                              alpha=0.1)])

# Setup 2: Penalty = 0 (squares)
ax.scatter(filtered_abstention_2, filtered_far1_2, marker='s', s=130, alpha=0.82, 
          color=color_0, edgecolors='#ffffff', linewidth=2.2,
          label='Penalty for incorrect answers = 0', zorder=5, rasterized=True,
          path_effects=[withSimplePatchShadow(offset=(0.5, -0.5), 
                                              shadow_rgbFace='black', 
                                              alpha=0.1)])

# ==================== PURE EVAL POINT ====================
ax.scatter([pure_eval_abstention], [pure_eval_far1], marker='X', s=220, 
           color='#2E8B57', edgecolors='#ffffff', linewidth=1.6,
           label='_nolegend_', zorder=6)
ax.annotate('Pure eval', (pure_eval_abstention, pure_eval_far1),
           xytext=(pure_eval_abstention + 0.01, pure_eval_far1 + 0.001),
           textcoords='data', ha='left', va='bottom',
           fontsize=12, fontweight='600', family='sans-serif',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffffff',
                     edgecolor='#d0cdc8', alpha=0.95, linewidth=0.6),
           arrowprops=dict(arrowstyle='-', lw=0.6, color='#2E8B57',
                           alpha=0.25, connectionstyle='arc3,rad=0.15',
                           shrinkA=2, shrinkB=2))



# ==================== PARETO FRONTIER ====================
ax.plot(pareto_x, pareto_y, color=color_pareto, linewidth=2.2, 
    label='_nolegend_', zorder=4, linestyle='--', alpha=0.95)


# ==================== INTELLIGENT ANNOTATIONS ====================
annotation_style_1 = dict(
    fontsize=13, alpha=0.87, family='sans-serif', weight='normal',
    color=color_neg1,
    bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffffff',
              edgecolor='#d0cdc8', alpha=0.90, linewidth=0.6),
    arrowprops=dict(arrowstyle='-', lw=0.6, color=color_neg1,
                   alpha=0.25, connectionstyle='arc3,rad=0.15',
                   shrinkA=2, shrinkB=2)
)

annotation_style_2 = dict(
    fontsize=13, alpha=0.87, family='sans-serif', weight='normal',
    color=color_0,
    bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffffff',
              edgecolor='#d0cdc8', alpha=0.90, linewidth=0.6),
    arrowprops=dict(arrowstyle='-', lw=0.6, color=color_0,
                   alpha=0.25, connectionstyle='arc3,rad=0.15',
                   shrinkA=2, shrinkB=2)
)

# Define custom offsets for each filtered point
offsets_1 = {
    0: (0.010, 0.008, 'left', 'center'),      # A
    1: (0.03, 0.008, 'left', 'center'),      # B (+0.2)
    2: (-0.005, 0.001, 'right', 'center'),    # B w/ norm (+0.2)
    3: (-0.02, 0.017, 'left', 'center'),      # B (+0.4)
    4: (-0.015, 0.002, 'right', 'center'),    # B w/ norm (+0.4)
    5: (-0.06, 0.003, 'left', 'center'),       # B (+0.6)
    6: (0.07, 0.014, 'right', 'center'),      # B w/ norm (+0.6)
    7: (0.015, -0.001, 'left', 'center'),     # B (+0.8)
    8: (-0.015, -0.003, 'right', 'center'),    # B w/ norm (+0.8)
}

offsets_2 = {
    0: (-0.010, -0.003, 'right', 'top'),      # A
    1: (-0.05, -0.005, 'left', 'center'),     # B (+0.2)
    2: (-0.001, -0.008, 'right', 'center'),   # B w/ norm (+0.2)
    3: (0.0, 0.01, 'left', 'center'),       # B (+0.4)
    4: (0.1, 0.001, 'right', 'center'),      # B w/ norm (+0.4)
    5: (0.0, 0.01, 'left', 'center'),       # B (+0.6)
    6: (-0.015, 0.001, 'right', 'center'),    # B w/ norm (+0.6)
    7: (-0.04, -0.009, 'left', 'center'),     # B (+0.8)
    8: (-0.01, -0.003, 'right', 'center'),    # B w/ norm (+0.8)
}

# Annotate Setup 1
for i, (x, y, label) in enumerate(zip(filtered_abstention_1, filtered_far1_1, filtered_setups_1)):
    x_offset, y_offset, ha, va = offsets_1.get(i, (0.015, 0, 'left', 'center'))
    ax.annotate(label, (x, y), xytext=(x + x_offset, y + y_offset),
               textcoords='data', ha=ha, va=va, **annotation_style_1)

# Annotate Setup 2
for i, (x, y, label) in enumerate(zip(filtered_abstention_2, filtered_far1_2, filtered_setups_2)):
    x_offset, y_offset, ha, va = offsets_2.get(i, (0.015, 0, 'left', 'center'))
    ax.annotate(label, (x, y), xytext=(x + x_offset, y + y_offset),
               textcoords='data', ha=ha, va=va, **annotation_style_2)

# ==================== AXES & LABELS ====================
ax.set_xlabel('Abstention Rate', 
             fontsize=22, fontweight='700', labelpad=14, 
             color=color_text, family='sans-serif')
ax.set_ylabel('False Answer Rate (Answered)', 
             fontsize=22, fontweight='700', labelpad=14, 
             color=color_text, family='sans-serif')

# Set limits (x is now abstention rate: 0 = answered everything, 1 = answered nothing)
ax.set_xlim(0.02, 0.49)
ax.set_ylim(0.335, 0.54)

# Force 0.05 step ticks on x-axis
ax.xaxis.set_major_locator(MultipleLocator(0.05))

# Format x-axis as decimal numbers
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.2f}'.format(y)))

# Spines
for spine in ax.spines.values():
    spine.set_edgecolor(color_text)
    spine.set_linewidth(1.8)

# Legend
legend = ax.legend(loc='upper right', fontsize=15, framealpha=0.98,
                  edgecolor='#999999', fancybox=False, 
                  frameon=True, shadow=False,
                  title_fontsize=20, labelspacing=1.3, handlelength=2.4)
legend.get_frame().set_linewidth(1.2)
legend.get_frame().set_facecolor('#ffffff')
for text in legend.get_texts():
    text.set_color('#2c2c2c')
    text.set_fontweight('500')
legend.get_title().set_color('#0a0a0a')
legend.get_title().set_fontweight('800')

# Ticks
ax.tick_params(axis='both', which='major', labelsize=20, 
              colors=color_text, length=7.5, width=1.4, 
              top=False, right=False)
ax.tick_params(axis='both', which='minor', length=4, width=0.9, 
              top=False, right=False, color=color_text)

for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontweight('500')

plt.tight_layout()

# Save
plt.savefig('parento_frontier.png', dpi=300, bbox_inches='tight', 
           facecolor='#ffffff', edgecolor='none', pad_inches=0.4)
plt.savefig('parento_frontier.pdf', bbox_inches='tight',
           facecolor='#ffffff', edgecolor='none', pad_inches=0.4)
print("Saved successfully.")
