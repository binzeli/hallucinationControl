import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

models = ['qwen-4B-instruct', 'Meta-Llama-3-8B-Instruct', 'gemini-3.1-flash-lite', 'gpt-5-mini', 'gpt-4o-mini']
model_labels = ['Qwen-4B\nInstruct', 'Meta-Llama-3\n8B-Instruct', 'Gemini-3.1\nFlash-Lite', 'GPT-5 mini', 'GPT-4o mini']
schemes = ['Pure Eval', 'Scheme A', 'Scheme B', 'Scheme B w/ norms']

far_vals = {
    'qwen-4B-instruct':          [0.780, 0.732, 0.719, 0.696],
    'Meta-Llama-3-8B-Instruct':  [0.683, 0.686, 0.639, 0.479],
    'gemini-3.1-flash-lite':     [0.444, 0.387, 0.395, 0.319],
    'gpt-5-mini':                [0.533, 0.482, 0.410, 0.342],
    'gpt-4o-mini':               [0.594, 0.492, 0.466, 0.417],
}
far_errs = {
    'qwen-4B-instruct':          [0.007, 0.009, 0.010, 0.011],
    'Meta-Llama-3-8B-Instruct':  [0.008, 0.008, 0.010, 0.014],
    'gemini-3.1-flash-lite':     [0.008, 0.009, 0.009, 0.009],
    'gpt-5-mini':                [0.008, 0.009, 0.010, 0.011],
    'gpt-4o-mini':               [0.008, 0.010, 0.010, 0.011],
}

aer_vals = {
    'qwen-4B-instruct':          [np.nan, 0.464, 0.520, 0.580],
    'Meta-Llama-3-8B-Instruct':  [np.nan, 0.145, 0.470, 0.791],
    'gemini-3.1-flash-lite':     [np.nan, 0.273, 0.285, 0.509],
    'gpt-5-mini':                [np.nan, 0.262, 0.499, 0.656],
    'gpt-4o-mini':               [np.nan, 0.465, 0.539, 0.651],
}
aer_errs = {m: [np.nan, np.nan, np.nan, np.nan] for m in models}

ece_vals = {
    'qwen-4B-instruct':          [np.nan, 0.6446, 0.6297, 0.6127],
    'Meta-Llama-3-8B-Instruct':  [np.nan, 0.6646, 0.6175, 0.4627],
    'gemini-3.1-flash-lite':     [np.nan, 0.3708, 0.3735, 0.3078],
    'gpt-5-mini':                [np.nan, 0.1449, 0.1506, 0.1503],
    'gpt-4o-mini':               [np.nan, 0.4257, 0.3958, 0.3604],
}
ece_errs = {m: [np.nan, np.nan, np.nan, np.nan] for m in models}

brier_vals = {
    'qwen-4B-instruct':          [np.nan, 0.6279, 0.6059, 0.5933],
    'Meta-Llama-3-8B-Instruct':  [np.nan, 0.6583, 0.6131, 0.4617],
    'gemini-3.1-flash-lite':     [np.nan, 0.3673, 0.3706, 0.3080],
    'gpt-5-mini':                [np.nan, 0.1999, 0.2102, 0.2061],
    'gpt-4o-mini':               [np.nan, 0.4163, 0.3906, 0.3597],
}
brier_errs = {
    'qwen-4B-instruct':          [np.nan, 0.0104, 0.0110, 0.0116],
    'Meta-Llama-3-8B-Instruct':  [np.nan, 0.0083, 0.0100, 0.0141],
    'gemini-3.1-flash-lite':     [np.nan, 0.0086, 0.0087, 0.0090],
    'gpt-5-mini':                [np.nan, 0.0072, 0.0082, 0.009 ],
    'gpt-4o-mini':               [np.nan, 0.0099, 0.0102, 0.011 ],
}

metrics = [
    ('FAR (Answered)', far_vals,   far_errs),
    ('AER',              aer_vals,   aer_errs),
    ('ECE (Answered)',   ece_vals,   ece_errs),
    ('Brier Score (Answered)', brier_vals, brier_errs),
]

colors  = ['#003F5C', '#2F7EB1', '#5BB8D4', '#A8DDE9']
x       = np.arange(len(models))
n       = len(schemes)
width   = 0.16
offsets = np.linspace(-(n-1)/2, (n-1)/2, n) * (width + 0.01)

fig, axes = plt.subplots(2, 2, figsize=(18, 6))

for ax, (title, vals, errs) in zip(axes.flat, metrics):
    for i, (scheme, color) in enumerate(zip(schemes, colors)):
        v  = [vals[m][i] for m in models]
        e  = [errs[m][i] for m in models]

        v_plot = [vi if not np.isnan(vi) else 0 for vi in v]
        e_plot = [ei if not np.isnan(ei) else 0 for ei in e]
        mask   = [not np.isnan(vi) for vi in v]

        bars = ax.bar(
            x + offsets[i], v_plot, width,
            color=color, alpha=0.85, label=scheme, zorder=3,
        )
        for j, (bar, has_data, vi, ei) in enumerate(zip(bars, mask, v_plot, e_plot)):
            if not has_data:
                bar.set_height(0)
                bar.set_visible(False)
            else:
                if ei > 0:
                    ax.errorbar(
                        bar.get_x() + bar.get_width()/2, vi,
                        yerr=ei, fmt='none', ecolor='#333', elinewidth=1.2, capsize=3, capthick=1.2, zorder=4
                    )

    ax.set_title(title, fontsize=17, fontweight='bold', pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, fontsize=14)
    ax.tick_params(axis='y', labelsize=16)
    ax.yaxis.grid(True, linestyle='--', alpha=0.45, zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    all_v = [vals[m][i] for m in models for i in range(n) if not np.isnan(vals[m][i])]
    ax.set_ylim(0, max(all_v) * 1.22)

handles = [mpatches.Patch(color=c, label=s, alpha=0.85) for c, s in zip(colors, schemes)]
fig.legend(handles=handles, title='Scheme', fontsize=14, title_fontsize=15,
           loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.1),
           framealpha=0.9, edgecolor='#ccc')

plt.tight_layout()
save_dir = os.path.dirname(os.path.abspath(__file__))
plt.savefig(os.path.join(save_dir, 'model_comparison.png'), dpi=150, bbox_inches='tight')
print("Saved.")