"""MFU and GPU Utilization vs internal GPU power, one row per GPU.

Single-column layout: rows = hardware, cols = predictor (MFU, GPU Util).
Y-axis shared per row, x-axis shared per column.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

from data import load_all_data, UTIL_EXCLUDE_HW
from create_figures.style import (
    HW_ORDER, DTYPE_COLORS, SINGLE_COL_W2, ROW_H, RESULTS_DIR,
    label, set_paper_style,
)

set_paper_style()
df_agg, _ = load_all_data()

PREDICTORS = [
    ("mfu_percentage_calflops_mean", "MFU (%)",          set()),
    ("gpu_utilization_mean",         "GPU Utilization (%)", UTIL_EXCLUDE_HW),
]
TARGET = "power_draw_watts_mean"

fig, axes = plt.subplots(
    len(HW_ORDER), 2,
    figsize=(SINGLE_COL_W2, 9),
    sharey="row", sharex="col",
    constrained_layout=True,
)

for i, hw in enumerate(HW_ORDER):
    g_hw = df_agg[df_agg["hardware"] == hw]
    for j, (pred, _title, exclude) in enumerate(PREDICTORS):
        ax = axes[i, j]
        if g_hw.empty:
            continue
        if hw in exclude:
            ax.text(0.5, 0.55, "Not available", transform=ax.transAxes,
                    ha="center", va="center", color="gray", fontstyle="italic")
            continue
        for dt, g in g_hw.groupby("dtype"):
            d = g.dropna(subset=[pred, TARGET])
            if d.empty:
                continue
            color = DTYPE_COLORS.get(dt, "#555")
            ax.scatter(d[pred], d[TARGET], s=6, alpha=0.6, color=color, label=dt)
            if len(d) >= 3 and d[pred].nunique() >= 2:
                fit = sm.OLS(d[TARGET], sm.add_constant(d[[pred]])).fit()
                xs = np.linspace(d[pred].min(), d[pred].max(), 20)
                ax.plot(xs, fit.predict(sm.add_constant(xs)), color=color, lw=1)
        ax.set_ylim(bottom=0)
    axes[i, 0].set_ylabel(label(hw), rotation=90, ha="center", va="center",
                          labelpad=12)

for j, (_p, title, _e) in enumerate(PREDICTORS):
    axes[0, j].set_title(title, pad=2)
for j, (_p, title, _e) in enumerate(PREDICTORS):
    axes[-1, j].set_xlabel(title)

axes[0, 0].set_xlim((-2,47))
fig.supylabel("GPU Power (W)", x=-0.04, fontsize=9)

handles, labels_ = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels_, title="dtype", loc="lower center", ncol=len(DTYPE_COLORS),
           bbox_to_anchor=(0.5, 1.01), frameon=False,
           handletextpad=0.3, columnspacing=0.8, borderpad=0)

sns.despine(fig)

fig.savefig(RESULTS_DIR / "predictor_vs_power.pdf", bbox_inches="tight")
print(f"PDF saved -> {RESULTS_DIR / 'predictor_vs_power.pdf'}")
