"""MFU and GPU Utilization as a function of batch size, per GPU.

Visualizes the memory-bound -> compute-bound transition. At small batch
(low arithmetic intensity), MFU collapses while GPU Utilization can remain
high; at large batch both saturate. Makes the regime story visual and shows
the MI210 binary-util pathology in a single panel.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from data import load_all_data
from create_figures.style import (
    HW_ORDER, HW_COLORS, SINGLE_COL_W, RESULTS_DIR,
    label, set_paper_style,
)

set_paper_style()
df_agg, _ = load_all_data()

MFU = "mfu_percentage_calflops_mean"
UTIL = "gpu_utilization_mean"

d = df_agg.dropna(subset=["batch_size"]).copy()

fig, axes = plt.subplots(1, 2, figsize=(SINGLE_COL_W, 2),
                         sharex=True, gridspec_kw={"wspace": 0.3})

for ax, col, title in [(axes[0], MFU, "MFU (%)"),
                       (axes[1], UTIL, "GPU Utilization (%)")]:
    for hw in HW_ORDER:
        g = d[d["hardware"] == hw]
        if g.empty:
            continue
        s = (g.dropna(subset=[col])
               .groupby("batch_size")[col]
               .agg(["mean", "std", "count"])
               .reset_index())
        if s.empty:
            continue
        c = HW_COLORS[hw]
        ls = "--" if hw == "AMD<br>MI210" else "-"
        ax.plot(s["batch_size"], s["mean"], marker="o", ms=3,
                color=c, lw=1.0, linestyle=ls, label=label(hw, sep=" "))
        sem = s["std"] / np.sqrt(s["count"].clip(lower=1))
        ax.fill_between(s["batch_size"], s["mean"] - sem, s["mean"] + sem,
                        color=c, alpha=0.15, linewidth=0)
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 4, 8, 16, 32, 64, 128])
    ax.set_xticklabels([1, 4, 8, 16, 32, 64, 128])
    ax.set_xlabel("Batch size")
    ax.set_ylabel(title)
    ax.set_ylim(bottom=0)

axes[0].legend(loc="lower left", frameon=False, # fontsize=6, handletextpad=0.3, labelspacing=0.2, borderpad=0.2,
               bbox_to_anchor=(-0.05, 1.05),
               ncols=2)

sns.despine(fig)
fig.savefig(RESULTS_DIR / "predictors_vs_batch.pdf", bbox_inches="tight")
print(f"PDF saved -> {RESULTS_DIR / 'predictors_vs_batch.pdf'}")
