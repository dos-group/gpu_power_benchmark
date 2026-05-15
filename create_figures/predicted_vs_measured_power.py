"""Predicted vs. measured power, one panel per GPU.

Two columns. Left: per-GPU pooled linear MFU->power model -- the batch=1 cluster
falls off the diagonal because the memory-bound regime breaks the single-slope
fit. Right: per-(GPU, dtype, batch) cell models -- conditioning on the two
regime variables tightens predictions onto the diagonal.

Color encodes batch size (log scale); marker shape encodes dtype.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from data import load_all_data
from create_figures.style import (
    HW_ORDER, SINGLE_COL_W2, RESULTS_DIR, label, set_paper_style,
)

set_paper_style()
df_agg, _ = load_all_data()

PRED = "mfu_percentage_calflops_mean"
TARGET = "power_draw_watts_mean"
DTYPE_MARKERS = {"float32": "o", "float16": "s", "bfloat16": "^"}
MIN_CELL_N = 3

d = df_agg.dropna(subset=[PRED, TARGET, "batch_size", "dtype"]).copy()
d = d[(d[PRED] >= 0) & (d[TARGET] > 0)]

fig, axes = plt.subplots(
    len(HW_ORDER), 2,
    figsize=(SINGLE_COL_W2, 9),
    constrained_layout=True,
    sharey="row",
)

batch_norm = LogNorm(vmin=1, vmax=128)
cmap = plt.get_cmap("viridis_r")


def fit_and_predict(g_in: pd.DataFrame, mode: str) -> pd.Series:
    """Return predicted power. mode='pooled' fits one OLS per GPU,
    mode='cell' fits one per (dtype, batch_size) cell with a fall-back."""
    pred = pd.Series(np.nan, index=g_in.index, dtype=float)
    if mode == "pooled":
        if g_in[PRED].nunique() < 2:
            return pred
        fit = sm.OLS(g_in[TARGET].values,
                     sm.add_constant(g_in[[PRED]].values, has_constant="add")).fit()
        pred[:] = fit.predict(sm.add_constant(g_in[[PRED]].values, has_constant="add"))
        return pred
    # cell-conditioned
    for (dt, bs), cell in g_in.groupby(["dtype", "batch_size"]):
        if len(cell) >= MIN_CELL_N and cell[PRED].nunique() >= 2:
            fit = sm.OLS(cell[TARGET].values,
                         sm.add_constant(cell[[PRED]].values, has_constant="add")).fit()
            pred.loc[cell.index] = fit.predict(
                sm.add_constant(cell[[PRED]].values, has_constant="add"))
        else:
            # too few points / no variance -> use the cell mean of measured power.
            pred.loc[cell.index] = cell[TARGET].mean()
    return pred


for col, (mode, col_title) in enumerate([
    ("pooled", "conditioned on\nGPU"),
    ("cell",   "conditioned on\n(GPU, dtype, batch size)"),
]):
    axes[0, col].set_title(col_title, pad=2)

for row, hw in enumerate(HW_ORDER):
    for col, mode in enumerate(["pooled", "cell"]):
        ax = axes[row, col]
        g = d[d["hardware"] == hw]
        if g.empty:
            ax.set_xticks([]); ax.set_yticks([])
            continue
        yhat = fit_and_predict(g, mode)
        ok = yhat.notna()
        gg = g[ok]
        yh = yhat[ok]

        for dt, mk in DTYPE_MARKERS.items():
            sub = gg[gg["dtype"] == dt]
            if sub.empty:
                continue
            ax.scatter(sub[TARGET], yh.loc[sub.index],
                       c=sub["batch_size"], cmap=cmap, norm=batch_norm,
                       marker=mk, s=10, alpha=0.8, linewidths=0)

        lo = min(gg[TARGET].min(), yh.min())
        hi = max(gg[TARGET].max(), yh.max())
        pad = 0.05 * (hi - lo)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
                color="0.3", lw=0.7, ls="--")
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        if row == len(HW_ORDER) - 1:
            ax.set_xlabel("GPU-reported power (W)")
        if col == 0:
            ax.set_ylabel(label(hw, sep="\n"))

fig.supylabel("Predicted power (W)", x=-0.04, fontsize=9)

# Batch-size legend (top-left): discrete proxy artists coloured from the cmap.
bs_handles = [plt.Line2D([0], [0], marker="o", ls="", ms=5,
                          markerfacecolor=cmap(batch_norm(bs)),
                          markeredgecolor="none", label=str(bs))
              for bs in [1, 4, 16, 64, 128]]
fig.legend(handles=bs_handles, title="Batch size",
           loc="lower left", bbox_to_anchor=(0.05, 1.01),
           ncol=len(bs_handles), frameon=False,
           handletextpad=0.3, columnspacing=0.5, borderpad=0)

# Dtype marker legend (top-right).
dt_handles = [plt.Line2D([0], [0], marker=m, linestyle="",
                         markerfacecolor="0.4", markeredgecolor="none",
                         markersize=5, label=dt)
              for dt, m in DTYPE_MARKERS.items()]
fig.legend(handles=dt_handles, title="dtype",
           loc="lower right", bbox_to_anchor=(0.98, 1.01),
           ncol=len(dt_handles), frameon=False,
           handletextpad=0.3, columnspacing=0.5, borderpad=0)

sns.despine(fig)

fig.savefig(RESULTS_DIR / "predicted_vs_measured_power.pdf", bbox_inches="tight")
print(f"PDF saved -> {RESULTS_DIR / 'predicted_vs_measured_power.pdf'}")
