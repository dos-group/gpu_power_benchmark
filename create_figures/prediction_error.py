"""Per-sample prediction error of MFU vs GPU Utilisation, by hardware.

Two rows. Top: pooled per-GPU OLS model error. Bottom: (dtype, batch)-cell
OLS error -- the same predictor conditioned on the two regime variables.
Sample assignments to cells come from the aggregated per-config metadata.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt

from data import load_all_data, UTIL_EXCLUDE_HW
from create_figures.style import (
    HW_ORDER, PREDICTOR_COLORS, SINGLE_COL_W, RESULTS_DIR,
    label, set_paper_style,
)

set_paper_style()
_, df_raw = load_all_data()

PREDS = [("mfu_percentage_calflops", "MFU"),
         ("gpu_utilization",         "GPU Utilization")]
TARGET = "power_draw_watts"
GROUP_COLS = ["dtype", "batch_size"]
MIN_CELL = 5


def fit_errors(g: pd.DataFrame, pred: str) -> pd.Series:
    d = g.dropna(subset=[pred, TARGET])
    if len(d) < 5 or d[pred].nunique() < 2:
        return pd.Series(dtype=float)
    fit = sm.OLS(d[TARGET], sm.add_constant(d[[pred]])).fit()
    return (((d[TARGET] - fit.predict()).abs() / d[TARGET].replace(0, np.nan))
            * 100).dropna()


rows = []
for hw, g in df_raw.groupby("hardware"):
    for col, lbl in PREDS:
        if lbl == "GPU Utilization" and hw in UTIL_EXCLUDE_HW:
            continue
        # Pooled per-GPU fit.
        for e in fit_errors(g, col):
            rows.append({"hardware": hw, "Predictor": lbl,
                         "Conditioning": "pooled", "error": e})
        # (dtype, batch)-cell fits.
        for _, cell in g.groupby(GROUP_COLS):
            if len(cell) < MIN_CELL:
                continue
            for e in fit_errors(cell, col):
                rows.append({"hardware": hw, "Predictor": lbl,
                             "Conditioning": "+ dtype, batch", "error": e})

df_err = pd.DataFrame(rows)
df_err["hardware_label"] = df_err["hardware"].apply(label)
hw_order_labels = [label(hw) for hw in HW_ORDER]

fig, axes = plt.subplots(2, 1, figsize=(SINGLE_COL_W, 2.8),
                         sharex=True, constrained_layout=True)

COND_LABELS = {
    "pooled":         "conditioned on\nGPU",
    "+ dtype, batch": "conditioned on\n(GPU, dtype, batch size)",
}

for ax, cond in zip(axes, ["pooled", "+ dtype, batch"]):
    sub = df_err[df_err["Conditioning"] == cond]
    sns.violinplot(
        data=sub, x="hardware_label", y="error",
        hue="Predictor", palette=PREDICTOR_COLORS,
        order=hw_order_labels,
        inner="box", linewidth=0.6, cut=0, ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel(COND_LABELS[cond])
    if ax is axes[0]:
        ax.set_ylim(0, 80)
        ax.legend(title=None, loc="upper left", frameon=False)
    else:
        ax.set_ylim(0, 80)
        if ax.get_legend() is not None:
            ax.get_legend().remove()

fig.supylabel("Relative prediction error (%)", x=-0.05, fontsize=9)
sns.despine(fig)
fig.savefig(RESULTS_DIR / "prediction_error.pdf", bbox_inches="tight")
print(f"PDF saved -> {RESULTS_DIR / 'prediction_error.pdf'}")
