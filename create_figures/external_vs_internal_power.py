"""External meter vs internal GPU power for the externally validated GPUs."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import seaborn as sns
import matplotlib.pyplot as plt

from data import load_all_data, EXTERNAL_VALIDATED_HW
from create_figures.style import (
    HW_COLORS, SINGLE_COL_W, RESULTS_DIR,
    label, set_paper_style,
)

set_paper_style()
df_agg, _ = load_all_data()

COLS = ["power_draw_watts_mean", "power_meter_active_power_w_mean"]
subset = (df_agg[df_agg["hardware"].isin(EXTERNAL_VALIDATED_HW)]
          .dropna(subset=COLS))

hws = [hw for hw in EXTERNAL_VALIDATED_HW if hw in subset["hardware"].unique()]
fig, axes = plt.subplots(
    1, len(hws),
    figsize=(SINGLE_COL_W, 2),
    sharey=True, constrained_layout=True,
)

for ax, hw in zip(axes, hws):
    d = subset[subset["hardware"] == hw]
    sns.regplot(
        data=d, x=COLS[0], y=COLS[1], ax=ax,
        scatter_kws={"alpha": 0.5, "s": 6},
        line_kws={"lw": 1.0},
        color=HW_COLORS[hw],
    )
    ax.set_title(label(hw, sep=" "))
    ax.set_xlabel("GPU-reported power (W)")

    ax.set_xlim((0,350))
    ax.set_ylim((0,520))

    mn = float(d[COLS].min().min())
    mx = float(d[COLS].max().max())
    ax.plot([0, 1000], [0, 1000], "k--", lw=0.6, alpha=0.5)

axes[0].set_ylabel("External meter power (W)")
for ax in axes[1:]:
    ax.set_ylabel("")

sns.despine(fig)

fig.savefig(RESULTS_DIR / "external_vs_internal_power.pdf", bbox_inches="tight")
print(f"PDF saved -> {RESULTS_DIR / 'external_vs_internal_power.pdf'}")
