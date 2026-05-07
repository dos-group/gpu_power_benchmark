"""External meter vs internal GPU power for the externally validated GPUs."""

import seaborn as sns
import matplotlib.pyplot as plt

from data import load_all_data, EXTERNAL_VALIDATED_HW
from figures.style import (
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
    figsize=(SINGLE_COL_W, 1.7),
    sharey=True, constrained_layout=True,
)
if len(hws) == 1:
    axes = [axes]

for ax, hw in zip(axes, hws):
    d = subset[subset["hardware"] == hw]
    sns.regplot(
        data=d, x=COLS[0], y=COLS[1], ax=ax,
        scatter_kws={"alpha": 0.5, "s": 6},
        line_kws={"lw": 1.0},
        color=HW_COLORS[hw],
    )
    ax.set_title(label(hw, sep=" "))
    ax.set_xlabel("Internal GPU power (W)")

    mn = float(d[COLS].min().min())
    mx = float(d[COLS].max().max())
    ax.plot([mn, mx], [mn, mx], "k--", lw=0.6, alpha=0.5)

axes[0].set_ylabel("External meter power (W)")
for ax in axes[1:]:
    ax.set_ylabel("")

fig.savefig(RESULTS_DIR / "external_vs_internal_power.pdf", bbox_inches="tight")
print(f"PDF saved -> {RESULTS_DIR / 'external_vs_internal_power.pdf'}")
