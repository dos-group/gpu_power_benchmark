"""Per-sample prediction error of MFU vs GPU Utilisation, by hardware."""

import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt

from data import load_all_data, UTIL_EXCLUDE_HW
from figures.style import (
    HW_ORDER, PREDICTOR_COLORS, DOUBLE_COL_W, RESULTS_DIR,
    label, set_paper_style,
)

set_paper_style()
_, df_raw = load_all_data()

# Per-hardware OLS prediction-error (% of measured power).
rows = []
for hw, g in df_raw.groupby("hardware"):
    for col, lbl in [("mfu_percentage_calflops", "MFU (%)"),
                     ("gpu_utilization",         "GPU Util (%)")]:
        if lbl == "GPU Util (%)" and hw in UTIL_EXCLUDE_HW:
            continue
        d = g.dropna(subset=[col, "power_draw_watts"])
        if len(d) < 5 or d[col].nunique() < 2:
            continue
        fit = sm.OLS(d["power_draw_watts"], sm.add_constant(d[[col]])).fit()
        err = ((d["power_draw_watts"] - fit.predict()).abs()
               / d["power_draw_watts"].replace(0, np.nan)) * 100
        rows.extend({"hardware": hw, "Predictor": lbl, "error": e}
                    for e in err.dropna())
df_err = pd.DataFrame(rows)
ymax = float(df_err["error"].quantile(0.99))

fig, axes = plt.subplots(
    1, len(HW_ORDER),
    figsize=(DOUBLE_COL_W, 2.0),
    sharey=True, constrained_layout=True,
)
for ax, hw in zip(axes, HW_ORDER):
    sub = df_err[df_err["hardware"] == hw]
    ax.set_title(label(hw))
    ax.set_xlabel("")
    if not sub.empty:
        sns.violinplot(
            data=sub, x="Predictor", y="error",
            hue="Predictor", palette=PREDICTOR_COLORS, legend=False,
            inner="box", linewidth=0.6, cut=0, ax=ax,
        )
    else:
        ax.set_xticks([])
    ax.set_ylim(0, ymax)

axes[0].set_ylabel("Abs. error in internal\npower prediction (%)")
for ax in axes[1:]:
    ax.set_ylabel("")

fig.savefig(RESULTS_DIR / "prediction_error.pdf", bbox_inches="tight")
print(f"PDF saved -> {RESULTS_DIR / 'prediction_error.pdf'}")
