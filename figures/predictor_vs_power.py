#!/usr/bin/env python
"""MFU and GPU Utilisation vs internal GPU power.

Grid layout: one row per GPU, two columns (MFU left, GPU Util right).
- Y-axis is shared row-wise (each row's two panels share the same power scale,
  no tick labels on the right column).
- X-axis is shared column-wise (the MFU column has one common x-range across
  all rows, the GPU Util column has another; tick labels only on the
  bottom row).

Color encodes dtype; AMD MI210 row has the GPU Util panel left empty
because its ROCm GRBM_COUNT-derived signal is binary.

Exports: results/predictor_vs_power.pdf
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import statsmodels.api as sm
from plotly.subplots import make_subplots

from figures.style import (
    HW_COLORS, FONT_LABEL, FONT_TICK, FONT_FACET, FONT_LEGEND,
    SINGLE_COL_W, RESULTS_DIR,
)
from analysis.data_loader import load_all_data, UTIL_EXCLUDE_HW

PDF_OUT = RESULTS_DIR / "predictor_vs_power.pdf"

X_LABEL_MFU  = "MFU (%)"
X_LABEL_UTIL = "GPU Utilization (%)"
Y_LABEL      = "Internal GPU power (W)"

HW_ORDER = [
    "NVIDIA A100",
    "NVIDIA L40",
    "NVIDIA L4",
    "Quadro 5000",
    "RTX 4070 Ti",
    "AMD MI210",
]

DTYPE_COLORS = {
    "float32":  "#1f77b4",
    "float16":  "#ff7f0e",
    "bfloat16": "#2ca02c",
}

# ---------------------------------------------------------------- Load
df_all, _ = load_all_data()
needed = {"mfu_percentage_calflops_mean", "gpu_utilization_mean",
          "power_draw_watts_mean", "hardware", "dtype"}
if not needed.issubset(df_all.columns):
    raise SystemExit(f"Missing columns: {needed - set(df_all.columns)}")
df_all["dtype"] = df_all["dtype"].astype(str)

# ---------------------------------------------------------------- Build figure
n_rows = len(HW_ORDER)
fig = make_subplots(
    rows=n_rows, cols=2,
    shared_xaxes="columns",   # MFU x-axis shared down col 1, Util down col 2
    shared_yaxes="rows",       # power y-axis shared across each row
    horizontal_spacing=0.04,
    vertical_spacing=0.03,
    column_titles=("MFU", "GPU Utilization"),
    row_titles=HW_ORDER,
)

seen_dt: set[str] = set()

for i, hw in enumerate(HW_ORDER, start=1):
    g_hw = df_all[df_all["hardware"] == hw]

    for col_idx, (predictor_col, exclude) in enumerate(
        [("mfu_percentage_calflops_mean", set()),
         ("gpu_utilization_mean", UTIL_EXCLUDE_HW)],
        start=1,
    ):
        if hw in exclude:
            continue

        for dt, g in g_hw.groupby("dtype"):
            g = g.dropna(subset=[predictor_col, "power_draw_watts_mean"])
            if g.empty:
                continue
            color = DTYPE_COLORS.get(dt, "#555")
            show = dt not in seen_dt
            if show:
                seen_dt.add(dt)
            fig.add_trace(
                go.Scatter(
                    x=g[predictor_col],
                    y=g["power_draw_watts_mean"],
                    mode="markers",
                    name=dt,
                    legendgroup=f"dt_{dt}",
                    showlegend=show,
                    marker=dict(color=color, size=6, line=dict(width=0)),
                    hovertemplate=(f"{hw} / {dt}<br>"
                                   "x=%{x:.2f}<br>y=%{y:.1f} W<extra></extra>"),
                ),
                row=i, col=col_idx,
            )

        # Per-dtype OLS line, per (hw, predictor) panel.
        for dt, g in g_hw.groupby("dtype"):
            g = g.dropna(subset=[predictor_col, "power_draw_watts_mean"])
            if len(g) < 3 or g[predictor_col].nunique() < 2:
                continue
            color = DTYPE_COLORS.get(dt, "#555")
            X = sm.add_constant(g[[predictor_col]], has_constant="add")
            try:
                fit = sm.OLS(g["power_draw_watts_mean"], X).fit()
                xs = np.linspace(g[predictor_col].min(),
                                 g[predictor_col].max(), 50)
                ys = fit.params.iloc[0] + fit.params.iloc[1] * xs
                fig.add_trace(
                    go.Scatter(
                        x=xs, y=ys, mode="lines",
                        line=dict(color=color, width=1.5),
                        legendgroup=f"dt_{dt}",
                        showlegend=False,
                        hoverinfo="skip",
                    ),
                    row=i, col=col_idx,
                )
            except Exception:
                pass

# ---------------------------------------------------------------- Cosmetics
# X-axis titles only on bottom row.
fig.update_xaxes(title_text=X_LABEL_MFU,  row=n_rows, col=1,
                 title_font=dict(size=FONT_LABEL))
fig.update_xaxes(title_text=X_LABEL_UTIL, row=n_rows, col=2,
                 title_font=dict(size=FONT_LABEL))
fig.update_xaxes(tickfont=dict(size=FONT_TICK))

# Y-axis title only on left column, vertically centered via a global annotation.
fig.update_yaxes(tickfont=dict(size=FONT_TICK))
fig.update_yaxes(title_text="", showticklabels=True)
# Hide y tick labels on right column (shared with left in the same row).
for r in range(1, n_rows + 1):
    fig.update_yaxes(showticklabels=False, row=r, col=2)

fig.add_annotation(
    x=-0.10, y=0.5, xref="paper", yref="paper",
    text=Y_LABEL, textangle=-90,
    showarrow=False, xanchor="center", yanchor="middle",
    font=dict(size=FONT_LABEL),
)

# Column titles + row titles fonts
for ann in fig.layout.annotations:
    if ann.text in ("MFU", "GPU Utilization") or ann.text in HW_ORDER:
        ann.font = dict(size=FONT_FACET)

fig.update_layout(
    template="simple_white",
    width=SINGLE_COL_W,
    height=180 * n_rows + 80,
    margin=dict(l=90, r=60, t=40, b=80),
    legend=dict(
        orientation="h",
        x=0.5, xanchor="center",
        y=-0.05, yanchor="top",
        font=dict(size=FONT_LEGEND),
        title_text="",
        itemsizing="constant",
    ),
)

pio.write_image(fig, PDF_OUT, format="pdf", scale=2)
print(f"PDF saved -> {PDF_OUT.resolve()}")
