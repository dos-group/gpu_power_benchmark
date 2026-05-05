#!/usr/bin/env python
"""External vs internal GPU power, by hardware (validation, appendix).

Exports: results/external_vs_internal_power.pdf
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import statsmodels.api as sm

from _style import (
    HW_CONFIGS, HW_COLORS, EXTERNAL_VALIDATED_HW,
    FONT_LABEL, FONT_TICK, FONT_FACET,
    SINGLE_COL_W, RESULTS_DIR,
)

PDF_OUT = RESULTS_DIR / "external_vs_internal_power.pdf"

# ------------------------------------------------------------------
# Axis labels  ← edit here
# ------------------------------------------------------------------
X_LABEL = "GPU-reported power (W)"
Y_LABEL = "External meter power (W)"

# ------------------------------------------------------------------
# Axis ranges  ← edit here
# ------------------------------------------------------------------
X_RANGE = [None, 350]
Y_RANGE = [200, None]

# ------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------
dfs = []
for agg_path, _, hw_name in HW_CONFIGS:
    if hw_name not in EXTERNAL_VALIDATED_HW:
        continue  # Only GPUs with real external-meter readings belong here.
    try:
        df = pd.read_csv(agg_path)
        if "model_name" in df.columns:
            df = df.query('model_name != "baseline"')
        df["hardware"] = hw_name
        dfs.append(df)
    except FileNotFoundError as e:
        print(f"  Skipping {hw_name}: {e}")

if not dfs:
    raise SystemExit("No data loaded – check HW_CONFIGS paths.")

df_all = pd.concat(dfs, ignore_index=True)

# ------------------------------------------------------------------
# Build figure
# ------------------------------------------------------------------
needed = {"power_draw_watts_mean", "power_meter_active_power_w_mean"}
if not needed.issubset(df_all.columns):
    raise SystemExit(f"Missing columns: {needed - set(df_all.columns)}")

subset = df_all.dropna(subset=list(needed))

symbol_col = "dtype" if "dtype" in subset.columns else None

fig = px.scatter(
    subset,
    x="power_draw_watts_mean",
    y="power_meter_active_power_w_mean",
    color="hardware",
    symbol=symbol_col,
    color_discrete_map=HW_COLORS,
    facet_col="hardware",
    facet_col_wrap=2,
    trendline="ols",
    labels={
        "power_draw_watts_mean": X_LABEL,
        "power_meter_active_power_w_mean": Y_LABEL,
        "hardware": "Hardware",
    },
)

# Identity (y = x) dashed line in every facet
mn = float(min(subset["power_draw_watts_mean"].min(),
               subset["power_meter_active_power_w_mean"].min()))
mx = float(max(subset["power_draw_watts_mean"].max(),
               subset["power_meter_active_power_w_mean"].max()))
for xax in [a for a in fig.layout if a.startswith("xaxis")]:
    sfx = xax.replace("xaxis", "")
    fig.add_shape(
        type="line", x0=mn, y0=mn, x1=mx, y1=mx,
        xref="x" + sfx, yref="y" + sfx,
        line=dict(color="black", dash="dash", width=1),
        layer="below",
    )

# Remove "Hardware=" prefix from facet labels
fig.for_each_annotation(lambda a: a.update(
    text=a.text.replace("hardware=", "").replace("Hardware=", ""),
    font=dict(size=FONT_FACET),
))

# Axis range
fig.update_xaxes(range=X_RANGE)
fig.update_yaxes(range=Y_RANGE)

# Typography & layout
fig.update_xaxes(title_font=dict(size=FONT_LABEL), tickfont=dict(size=FONT_TICK))
fig.update_yaxes(title_font=dict(size=FONT_LABEL), tickfont=dict(size=FONT_TICK))

fig.update_layout(
    template="simple_white",
    width=SINGLE_COL_W,
    height=320,
    margin=dict(l=80, r=20, t=40, b=70),
    showlegend=False,
    font=dict(size=FONT_TICK),
    title=None,
)

# ------------------------------------------------------------------
# Export
# ------------------------------------------------------------------
pio.write_image(fig, PDF_OUT, format="pdf", scale=2)
print(f"PDF saved → {PDF_OUT.resolve()}")