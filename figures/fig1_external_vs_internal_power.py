#!/usr/bin/env python
"""
Figure 1 – External vs Internal Power by hardware.
Exports: results/fig1_external_vs_internal_power.pdf
"""

from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import statsmodels.api as sm

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
HW_CONFIGS = [
    ("../aggregation_results/mfu_aggregated_per_config_A100.csv",  "../benchmark_results/mfu_benchmark_results_A100_128.csv",  "NVIDIA A100"),
    ("../aggregation_results/mfu_aggregated_per_config_L40.csv",   "../benchmark_results/mfu_benchmark_results_L40_128.csv",   "NVIDIA L40"),
    ("../aggregation_results/mfu_aggregated_per_config_GPU06.csv", "../benchmark_results/mfu_benchmark_results_GPU06_128.csv", "Quadro 5000"),
    ("../aggregation_results/mfu_aggregated_per_config_4070.csv",  "../benchmark_results/mfu_benchmark_results_4070_128.csv",  "RTX 4070 Ti"),
]

PDF_OUT = Path("../results/fig1_external_vs_internal_power.pdf")
PDF_OUT.parent.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# Typography  ← edit here to restyle the whole figure
# ------------------------------------------------------------------
FONT_LABEL  = 20   # axis labels (same size as body text in the paper)
FONT_TICK   = 14    # tick labels (slightly smaller, as suggested)
FONT_FACET  = 20   # facet / panel titles

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
    color="model_name",
    symbol=symbol_col,
    facet_col="hardware",
    facet_col_wrap=2,
    trendline="ols",
    labels={
        "power_draw_watts_mean": X_LABEL,
        "power_meter_active_power_w_mean": Y_LABEL,
        "model_name": "Model",
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
    width=900,
    height=300,
    margin=dict(l=90, r=40, t=40, b=80),
    showlegend=False,
    font=dict(size=FONT_TICK),
    title=None,
)

# ------------------------------------------------------------------
# Export
# ------------------------------------------------------------------
pio.write_image(fig, PDF_OUT, format="pdf", scale=2)
print(f"PDF saved → {PDF_OUT.resolve()}")