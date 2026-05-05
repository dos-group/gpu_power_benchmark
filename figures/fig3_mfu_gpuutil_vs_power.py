#!/usr/bin/env python
"""
Figure 3 – MFU and GPU Utilisation vs internal power across dtypes.
Exports: results/fig3_mfu_gpuutil_vs_power.pdf
"""

from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.io as pio

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
HW_CONFIGS = [
    ("../aggregation_results/mfu_aggregated_per_config_A100.csv",  "../benchmark_results/mfu_benchmark_results_A100_128.csv",  "NVIDIA A100"),
    ("../aggregation_results/mfu_aggregated_per_config_L40.csv",   "../benchmark_results/mfu_benchmark_results_L40_128.csv",   "NVIDIA L40"),
    ("../aggregation_results/mfu_aggregated_per_config_GPU06.csv", "../benchmark_results/mfu_benchmark_results_GPU06_128.csv", "Quadro 5000"),
    ("../aggregation_results/mfu_aggregated_per_config_4070.csv",  "../benchmark_results/mfu_benchmark_results_4070_128.csv",  "RTX 4070 Ti"),
]

PDF_OUT = Path("../results/fig3_mfu_gpuutil_vs_power.pdf")
PDF_OUT.parent.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# Typography  ← edit here to restyle the whole figure
# ------------------------------------------------------------------
FONT_LABEL  = 20   # axis labels
FONT_TICK   = 14    # tick labels
FONT_FACET  = 20   # facet / panel titles
FONT_LEGEND = 14   # legend entries

# ------------------------------------------------------------------
# Axis labels  ← edit here
# ------------------------------------------------------------------
X_LABEL = "Predictor value (%)"
Y_LABEL = "Internal GPU power (W)"

# ------------------------------------------------------------------
# Legend / predictor display names  ← edit here
# ------------------------------------------------------------------
PREDICTOR_MAP = {
    "mfu_percentage_calflops_mean": "MFU (%)",
    "gpu_utilization_mean":         "GPU Util (%)",
}

# ------------------------------------------------------------------
# Load aggregated data
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
# Reshape to long format
# ------------------------------------------------------------------
needed = {"mfu_percentage_calflops_mean", "gpu_utilization_mean",
          "power_draw_watts_mean", "hardware", "dtype"}
if not needed.issubset(df_all.columns):
    raise SystemExit(f"Missing columns: {needed - set(df_all.columns)}")

df_dt = df_all[list(needed)].dropna().copy()
df_dt["dtype"] = df_dt["dtype"].astype(str)

df_long = df_dt.melt(
    id_vars=["hardware", "dtype", "power_draw_watts_mean"],
    value_vars=list(PREDICTOR_MAP.keys()),
    var_name="Predictor", value_name="Predictor_value",
)
df_long["Predictor"] = df_long["Predictor"].map(PREDICTOR_MAP)

# ------------------------------------------------------------------
# Build figure
# ------------------------------------------------------------------
fig = px.scatter(
    df_long,
    x="Predictor_value",
    y="power_draw_watts_mean",
    color="dtype",
    symbol="Predictor",
    facet_col="hardware",
    facet_col_wrap=2,
    trendline="ols",
    labels={
        "Predictor_value":      X_LABEL,
        "power_draw_watts_mean": Y_LABEL,
        "dtype":                "Dtype",
        "Predictor":            "Predictor",
    },
)
fig.update_traces(marker=dict(size=6))

# Remove "hardware=" / "Hardware=" prefix and any leading "=" from facet labels
def clean_facet_label(a):
    t = a.text
    for prefix in ("hardware=", "Hardware="):
        t = t.replace(prefix, "")
    t = t.lstrip("=")
    a.update(text=t, font=dict(size=FONT_FACET))

fig.for_each_annotation(clean_facet_label)

# Global y-axis label via annotation
fig.for_each_yaxis(lambda ax: ax.update(title_text=""))
fig.add_annotation(
    x=-0.07, y=0.5, xref="paper", yref="paper",
    text=Y_LABEL, textangle=-90,
    showarrow=False, xanchor="center", yanchor="middle",
    font=dict(size=FONT_LABEL),
)

# Typography & layout
fig.update_xaxes(title_font=dict(size=FONT_LABEL), tickfont=dict(size=FONT_TICK))
fig.update_yaxes(title_font=dict(size=FONT_LABEL), tickfont=dict(size=FONT_TICK))

fig.update_layout(
    template="simple_white",
    width=900,
    height=600,                # enough vertical space for 2-row facets
    margin=dict(l=100, r=40, t=40, b=120),
    font=dict(size=FONT_TICK),
    title=None,
    legend=dict(
        orientation="h",
        x=0.5, xanchor="center",
        y=-0.15, yanchor="top",
        title_text="",
        font=dict(size=FONT_LEGEND),
        itemsizing="constant",
        # Force 3 entries per row → 3×2 grid for 6 legend items
        entrywidth=0.30,
        entrywidthmode="fraction",
    ),
)

# ------------------------------------------------------------------
# Export
# ------------------------------------------------------------------
pio.write_image(fig, PDF_OUT, format="pdf", scale=2)
print(f"PDF saved → {PDF_OUT.resolve()}")