#!/usr/bin/env python
"""
Figure 2 – Prediction error of MFU vs GPU Utilisation for internal power.
Exports: Euro-Par Paper/pdf/fig2_prediction_error.pdf
"""

from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import statsmodels.api as sm

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
HW_CONFIGS = [
    ("./aggregation_128/mfu_aggregated_per_config_A100.csv",  "./mfu_benchmark_results_A100_128.csv",  "NVIDIA A100"),
    ("./aggregation_128/mfu_aggregated_per_config_L40.csv",   "./mfu_benchmark_results_L40_128.csv",   "NVIDIA L40"),
    ("./aggregation_128/mfu_aggregated_per_config_GPU06.csv", "./mfu_benchmark_results_GPU06_128.csv", "Quadro 5000"),
    ("./aggregation_128/mfu_aggregated_per_config_4070.csv",  "./mfu_benchmark_results_4070_128.csv",  "RTX 4070 Ti"),
]

PDF_OUT = Path("Euro-Par Paper/pdf/fig2_prediction_error.pdf")
PDF_OUT.parent.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# Typography  ← edit here to restyle the whole figure
# ------------------------------------------------------------------
FONT_LABEL  = 20   # axis labels
FONT_TICK   = 14    # tick labels
FONT_FACET  = 20   # facet / panel titles

# ------------------------------------------------------------------
# Axis labels  ← edit here ("" to hide)
# ------------------------------------------------------------------
X_LABEL = ""                                          # supervisor: x-label can go
Y_LABEL = "Absolute error in internal power pred. (%)"

# ------------------------------------------------------------------
# Colors for the two predictors  ← edit here
# ------------------------------------------------------------------
COLOR_MAP = {
    "MFU (%)":      "steelblue",
    "GPU Util (%)": "darkorange",
}

# ------------------------------------------------------------------
# Load raw data
# ------------------------------------------------------------------
dfs_raw = []
for _, raw_path, hw_name in HW_CONFIGS:
    try:
        df = pd.read_csv(raw_path)
        if "model_name" in df.columns:
            df = df.query('model_name != "baseline"')
        df["hardware"] = hw_name
        dfs_raw.append(df)
    except FileNotFoundError as e:
        print(f"  Skipping {hw_name}: {e}")

if not dfs_raw:
    raise SystemExit("No data loaded – check HW_CONFIGS paths.")

df_raw_all = pd.concat(dfs_raw, ignore_index=True)

# ------------------------------------------------------------------
# Compute per-row OLS residuals
# ------------------------------------------------------------------
needed = {"hardware", "power_draw_watts", "mfu_percentage_calflops", "gpu_utilization"}
if not needed.issubset(df_raw_all.columns):
    raise SystemExit(f"Missing columns: {needed - set(df_raw_all.columns)}")

rows = []
for hw, g in df_raw_all.groupby("hardware"):
    for pred_col, pred_label in [
        ("mfu_percentage_calflops", "MFU (%)"),
        ("gpu_utilization",         "GPU Util (%)"),
    ]:
        if pred_col not in g.columns:
            continue
        g_sub = g.dropna(subset=[pred_col, "power_draw_watts"]).copy()
        if g_sub[pred_col].nunique() < 2 or len(g_sub) < 5:
            continue
        X = sm.add_constant(g_sub[[pred_col]], has_constant="add")
        try:
            fit   = sm.OLS(g_sub["power_draw_watts"], X).fit()
            y_hat = fit.predict(X)
            abs_e = (g_sub["power_draw_watts"] - y_hat).abs()
            rel_e = abs_e / g_sub["power_draw_watts"].replace(0, np.nan)
            for (_, row), ae, re in zip(g_sub.iterrows(), abs_e, rel_e):
                if np.isfinite(re):
                    rows.append({
                        "hardware":       hw,
                        "Predictor":      pred_label,
                        "abs_error_w":    float(ae),
                        "rel_error_pct":  float(re * 100),
                        "model_name":     row.get("model_name"),
                        "dtype":          row.get("dtype"),
                        "batch_size":     row.get("batch_size"),
                    })
        except Exception as exc:
            print(f"  OLS failed for {hw}/{pred_label}: {exc}")

pred_err_df = pd.DataFrame(rows)
if pred_err_df.empty:
    raise SystemExit("No prediction-error data computed.")

# ------------------------------------------------------------------
# Build figure
# ------------------------------------------------------------------
hover_cols = [c for c in ["model_name", "dtype", "batch_size"] if c in pred_err_df.columns]

fig = px.violin(
    pred_err_df,
    x="Predictor",
    y="rel_error_pct",
    color="Predictor",
    facet_col="hardware",
    facet_col_wrap=2,
    box=True,
    points="outliers",
    color_discrete_map=COLOR_MAP,
    hover_data=hover_cols,
    labels={
        "Predictor":     X_LABEL,
        "rel_error_pct": Y_LABEL,
        "hardware":      "",
    },
)

# Clip y-axis at 99th percentile to suppress extreme outliers
y99 = pred_err_df["rel_error_pct"].quantile(0.99)
fig.update_yaxes(range=[0, y99])

# Remove "hardware=" / "Hardware=" prefix and any leading "=" from facet labels
def clean_facet_label(a):
    t = a.text
    for prefix in ("hardware=", "Hardware="):
        t = t.replace(prefix, "")
    t = t.lstrip("=")
    a.update(text=t, font=dict(size=FONT_FACET))

fig.for_each_annotation(clean_facet_label)

# Hide x-axis title (supervisor request) but keep tick labels
fig.update_xaxes(title_text=X_LABEL)

# Global y-axis label via annotation (single label for all facets)
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
    height=550,                # enough vertical space to read facet labels clearly
    margin=dict(l=100, r=40, t=40, b=80),
    showlegend=False,
    font=dict(size=FONT_TICK),
    title=None,
)

# ------------------------------------------------------------------
# Export
# ------------------------------------------------------------------
pio.write_image(fig, PDF_OUT, format="pdf", scale=2)
print(f"PDF saved → {PDF_OUT.resolve()}")