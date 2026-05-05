#!/usr/bin/env python
"""Per-sample prediction error for MFU vs GPU Utilisation (boxplots).

Exports: results/prediction_error.pdf
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import statsmodels.api as sm

from figures.style import (
    PREDICTOR_COLORS, FONT_LABEL, FONT_TICK, FONT_FACET,
    DOUBLE_COL_W, RESULTS_DIR,
)
from analysis.data_loader import load_all_data, UTIL_EXCLUDE_HW

# Hardware order on the x-axis
HW_ORDER = [
    "NVIDIA A100",
    "NVIDIA L40",
    "NVIDIA L4",
    "Quadro 5000",
    "RTX 4070 Ti",
    "AMD MI210",
]

PDF_OUT = RESULTS_DIR / "prediction_error.pdf"

# ------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------
_, df_raw_all = load_all_data()

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
        # Skip GPU Util on hardware where the signal is not meaningful
        # (e.g. AMD MI210 GRBM_COUNT is binary).
        if pred_label == "GPU Util (%)" and hw in UTIL_EXCLUDE_HW:
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

# Pin facet order; missing hardware (e.g. L4 placeholder) stays as an
# empty panel because facet_col uses a categorical with the full order.
pred_err_df["hardware"] = pd.Categorical(
    pred_err_df["hardware"], categories=HW_ORDER, ordered=True,
)

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
    category_orders={"hardware": HW_ORDER},
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
    width=DOUBLE_COL_W,
    height=420,
    margin=dict(l=80, r=20, t=40, b=80),
    showlegend=False,
    font=dict(size=FONT_TICK),
    title=None,
)

# ------------------------------------------------------------------
# Export
# ------------------------------------------------------------------
pio.write_image(fig, PDF_OUT, format="pdf", scale=2)
print(f"PDF saved → {PDF_OUT.resolve()}")