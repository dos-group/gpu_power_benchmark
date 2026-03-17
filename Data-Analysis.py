#!/usr/bin/env python
"""Cross-hardware energy predictor analysis.
Loads CSVs, produces three PDF figures, and saves an HTML dashboard.
"""

from __future__ import annotations
from pathlib import Path
from typing import Union, Tuple
import math
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.api as sm

PathLike = Union[str, Path]

# ----------------------------------------------------------
# Configuration
# ----------------------------------------------------------

HW_CONFIGS: list[tuple[PathLike, PathLike, str]] = [
    ("./aggregation_128/mfu_aggregated_per_config_A100.csv",  "./mfu_benchmark_results_A100_128.csv",  "NVIDIA A100"),
    ("./aggregation_128/mfu_aggregated_per_config_L40.csv",   "./mfu_benchmark_results_L40_128.csv",   "NVIDIA L40"),
    ("./aggregation_128/mfu_aggregated_per_config_GPU06.csv", "./mfu_benchmark_results_GPU06_128.csv", "Quadro 5000"),
    ("./aggregation_128/mfu_aggregated_per_config_4070.csv",  "./mfu_benchmark_results_4070_128.csv",  "RTX 4070 Ti"),
    ("./aggregation_128/mfu_aggregated_per_config_MI210.csv", "./mfu_benchmark_results_MI210_128.csv", "AMD MI210"),
]

OUTDIR  = Path("Euro-Par Paper")
OUTFILE = OUTDIR / "cross_hardware_energy_predictors_v2.html"
PDF_DIR = OUTDIR / "pdf"
OUTDIR.mkdir(exist_ok=True)
PDF_DIR.mkdir(exist_ok=True)

# ----------------------------------------------------------
# Chart style – edit only here to restyle all figures
# ----------------------------------------------------------

STYLE = dict(
    template       = "simple_white",
    width          = 900,
    height         = 550,         # taller figures use 750
    title_font     = 28,
    axis_title_font= 22,
    tick_font      = 14,
    facet_font     = 22,
    legend_font    = 14,
    base_font      = 12,
    marker_size    = 8,
    color_mfu      = "steelblue",
    color_util     = "darkorange",
)

MARGIN = dict(l=120, r=40, t=90, b=110)   # default; override per figure as needed

LEGEND_H = dict(               # horizontal legend below the plot
    orientation="h",
    x=0.5, xanchor="center",
    y=-0.22, yanchor="top",
    title_text="",
    entrywidth=170, entrywidthmode="pixels",
    itemsizing="constant",
    font=dict(size=STYLE["legend_font"]),
)

COLOR_MAP = {"MFU (%)": STYLE["color_mfu"], "GPU Util (%)": STYLE["color_util"]}


def apply_style(fig, *, title=None, height=None, margin=None, show_legend=True, global_ylabel=None):
    """Apply the global STYLE dict to any figure."""
    fig.update_layout(
        template   = STYLE["template"],
        width      = STYLE["width"],
        height     = height or STYLE["height"],
        font       = dict(size=STYLE["base_font"]),
        margin     = margin or MARGIN,
        showlegend = show_legend,
        legend     = LEGEND_H if show_legend else {},
    )
    if title:
        fig.update_layout(title=dict(
            text=title, x=0.5, y=0.98,
            xanchor="center", yanchor="top",
            font=dict(size=STYLE["title_font"]),
        ))
    fig.update_xaxes(title_font=dict(size=STYLE["axis_title_font"]), tickfont=dict(size=STYLE["tick_font"]))
    fig.update_yaxes(title_font=dict(size=STYLE["axis_title_font"]), tickfont=dict(size=STYLE["tick_font"]))
    fig.for_each_annotation(lambda a: a.update(font=dict(size=STYLE["facet_font"])))

    if global_ylabel:
        fig.for_each_yaxis(lambda ax: ax.update(title_text=""))
        # Remove any previous global y-label annotation so re-applying style
        # (e.g. in PDF_EXPORTS) never stacks duplicates with the old font size.
        fig.layout.annotations = [
            a for a in fig.layout.annotations
            if getattr(a, "textangle", 0) != -90
        ]
        fig.add_annotation(
            x=-0.09, y=0.5, xref="paper", yref="paper",
            text=global_ylabel, textangle=-90,
            showarrow=False, xanchor="center", yanchor="middle",
            font=dict(size=STYLE["axis_title_font"]),
        )
    return fig


# ----------------------------------------------------------
# Data loading
# ----------------------------------------------------------

def load_hw(agg_path: PathLike, raw_path: PathLike, hw_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    agg_path, raw_path = Path(agg_path), Path(raw_path)
    print(f"→ Loading {hw_name}")
    df     = pd.read_csv(agg_path)
    df_raw = pd.read_csv(raw_path)
    for d in (df, df_raw):
        if "model_name" in d.columns:
            d.query('model_name != "baseline"', inplace=True)
    df["hardware"]     = hw_name
    df_raw["hardware"] = hw_name
    return df.copy(), df_raw.copy()


dfs, dfs_raw = [], []
for agg_path, raw_path, hw_name in HW_CONFIGS:
    try:
        d, dr = load_hw(agg_path, raw_path, hw_name)
        dfs.append(d); dfs_raw.append(dr)
    except FileNotFoundError as e:
        print(f"  Skipping {hw_name}: {e}")

if not dfs:
    raise SystemExit("No hardware datasets loaded – check HW_CONFIGS paths.")

df_all     = pd.concat(dfs,     ignore_index=True)
df_raw_all = pd.concat(dfs_raw, ignore_index=True)

if "batch_size" in df_all.columns:
    df_all["batch_size"]     = pd.to_numeric(df_all["batch_size"], errors="coerce")
    df_all["batch_size_str"] = df_all["batch_size"].astype("Int64").astype(str)
else:
    df_all["batch_size_str"] = "NA"

# Registry of figures to export
named_figures: dict[str, go.Figure] = {}

def register(fig: go.Figure, name: str) -> go.Figure:
    fig.update_layout(title=dict(text=name, y=0.95))
    named_figures[name] = fig
    return fig


# ----------------------------------------------------------
# Figure 1 – External vs Internal Power by hardware
# ----------------------------------------------------------

meter_fit_rows: list[dict] = []

if {"power_draw_watts_mean", "power_meter_active_power_w_mean"}.issubset(df_all.columns):
    subset = df_all.dropna(subset=["power_draw_watts_mean", "power_meter_active_power_w_mean"])

    if not subset.empty:
        fig1 = px.scatter(
            subset,
            x="power_draw_watts_mean",
            y="power_meter_active_power_w_mean",
            color="model_name",
            symbol="dtype" if "dtype" in subset.columns else None,
            facet_col="hardware", facet_col_wrap=2,
            trendline="ols",
            labels=dict(
                power_draw_watts_mean="GPU-Reported Power (W)",
                power_meter_active_power_w_mean="External Meter Power (W)",
            ),
        )

        # Identity line in every facet
        mn = float(min(subset["power_draw_watts_mean"].min(), subset["power_meter_active_power_w_mean"].min()))
        mx = float(max(subset["power_draw_watts_mean"].max(), subset["power_meter_active_power_w_mean"].max()))
        for xax in [a for a in fig1.layout if a.startswith("xaxis")]:
            sfx  = xax.replace("xaxis", "")
            xref = "x" + sfx
            yref = "y" + sfx
            fig1.add_shape(type="line", x0=mn, y0=mn, x1=mx, y1=mx,
                           xref=xref, yref=yref,
                           line=dict(color="black", dash="dash", width=1), layer="below")

        apply_style(fig1,
                    title="External vs Internal Power by hardware",
                    height=450,
                    margin=dict(l=90, r=40, t=80, b=60),
                    show_legend=False)
        register(fig1, "External vs Internal Power by hardware")

        # OLS per (hardware, model) for summary table
        for (hw, model), g in subset.groupby(["hardware", "model_name"]):
            g2 = g.dropna(subset=["power_draw_watts_mean", "power_meter_active_power_w_mean"])
            if len(g2) < 3:
                continue
            X = sm.add_constant(g2["power_draw_watts_mean"], has_constant="add")
            try:
                fit = sm.OLS(g2["power_meter_active_power_w_mean"], X).fit()
                meter_fit_rows.append({"hardware": hw, "model_name": model,
                                       "slope": float(fit.params.get("power_draw_watts_mean", np.nan)),
                                       "intercept": float(fit.params.get("const", np.nan)),
                                       "R2": float(fit.rsquared), "n_points": len(g2)})
            except Exception:
                pass

meter_fit_model = pd.DataFrame(meter_fit_rows)
meter_fit_hw = (meter_fit_model.groupby("hardware")
                .agg(slope_mean=("slope","mean"), slope_std=("slope","std"),
                     R2_mean=("R2","mean"), R2_min=("R2","min"), R2_max=("R2","max"),
                     n_models=("model_name","nunique"))
                .reset_index()) if not meter_fit_model.empty else pd.DataFrame()


# ----------------------------------------------------------
# Figure 2 – Prediction error of MFU vs GPU Util for internal power
# ----------------------------------------------------------

pred_err_rows: list[dict] = []
err_needed = {"hardware", "power_draw_watts", "mfu_percentage_calflops", "gpu_utilization"}

if err_needed.issubset(df_raw_all.columns):
    for hw, g in df_raw_all.groupby("hardware"):
        cols = [c for c in ["hardware","model_name","dtype","batch_size",
                             "power_draw_watts","mfu_percentage_calflops","gpu_utilization"]
                if c in g.columns]
        g_base = g[cols].copy()
        if len(g_base) < 5:
            continue
        for pred_col, pred_label in [("mfu_percentage_calflops","MFU (%)"),("gpu_utilization","GPU Util (%)")]:
            if pred_col not in g_base.columns:
                continue
            g_sub = g_base.dropna(subset=[pred_col, "power_draw_watts"]).copy()
            if g_sub[pred_col].nunique() < 2 or g_sub["power_draw_watts"].nunique() < 2:
                continue
            X = sm.add_constant(g_sub[[pred_col]], has_constant="add")
            try:
                fit   = sm.OLS(g_sub["power_draw_watts"], X).fit()
                y_hat = fit.predict(X)
                abs_e = (g_sub["power_draw_watts"] - y_hat).abs()
                rel_e = abs_e / g_sub["power_draw_watts"].replace(0, np.nan)
                for (_, row), ae, re in zip(g_sub.iterrows(), abs_e, rel_e):
                    if np.isfinite(re):
                        pred_err_rows.append({"hardware": hw, "Predictor": pred_label,
                                              "abs_error_w": float(ae), "rel_error_pct": float(re * 100),
                                              "model_name": row.get("model_name"), "dtype": row.get("dtype"),
                                              "batch_size": row.get("batch_size")})
            except Exception:
                pass

pred_err_df = pd.DataFrame(pred_err_rows)

if not pred_err_df.empty:
    hover_cols = [c for c in ["model_name","dtype","batch_size"] if c in pred_err_df.columns]
    fig2 = px.violin(
        pred_err_df,
        x="Predictor", y="rel_error_pct",
        color="Predictor", facet_col="hardware", facet_col_wrap=2,
        box=True, points="outliers",
        color_discrete_map=COLOR_MAP,
        hover_data=hover_cols,
        labels=dict(Predictor="Predictor",
                    rel_error_pct="Absolute % error in internal power prediction",
                    hardware="Hardware"),
    )
    fig2.update_yaxes(range=[0, pred_err_df["rel_error_pct"].quantile(0.99)])
    apply_style(fig2,
                title="Prediction error of MFU vs GPU Utilization for internal power",
                height=750,
                global_ylabel="Absolute error in int. power prediction (%)")
    register(fig2, "Prediction error of MFU vs GPU Utilization for internal power")


# ----------------------------------------------------------
# Figure 3 – MFU and GPU Util vs internal power across dtypes
# ----------------------------------------------------------

dtype_needed = {"mfu_percentage_calflops_mean","gpu_utilization_mean","power_draw_watts_mean","hardware","dtype"}

if dtype_needed.issubset(df_all.columns):
    df_dt = df_all[list(dtype_needed)].dropna().copy()
    df_dt["dtype"] = df_dt["dtype"].astype(str)

    df_dt_long = pd.melt(df_dt,
        id_vars=["hardware","dtype","power_draw_watts_mean"],
        value_vars=["mfu_percentage_calflops_mean","gpu_utilization_mean"],
        var_name="Predictor", value_name="Predictor_value")
    df_dt_long["Predictor"] = df_dt_long["Predictor"].map(
        {"mfu_percentage_calflops_mean":"MFU (%)","gpu_utilization_mean":"GPU Util (%)"})

    fig3 = px.scatter(
        df_dt_long,
        x="Predictor_value", y="power_draw_watts_mean",
        color="dtype", symbol="Predictor",
        facet_col="hardware", facet_col_wrap=2,
        trendline="ols",
        labels=dict(Predictor_value="Predictor value (%)",
                    power_draw_watts_mean="Internal GPU power (W)",
                    hardware="Hardware", dtype="Dtype"),
    )
    fig3.update_traces(marker=dict(size=STYLE["marker_size"]))
    apply_style(fig3,
                title="MFU and GPU Utilization vs internal power across dtypes",
                height=750,
                global_ylabel="Internal GPU power (W)")
    register(fig3, "MFU and GPU Utilization vs internal power across dtypes")


# ----------------------------------------------------------
# OLS fits for dtype-level summary table
# ----------------------------------------------------------

dtype_fit_rows: list[dict] = []
if dtype_needed.issubset(df_all.columns):
    df_dt_fit = df_all[list(dtype_needed)].dropna().copy()
    df_dt_fit["dtype"] = df_dt_fit["dtype"].astype(str)
    df_dt_long_fit = pd.melt(df_dt_fit,
        id_vars=["hardware","dtype","power_draw_watts_mean"],
        value_vars=["mfu_percentage_calflops_mean","gpu_utilization_mean"],
        var_name="Predictor", value_name="Predictor_value")
    df_dt_long_fit["Predictor"] = df_dt_long_fit["Predictor"].map(
        {"mfu_percentage_calflops_mean":"MFU (%)","gpu_utilization_mean":"GPU Util (%)"})

    for (hw, dt, pred), g in df_dt_long_fit.groupby(["hardware","dtype","Predictor"]):
        g2 = g.dropna(subset=["Predictor_value","power_draw_watts_mean"])
        if len(g2) < 3 or g2["Predictor_value"].nunique() < 2:
            continue
        X = sm.add_constant(g2[["Predictor_value"]], has_constant="add")
        try:
            fit = sm.OLS(g2["power_draw_watts_mean"], X).fit()
            dtype_fit_rows.append({"hardware": hw, "dtype": dt, "Predictor": pred,
                                   "R2": float(fit.rsquared),
                                   "slope_W_per_pct": float(fit.params.get("Predictor_value", np.nan)),
                                   "n_points": len(g2)})
        except Exception:
            pass

dtype_fit_df = pd.DataFrame(dtype_fit_rows)

# ----------------------------------------------------------
# Small paper tables – config-level R² summaries
# ----------------------------------------------------------

CFG_KEYS = [c for c in [
    "hardware", "model_name", "dtype", "batch_size", "context_window",
    "warmup_iterations", "cooldown_seconds",
] if c in df_raw_all.columns]

TARGET_COL = "power_draw_watts"
PRED_COLS = {
    "MFU (%)": "mfu_percentage_calflops",
    "UTIL": "gpu_utilization",
}


def fit_r2_for_group(g: pd.DataFrame, x_col: str, y_col: str) -> float:
    g2 = g.dropna(subset=[x_col, y_col]).copy()
    if len(g2) < 5 or g2[x_col].nunique() < 2 or g2[y_col].nunique() < 2:
        return np.nan
    X = sm.add_constant(g2[[x_col]], has_constant="add")
    y = g2[y_col]
    try:
        fit = sm.OLS(y, X).fit()
        return float(fit.rsquared) if np.isfinite(fit.rsquared) else np.nan
    except Exception:
        return np.nan


cfg_r2_rows: list[dict] = []
needed_cfg = set(CFG_KEYS) | {TARGET_COL} | set(PRED_COLS.values())

if needed_cfg.issubset(df_raw_all.columns) and CFG_KEYS:
    d = df_raw_all[list(needed_cfg)].copy()

    if "batch_size" in d.columns:
        d["batch_size"] = pd.to_numeric(d["batch_size"], errors="coerce")
    if "context_window" in d.columns:
        d["context_window"] = pd.to_numeric(d["context_window"], errors="coerce")

    d = d.dropna(subset=CFG_KEYS + [TARGET_COL])

    for keys, g in d.groupby(CFG_KEYS, dropna=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(CFG_KEYS, keys))
        row["n_raw"] = int(len(g))                                                      #type: ignore
        for pred_label, pred_col in PRED_COLS.items():
            row[f"R2_{pred_label}"] = fit_r2_for_group(g, pred_col, TARGET_COL)         #type: ignore
        cfg_r2_rows.append(row)

cfg_r2_df = pd.DataFrame(cfg_r2_rows)


def r2_mean_std_table(cfg_df: pd.DataFrame, param: str) -> pd.DataFrame:
    if cfg_df.empty or param not in cfg_df.columns:
        return pd.DataFrame()

    df = cfg_df.copy()
    r2_mfu = "R2_MFU (%)"
    r2_util = "R2_UTIL"

    keep = df[[r2_mfu, r2_util]].notna().any(axis=1)
    df = df.loc[keep].copy()

    if param in ["batch_size", "context_window"]:
        df[param] = pd.to_numeric(df[param], errors="coerce")

    out = (
        df.groupby(param, dropna=True)
        .agg(
            n_configs=("n_raw", "count"),
            n_raw_total=("n_raw", "sum"),
            MFU_R2_mean=(r2_mfu, "mean"),
            MFU_R2_std=(r2_mfu, "std"),
            UTIL_R2_mean=(r2_util, "mean"),
            UTIL_R2_std=(r2_util, "std"),
            n_hw=("hardware", "nunique") if "hardware" in df.columns else ("n_raw", "count"),
            n_models=("model_name", "nunique") if "model_name" in df.columns else ("n_raw", "count"),
            n_dtypes=("dtype", "nunique") if "dtype" in df.columns else ("n_raw", "count"),
        )
        .reset_index()
    )

    out["MFU-energy correlation"] = (
        (out["MFU_R2_mean"] * 100).round(1).astype(str)
        + " ± "
        + (out["MFU_R2_std"] * 100).round(1).fillna(0).astype(str)
        + " %"
    )
    out["UTIL-energy correlation"] = (
        (out["UTIL_R2_mean"] * 100).round(1).astype(str)
        + " ± "
        + (out["UTIL_R2_std"] * 100).round(1).fillna(0).astype(str)
        + " %"
    )

    out = out.sort_values(param)

    cols = [
        param,
        "MFU-energy correlation",
        "UTIL-energy correlation",
        "n_configs",
        "n_raw_total",
        "n_hw",
        "n_models",
        "n_dtypes",
    ]
    return out[cols]


param_tables: dict[str, pd.DataFrame] = {}
for param in ["batch_size", "context_window", "dtype", "hardware"]:
    t = r2_mean_std_table(cfg_r2_df, param)
    if not t.empty:
        param_tables[param] = t
        print(f"\nEffect of {param} on MFU/UTIL -> INTERNAL power predictability (config-level R^2)")
        print(t.to_string(index=False))


# ----------------------------------------------------------
# Export HTML dashboard
# ----------------------------------------------------------

figs = list(named_figures.values())
if not figs:
    raise SystemExit("No figures generated – check your input data.")

html_parts = [pio.to_html(figs[0], include_plotlyjs=True,  full_html=False)]
html_parts += [pio.to_html(f,       include_plotlyjs=False, full_html=False) for f in figs[1:]]

tables_html = ""
if not meter_fit_hw.empty:
    tables_html += "<h2>Per-hardware meter agreement (External vs Internal)</h2>"
    tables_html += meter_fit_hw.round(4).to_html(index=False, border=0)
if not dtype_fit_df.empty:
    tables_html += "<h2>OLS fits: Internal power vs Predictor per dtype</h2>"
    tables_html += dtype_fit_df.round(4).to_html(index=False, border=0)
for param, table_df in param_tables.items():
    pretty = {
        "batch_size": "Batch size",
        "context_window": "Context window",
        "dtype": "Dtype",
        "hardware": "Hardware",
    }.get(param, param)
    tables_html += f"<h2>Effect of {pretty} on MFU/UTIL → INTERNAL power predictability (config-level R²)</h2>"
    tables_html += table_df.to_html(index=False, border=0)

html = (
    "<!doctype html><html><head><meta charset='utf-8'/>"
    "<title>Cross-Hardware Energy Predictors</title>"
    "<style>body{font-family:system-ui,Arial;margin:16px}"
    ".fig{margin-bottom:32px}"
    "table{border-collapse:collapse;margin:16px 0 32px}"
    "th,td{border:1px solid #ccc;padding:4px 8px;font-size:12px}"
    "h2{margin-top:32px}</style></head><body>"
    "<h1>Cross-Hardware MFU vs GPU Utilization Analysis</h1>"
    + "".join(f'<div class="fig">{h}</div>' for h in html_parts)
    + tables_html
    + "</body></html>"
)

with open(OUTFILE, "w", encoding="utf-8") as f:
    f.write(html)
print(f"\nHTML saved → {OUTFILE.resolve()}")


# ----------------------------------------------------------
# Export PDFs
# ----------------------------------------------------------

PDF_EXPORTS = {
    "External vs Internal Power by hardware": dict(height=400, margin=dict(l=90, r=40, t=80, b=60), show_legend=False),
    "Prediction error of MFU vs GPU Utilization for internal power": dict(height=600, show_legend=False),
    "MFU and GPU Utilization vs internal power across dtypes": dict(height=750),
}

for name, overrides in PDF_EXPORTS.items():
    fig = named_figures.get(name)
    if fig is None:
        print(f"  Figure not found: '{name}'")
        continue
    # Re-apply style with any per-figure overrides
    apply_style(fig, title=name, **overrides)
    pdf_path = PDF_DIR / f"{name}.pdf"
    pio.write_image(fig, pdf_path, format="pdf", scale=2)
    print(f"PDF saved → {pdf_path}")