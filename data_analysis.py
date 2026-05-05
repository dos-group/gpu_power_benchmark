#!/usr/bin/env python
"""Cross-hardware energy predictor analysis.
Loads CSVs and produces paper tables / table-only HTML output.
"""

from __future__ import annotations
from pathlib import Path
from typing import Union, Tuple
import numpy as np
import pandas as pd
import statsmodels.api as sm

PathLike = Union[str, Path]

# ----------------------------------------------------------
# Configuration
# ----------------------------------------------------------

HW_CONFIGS: list[tuple[PathLike, PathLike, str]] = [
    ("./aggregation_results/mfu_aggregated_per_config_A100.csv",  "./benchmark_results/mfu_benchmark_results_A100_128.csv",  "NVIDIA A100"),
    ("./aggregation_results/mfu_aggregated_per_config_L40.csv",   "./benchmark_results/mfu_benchmark_results_L40_128.csv",   "NVIDIA L40"),
    ("./aggregation_results/mfu_aggregated_per_config_GPU06.csv", "./benchmark_results/mfu_benchmark_results_GPU06_128.csv", "Quadro 5000"),
    ("./aggregation_results/mfu_aggregated_per_config_4070.csv",  "./benchmark_results/mfu_benchmark_results_4070_128.csv",  "RTX 4070 Ti"),
    ("./aggregation_results/mfu_aggregated_per_config_MI210.csv", "./benchmark_results/mfu_benchmark_results_MI210_128.csv", "AMD MI210"),
]

OUTDIR = Path("results")
OUTFILE = OUTDIR / "cross_hardware_energy_predictors.html"
OUTDIR.mkdir(exist_ok=True)


# ----------------------------------------------------------
# Data loading
# ----------------------------------------------------------


def load_hw(agg_path: PathLike, raw_path: PathLike, hw_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    agg_path, raw_path = Path(agg_path), Path(raw_path)
    print(f"→ Loading {hw_name}")
    df = pd.read_csv(agg_path)
    df_raw = pd.read_csv(raw_path)
    for d in (df, df_raw):
        if "model_name" in d.columns:
            d.query('model_name != "baseline"', inplace=True)
    df["hardware"] = hw_name
    df_raw["hardware"] = hw_name
    return df.copy(), df_raw.copy()



dfs, dfs_raw = [], []
for agg_path, raw_path, hw_name in HW_CONFIGS:
    try:
        d, dr = load_hw(agg_path, raw_path, hw_name)
        dfs.append(d)
        dfs_raw.append(dr)
    except FileNotFoundError as e:
        print(f"  Skipping {hw_name}: {e}")

if not dfs:
    raise SystemExit("No hardware datasets loaded – check HW_CONFIGS paths.")

df_all = pd.concat(dfs, ignore_index=True)
df_raw_all = pd.concat(dfs_raw, ignore_index=True)

if "batch_size" in df_all.columns:
    df_all["batch_size"] = pd.to_numeric(df_all["batch_size"], errors="coerce")
    df_all["batch_size_str"] = df_all["batch_size"].astype("Int64").astype(str)
else:
    df_all["batch_size_str"] = "NA"


# ----------------------------------------------------------
# Table – External vs Internal Power by hardware
# ----------------------------------------------------------

meter_fit_rows: list[dict] = []

if {"power_draw_watts_mean", "power_meter_active_power_w_mean"}.issubset(df_all.columns):
    subset = df_all.dropna(subset=["power_draw_watts_mean", "power_meter_active_power_w_mean"])

    for (hw, model), g in subset.groupby(["hardware", "model_name"]):
        g2 = g.dropna(subset=["power_draw_watts_mean", "power_meter_active_power_w_mean"])
        if len(g2) < 3:
            continue
        X = sm.add_constant(g2["power_draw_watts_mean"], has_constant="add")
        try:
            fit = sm.OLS(g2["power_meter_active_power_w_mean"], X).fit()
            meter_fit_rows.append(
                {
                    "hardware": hw,
                    "model_name": model,
                    "slope": float(fit.params.get("power_draw_watts_mean", np.nan)),
                    "intercept": float(fit.params.get("const", np.nan)),
                    "R2": float(fit.rsquared),
                    "n_points": len(g2),
                }
            )
        except Exception:
            pass

meter_fit_model = pd.DataFrame(meter_fit_rows)
meter_fit_hw = (
    meter_fit_model.groupby("hardware")
    .agg(
        slope_mean=("slope", "mean"),
        slope_std=("slope", "std"),
        R2_mean=("R2", "mean"),
        R2_min=("R2", "min"),
        R2_max=("R2", "max"),
        n_models=("model_name", "nunique"),
    )
    .reset_index()
    if not meter_fit_model.empty
    else pd.DataFrame()
)


# ----------------------------------------------------------
# Table – MFU vs GPU Utilization as predictors of INTERNAL power
#        (same data source as the cross_hardware_analysis bar figure)
# ----------------------------------------------------------

pred_int_df = pd.DataFrame()
pred_int_table = pd.DataFrame()

raw_needed = {
    "hardware",
    "power_draw_watts",
    "mfu_percentage_calflops",
    "gpu_utilization",
}

if raw_needed.issubset(df_raw_all.columns):
    pred_rows_int: list[dict] = []

    for hw, g in df_raw_all.groupby("hardware"):
        target = "power_draw_watts"

        for pred_col in ["mfu_percentage_calflops", "gpu_utilization"]:
            g_sub = g.dropna(subset=[pred_col, target]).copy()

            if (
                len(g_sub) < 3
                or g_sub[pred_col].nunique() < 2
                or g_sub[target].nunique() < 2
            ):
                continue

            X = sm.add_constant(g_sub[[pred_col]], has_constant="add")
            y = g_sub[target]

            try:
                fit = sm.OLS(y, X).fit()
                r2 = float(fit.rsquared) if np.isfinite(fit.rsquared) else np.nan
            except Exception:
                r2 = np.nan

            pred_rows_int.append(
                {
                    "hardware": hw,
                    "Predictor": pred_col,
                    "R2": r2,
                    "n_points": int(len(g_sub)),
                }
            )

    pred_int_df = pd.DataFrame(pred_rows_int)

    if not pred_int_df.empty:
        pred_int_df["Predictor"] = pred_int_df["Predictor"].map(
            {
                "mfu_percentage_calflops": "MFU (%)",
                "gpu_utilization": "GPU Util (%)",
            }
        )

        pred_int_table = (
            pred_int_df.pivot(index="hardware", columns="Predictor", values="R2").reset_index()
        )

        predictor_cols = [c for c in ["MFU (%)", "GPU Util (%)"] if c in pred_int_table.columns]
        pred_int_table = pred_int_table[["hardware"] + predictor_cols]

        for col in predictor_cols:
            pred_int_table[col] = pred_int_table[col].round(4)

        print("\nMFU vs GPU Utilization as predictors of INTERNAL power (table):")
        print(pred_int_table.to_string(index=False))


# ----------------------------------------------------------
# OLS fits for dtype-level summary table
# ----------------------------------------------------------

dtype_needed = {
    "mfu_percentage_calflops_mean",
    "gpu_utilization_mean",
    "power_draw_watts_mean",
    "hardware",
    "dtype",
}

dtype_fit_rows: list[dict] = []
if dtype_needed.issubset(df_all.columns):
    df_dt_fit = df_all[list(dtype_needed)].dropna().copy()
    df_dt_fit["dtype"] = df_dt_fit["dtype"].astype(str)
    df_dt_long_fit = pd.melt(
        df_dt_fit,
        id_vars=["hardware", "dtype", "power_draw_watts_mean"],
        value_vars=["mfu_percentage_calflops_mean", "gpu_utilization_mean"],
        var_name="Predictor",
        value_name="Predictor_value",
    )
    df_dt_long_fit["Predictor"] = df_dt_long_fit["Predictor"].map(
        {
            "mfu_percentage_calflops_mean": "MFU (%)",
            "gpu_utilization_mean": "GPU Util (%)",
        }
    )

    for (hw, dt, pred), g in df_dt_long_fit.groupby(["hardware", "dtype", "Predictor"]):
        g2 = g.dropna(subset=["Predictor_value", "power_draw_watts_mean"])
        if len(g2) < 3 or g2["Predictor_value"].nunique() < 2:
            continue
        X = sm.add_constant(g2[["Predictor_value"]], has_constant="add")
        try:
            fit = sm.OLS(g2["power_draw_watts_mean"], X).fit()
            dtype_fit_rows.append(
                {
                    "hardware": hw,
                    "dtype": dt,
                    "Predictor": pred,
                    "R2": float(fit.rsquared),
                    "slope_W_per_pct": float(fit.params.get("Predictor_value", np.nan)),
                    "n_points": len(g2),
                }
            )
        except Exception:
            pass

dtype_fit_df = pd.DataFrame(dtype_fit_rows)


# ----------------------------------------------------------
# Small paper tables – config-level R² summaries
# ----------------------------------------------------------

CFG_KEYS = [
    c
    for c in [
        "hardware",
        "model_name",
        "dtype",
        "batch_size",
        "context_window",
        "warmup_iterations",
        "cooldown_seconds",
    ]
    if c in df_raw_all.columns
]

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


# MI210's GPU Utilization is a binary GRBM_COUNT activity counter (see paper
# Section 4.1), not a throughput proxy. We therefore exclude it from any
# GPU-Utilization aggregate to avoid skewing per-parameter means.
UTIL_EXCLUDE_HW = {"AMD MI210"}


def _fmt_pct(mean: float, std: float) -> str:
    if pd.isna(mean):
        return "--"
    std_part = "0.0" if pd.isna(std) else f"{std * 100:.1f}"
    return f"{mean * 100:.1f} $\\pm$ {std_part}\\,\\%"


def r2_summary_table(cfg_df: pd.DataFrame, param: str) -> pd.DataFrame:
    """Per-slice mean +/- std of config-level R^2 for MFU and GPU Util.

    GPU Util column excludes MI210 (binary GRBM counter). MFU column uses all
    GPUs. Returned table reports the actual config count behind each cell so
    the asymmetry is visible.
    """
    if cfg_df.empty or param not in cfg_df.columns:
        return pd.DataFrame()

    df = cfg_df.copy()
    r2_mfu = "R2_MFU (%)"
    r2_util = "R2_UTIL"

    if param in ("batch_size", "context_window"):
        df[param] = pd.to_numeric(df[param], errors="coerce")

    df_mfu = df.dropna(subset=[r2_mfu]).copy()
    df_util = df[~df["hardware"].isin(UTIL_EXCLUDE_HW)].dropna(subset=[r2_util]).copy()

    mfu_agg = (
        df_mfu.groupby(param, dropna=True)[r2_mfu]
        .agg(MFU_mean="mean", MFU_std="std", MFU_n="count")
        .reset_index()
    )
    util_agg = (
        df_util.groupby(param, dropna=True)[r2_util]
        .agg(UTIL_mean="mean", UTIL_std="std", UTIL_n="count")
        .reset_index()
    )
    out = mfu_agg.merge(util_agg, on=param, how="outer").sort_values(param)

    out["MFU $R^2$"] = [_fmt_pct(m, s) for m, s in zip(out["MFU_mean"], out["MFU_std"])]
    out["GPU Util $R^2$"] = [_fmt_pct(m, s) for m, s in zip(out["UTIL_mean"], out["UTIL_std"])]
    out["$n_\\text{MFU}$"] = out["MFU_n"].fillna(0).astype(int)
    out["$n_\\text{Util}$"] = out["UTIL_n"].fillna(0).astype(int)

    return out[[param, "MFU $R^2$", "GPU Util $R^2$",
                "$n_\\text{MFU}$", "$n_\\text{Util}$"]]


param_tables: dict[str, pd.DataFrame] = {}
for param in ["hardware", "dtype", "batch_size", "context_window"]:
    t = r2_summary_table(cfg_r2_df, param)
    if not t.empty:
        param_tables[param] = t
        print(f"\nConfig-level R^2 sliced by {param} (Util excludes MI210):")
        print(t.to_string(index=False))


# --- LaTeX export of the restructured Table 2 ----------------------------

def _emit_latex_table(param_tables: dict[str, pd.DataFrame]) -> str:
    label_map = {
        "hardware": "Hardware",
        "dtype": "Dtype",
        "batch_size": "Batch size",
        "context_window": "Context window",
    }
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Mean configuration-level explained variance ($R^2$) of linear "
        r"power models, sliced by hardware and by each independent variable. "
        r"Each row aggregates the per-configuration $R^2$ values "
        r"(one fit per (hw, model, dtype, batch, ctx, warmup, cooldown) tuple) "
        r"falling into that slice, reported as mean~$\pm$~std. "
        r"$n_\text{MFU}$ and $n_\text{Util}$ give the number of configurations "
        r"behind each cell. The AMD MI210 is excluded from the GPU Utilization "
        r"column because its ROCm \texttt{GRBM\_COUNT}-derived signal is a "
        r"binary activity flag rather than a throughput proxy "
        r"(see Section~\ref{Predictor Performance}); MFU values for the MI210 "
        r"remain meaningful and are reported.}",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{1.15}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"\textbf{Slice} & \textbf{MFU $R^2$} & \textbf{GPU Util $R^2$} & "
        r"$\boldsymbol{n_\text{MFU}}$ & $\boldsymbol{n_\text{Util}}$ \\",
        r"\midrule",
    ]
    first = True
    for param in ["hardware", "dtype", "batch_size", "context_window"]:
        if param not in param_tables:
            continue
        t = param_tables[param]
        if not first:
            lines.append(r"\midrule")
        first = False
        lines.append(rf"\multicolumn{{5}}{{l}}{{\textit{{{label_map[param]}}}}} \\")
        for _, r in t.iterrows():
            slice_name = str(r[param])
            if param in ("batch_size", "context_window"):
                try:
                    slice_name = str(int(float(slice_name)))
                except Exception:
                    pass
            lines.append(
                f"\\quad {slice_name} & {r['MFU $R^2$']} & {r['GPU Util $R^2$']} & "
                f"{r['$n_\\text{MFU}$']} & {r['$n_\\text{Util}$']} \\\\"
            )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\label{tab:mfu_util_internal_power_per_config}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


tex_out = OUTDIR / "tab_configuration_variance.tex"
tex_out.write_text(_emit_latex_table(param_tables))
print(f"\nLaTeX table written -> {tex_out.resolve()}")


# ----------------------------------------------------------
# Export table-only HTML
# ----------------------------------------------------------

tables_html = ""
if not meter_fit_hw.empty:
    tables_html += "<h2>Per-hardware meter agreement (External vs Internal)</h2>"
    tables_html += meter_fit_hw.round(4).to_html(index=False, border=0)
if not pred_int_table.empty:
    tables_html += "<h2>MFU vs GPU Utilization as predictors of INTERNAL power</h2>"
    tables_html += pred_int_table.to_html(index=False, border=0)
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
    "table{border-collapse:collapse;margin:16px 0 32px}"
    "th,td{border:1px solid #ccc;padding:4px 8px;font-size:12px}"
    "h2{margin-top:32px}</style></head><body>"
    "<h1>Cross-Hardware MFU vs GPU Utilization Analysis</h1>"
    + tables_html
    + "</body></html>"
)

with open(OUTFILE, "w", encoding="utf-8") as f:
    f.write(html)
print(f"\nHTML saved → {OUTFILE.resolve()}")
