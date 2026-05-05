import numpy as np
import pandas as pd
import statsmodels.api as sm
from pathlib import Path
from analysis.data_loader import load_all_data, PROJECT_ROOT, UTIL_EXCLUDE_HW

OUTDIR = PROJECT_ROOT / "results"
OUTFILE = OUTDIR / "cross_hardware_energy_predictors.html"


def generate_meter_agreement(df_all: pd.DataFrame):
    meter_fit_rows = []
    subset = df_all.dropna(subset=["power_draw_watts_mean", "power_meter_active_power_w_mean"])
    for (hw, model), g in subset.groupby(["hardware", "model_name"]):
        if len(g) < 3: continue
        X = sm.add_constant(g["power_draw_watts_mean"])
        try:
            fit = sm.OLS(g["power_meter_active_power_w_mean"], X).fit()
            meter_fit_rows.append({
                "hardware": hw, "model": model, 
                "slope": fit.params.get("power_draw_watts_mean", np.nan),
                "intercept": fit.params.get("const", np.nan),
                "R2": fit.rsquared
            })
        except Exception: pass
    return pd.DataFrame(meter_fit_rows)


def generate_predictor_comparison(df_raw: pd.DataFrame):
    pred_rows = []
    for hw, g in df_raw.groupby("hardware"):
        for pred_col, label in [("mfu_percentage_calflops", "MFU (%)"), ("gpu_utilization", "GPU Util (%)")]:
            if label == "GPU Util (%)" and hw in UTIL_EXCLUDE_HW: continue
            g_sub = g.dropna(subset=[pred_col, "power_draw_watts"])
            if len(g_sub) < 5: continue
            X = sm.add_constant(g_sub[[pred_col]])
            fit = sm.OLS(g_sub["power_draw_watts"], X).fit()
            pred_rows.append({"hardware": hw, "Predictor": label, "R2": fit.rsquared})
    return pd.DataFrame(pred_rows)


def main():
    OUTDIR.mkdir(exist_ok=True)
    df_agg, df_raw = load_all_data()
    
    print("Generating Analysis Tables...")
    agreement = generate_meter_agreement(df_agg)
    comparison = generate_predictor_comparison(df_raw)
    
    html = f"""
    <!doctype html><html><head><title>Results</title></head><body>
    <h1>Meter Agreement</h1>{agreement.to_html()}
    <h1>Predictor Comparison</h1>{comparison.to_html()}
    </body></html>
    """
    OUTFILE.write_text(html)
    print(f"Results saved to {OUTFILE}")


if __name__ == "__main__":
    main()
