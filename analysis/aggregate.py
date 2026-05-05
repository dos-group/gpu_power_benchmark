import pandas as pd
from pathlib import Path
from analysis.data_loader import PROJECT_ROOT, HW_CONFIGS

OUTPUT_DIR = PROJECT_ROOT / "aggregation_results"
MAX_SAMPLES_PER_CONFIG = 300

METRIC_COLS = [
    "tokens_per_second", "mfu_percentage", "mfu_percentage_calflops",
    "sequence_length", "gpu_utilization", "memory_utilization",
    "memory_used_mb", "temperature_celsius", "power_draw_watts",
    "power_meter_active_power_w",
]

CONFIG_KEYS = ["model_name", "batch_size", "dtype", "sequence_length", "context_window"]


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in METRIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def aggregate_by_config(df: pd.DataFrame) -> pd.DataFrame:
    available_configs = [k for k in CONFIG_KEYS if k in df.columns]
    available_metrics = [m for m in METRIC_COLS if m in df.columns]
    
    grouped = df.groupby(available_configs, dropna=False)
    means = grouped[available_metrics].mean().add_suffix("_mean")
    counts = grouped[available_metrics[0]].count().rename("samples_per_config")
    
    result = pd.concat([means, counts], axis=1).reset_index()
    result["cap_reached"] = result["samples_per_config"] >= MAX_SAMPLES_PER_CONFIG
    return result


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    for agg_rel, raw_rel, hw_name in HW_CONFIGS:
        raw_path = PROJECT_ROOT / raw_rel
        if not raw_path.exists():
            print(f"Skipping {hw_name}: {raw_path} not found")
            continue
            
        print(f"Aggregating {hw_name}...")
        df = pd.read_csv(raw_path)
        df = clean_dataframe(df)
        agg_df = aggregate_by_config(df)
        
        output_path = PROJECT_ROOT / agg_rel
        agg_df.to_csv(output_path, index=False)
        print(f"  Saved to {output_path}")


if __name__ == "__main__":
    main()
