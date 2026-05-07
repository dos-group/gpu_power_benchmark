#!/usr/bin/env python3
"""
Simplified MFU aggregation script.
Aggregates benchmark results by configuration and flags capped runs.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================================
# CONFIGURATION - Edit these values as needed
# ============================================================================
INPUT_DIR = "./benchmark_results"
OUTPUT_DIR = "./aggregation_results"
INPUT_GLOB = "mfu_benchmark_results_*.csv"
MAX_SAMPLES_PER_CONFIG = 300
GENERATE_CHARTS = True

# ============================================================================
# Column definitions
# ============================================================================
METRIC_COLS = [
    "tokens_per_second",
    "mfu_percentage",
    "mfu_percentage_calflops",
    "sequence_length",
    "gpu_utilization",
    "memory_utilization",
    "memory_used_mb",
    "temperature_celsius",
    "power_draw_watts",
    "num_layers",
    "num_heads",
    "head_dim",
    "formula_flops_per_token",
    "calflops_total",
    "io_latency_actual_ms",
    "io_latency_mean_config_ms",
    "power_meter_active_power_w",
    "power_meter_reactive_power_var",
    "power_meter_apparent_power_va",
]

CONFIG_KEYS = [
    "model_name",
    "batch_size",
    "dtype",
    "sequence_length",
    "context_window",
    "learning_rate",
    "warmup_iterations",
    "cooldown_seconds",
    "io_latency_enabled",
    "io_latency_profile",
    "io_latency_pattern",
    "power_meter_available",
]


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Convert columns to appropriate data types."""
    df = df.copy()
    
    # Convert timestamp columns
    for col in ["timestamp", "power_meter_timestamp"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    
    # Convert metric columns to numeric
    for col in METRIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Convert integer config columns
    for col in ["batch_size", "sequence_length", "context_window", "warmup_iterations"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Convert boolean columns
    for col in ["io_latency_enabled", "power_meter_available"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().map({
                "true": True, "false": False, "1": True, "0": False
            })
    
    # Split io_latency_type into profile and pattern
    if "io_latency_type" in df.columns:
        split_data = df["io_latency_type"].astype(str).str.split("/", n=1, expand=True)
        df["io_latency_profile"] = split_data[0].str.strip() if 0 in split_data.columns else pd.NA
        df["io_latency_pattern"] = split_data[1].str.strip() if 1 in split_data.columns else pd.NA
        
        # Handle missing values
        missing_mask = df["io_latency_type"].isna()
        df.loc[missing_mask, ["io_latency_profile", "io_latency_pattern"]] = pd.NA
    
    return df


def aggregate_by_config(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate metrics by configuration."""
    # Get available config keys and metrics
    available_config_keys = [k for k in CONFIG_KEYS if k in df.columns]
    available_metrics = [m for m in METRIC_COLS if m in df.columns]
    
    if not available_config_keys:
        raise ValueError("No configuration keys found in dataframe")
    
    # Group by configuration
    grouped = df.groupby(available_config_keys, dropna=False)
    
    # Calculate means
    means = grouped[available_metrics].mean()
    means.columns = [f"{col}_mean" for col in means.columns]
    
    # Calculate counts (use first available metric)
    counts = grouped[available_metrics[0]].count()
    counts.name = "samples_per_config"
    
    # Combine into single dataframe
    result = pd.concat([means, counts], axis=1).reset_index()
    
    # Flag configs that hit the cap
    result["cap_reached"] = result["samples_per_config"] >= MAX_SAMPLES_PER_CONFIG
    
    return result


def aggregate_one(input_csv: Path, output_dir: Path) -> None:
    # Derive GPU label from filename: mfu_benchmark_results_<label>_<batch>.csv
    label = "_".join(input_csv.stem.split("_")[3:-1])

    print(f"\n[{label}] {input_csv}")
    df = clean_dataframe(pd.read_csv(input_csv))
    agg_df = aggregate_by_config(df)
    print(f"  configs: {len(agg_df)}, capped: {agg_df['cap_reached'].sum()}")

    out = output_dir / f"mfu_aggregated_per_config_{label}.csv"
    agg_df.to_csv(out, index=False)
    print(f"  -> {out}")


def main():
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    inputs = sorted(Path(INPUT_DIR).glob(INPUT_GLOB))
    if not inputs:
        raise SystemExit(f"No inputs match {INPUT_DIR}/{INPUT_GLOB}")

    print("=" * 70)
    print(f"Aggregating {len(inputs)} benchmark file(s) -> {output_dir}")
    print("=" * 70)
    for path in inputs:
        aggregate_one(path, output_dir)


if __name__ == "__main__":
    main()