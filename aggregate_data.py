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
INPUT_CSV = "mfu_benchmark_results_A100_128.csv"
OUTPUT_DIR = "./aggregation_128"
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


def main():
    print("=" * 70)
    print("MFU Benchmark Aggregation Script")
    print("=" * 70)
    
    # Create output directory
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    
    # Load and clean data
    print(f"\nLoading data from: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    print(f"  Raw samples: {len(df)}")
    
    df = clean_dataframe(df)
    print(f"  Cleaned samples: {len(df)}")
    
    # Aggregate by configuration
    print("\nAggregating by configuration...")
    agg_df = aggregate_by_config(df)
    print(f"  Unique configurations: {len(agg_df)}")
    print(f"  Capped configurations: {agg_df['cap_reached'].sum()}")
    
    # Save aggregated results
    agg_path = output_dir / "mfu_aggregated_per_config_A100.csv"
    agg_df.to_csv(agg_path, index=False)
    print(f"\nSaved aggregated results: {agg_path}")
    
    # Save capped configurations separately
    capped_df = agg_df[agg_df["cap_reached"]].copy()
    if len(capped_df) > 0:
        capped_path = output_dir / "mfu_capped_configs.csv"
        capped_df.to_csv(capped_path, index=False)
        print(f"Saved capped configs: {capped_path}")


if __name__ == "__main__":
    main()