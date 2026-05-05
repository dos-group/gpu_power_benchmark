from __future__ import annotations
from pathlib import Path
from typing import Tuple
import pandas as pd

# Project root resolution
PROJECT_ROOT = Path(__file__).parent.parent

# ---------------------------------------------------------------- HW configs
# (aggregation_csv, raw_csv, display_name)
# Paths are relative to PROJECT_ROOT
HW_CONFIGS: list[tuple[str, str, str]] = [
    ("aggregation_results/mfu_aggregated_per_config_A100.csv",
     "benchmark_results/mfu_benchmark_results_A100_128.csv",
     "NVIDIA A100"),
    ("aggregation_results/mfu_aggregated_per_config_L40.csv",
     "benchmark_results/mfu_benchmark_results_L40_128.csv",
     "NVIDIA L40"),
    ("aggregation_results/mfu_aggregated_per_config_GPU06.csv",
     "benchmark_results/mfu_benchmark_results_GPU06_128.csv",
     "Quadro 5000"),
    ("aggregation_results/mfu_aggregated_per_config_4070.csv",
     "benchmark_results/mfu_benchmark_results_4070_128.csv",
     "RTX 4070 Ti"),
    ("aggregation_results/mfu_aggregated_per_config_MI210.csv",
     "benchmark_results/mfu_benchmark_results_MI210_128.csv",
     "AMD MI210"),
]

# Hardware to exclude from any GPU-Utilization aggregation/figure.
UTIL_EXCLUDE_HW: set[str] = {"AMD MI210"}

# Hardware for which external power-meter measurements were collected.
EXTERNAL_VALIDATED_HW: set[str] = {"NVIDIA L40", "Quadro 5000"}


def load_hw_data(agg_path_rel: str, raw_path_rel: str, hw_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and clean a single hardware's aggregated and raw data."""
    agg_path = PROJECT_ROOT / agg_path_rel
    raw_path = PROJECT_ROOT / raw_path_rel
    
    print(f"→ Loading {hw_name}")
    df_agg = pd.read_csv(agg_path)
    df_raw = pd.read_csv(raw_path)
    
    for d in (df_agg, df_raw):
        if "model_name" in d.columns:
            # Filter out baseline runs
            d.query('model_name != "baseline"', inplace=True)
            
    df_agg["hardware"] = hw_name
    df_raw["hardware"] = hw_name
    
    return df_agg.copy(), df_raw.copy()


def load_all_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and combine data for all configured hardware."""
    dfs_agg, dfs_raw = [], []
    for agg_p, raw_p, hw_name in HW_CONFIGS:
        try:
            d_agg, d_raw = load_hw_data(agg_p, raw_p, hw_name)
            dfs_agg.append(d_agg)
            dfs_raw.append(d_raw)
        except FileNotFoundError as e:
            print(f"  Skipping {hw_name}: {e}")

    if not dfs_agg:
        raise RuntimeError("No hardware datasets loaded. Check HW_CONFIGS paths.")

    df_all_agg = pd.concat(dfs_agg, ignore_index=True)
    df_all_raw = pd.concat(dfs_raw, ignore_index=True)

    return df_all_agg, df_all_raw
