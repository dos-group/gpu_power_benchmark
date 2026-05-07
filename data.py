"""Shared loader for the published per-GPU benchmark and aggregation CSVs.

Used by both ``Data-Analysis.py`` (table generation) and the ``figures/``
plot scripts. Hardware display names use ``<br>`` as a soft-break placeholder;
callers replace it with space or newline as needed for their output medium.
"""

from __future__ import annotations
from pathlib import Path
from typing import Tuple
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent

# (aggregation_csv, raw_csv, display_name)
HW_CONFIGS: list[tuple[str, str, str]] = [
    ("aggregation_results/mfu_aggregated_per_config_A100.csv",
     "benchmark_results/mfu_benchmark_results_A100_128.csv",
     "NVIDIA<br>A100"),
    ("aggregation_results/mfu_aggregated_per_config_L40.csv",
     "benchmark_results/mfu_benchmark_results_L40_128.csv",
     "NVIDIA<br>L40"),
    ("aggregation_results/mfu_aggregated_per_config_GPU06.csv",
     "benchmark_results/mfu_benchmark_results_GPU06_128.csv",
     "NVIDIA<br>Quadro 5000"),
    ("aggregation_results/mfu_aggregated_per_config_4070.csv",
     "benchmark_results/mfu_benchmark_results_4070_128.csv",
     "NVIDIA<br>RTX 4070 Ti"),
    ("aggregation_results/mfu_aggregated_per_config_MI210.csv",
     "benchmark_results/mfu_benchmark_results_MI210_128.csv",
     "AMD<br>MI210"),
]

# AMD MI210's GPU Utilization is a binary GRBM_COUNT activity counter, not
# a throughput proxy; exclude from any GPU-Util aggregate.
UTIL_EXCLUDE_HW: set[str] = {"AMD<br>MI210"}

# Only these GPUs were instrumented with an external power meter.
EXTERNAL_VALIDATED_HW: set[str] = {"NVIDIA<br>L40", "NVIDIA<br>Quadro 5000"}


def load_all_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (aggregated_df, raw_df) concatenated across all hardware."""
    dfs_agg, dfs_raw = [], []
    for agg_p, raw_p, hw in HW_CONFIGS:
        try:
            a = pd.read_csv(PROJECT_ROOT / agg_p)
            r = pd.read_csv(PROJECT_ROOT / raw_p)
        except FileNotFoundError as e:
            print(f"  Skipping {hw.replace('<br>', ' ')}: {e}")
            continue
        for d in (a, r):
            if "model_name" in d.columns:
                d.query('model_name != "baseline"', inplace=True)
            d["hardware"] = hw
        dfs_agg.append(a)
        dfs_raw.append(r)
        print(f"-> Loaded {hw.replace('<br>', ' ')}")
    if not dfs_agg:
        raise RuntimeError("No hardware datasets loaded.")
    return (pd.concat(dfs_agg, ignore_index=True),
            pd.concat(dfs_raw, ignore_index=True))
