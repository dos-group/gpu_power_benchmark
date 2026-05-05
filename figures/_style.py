"""Shared styling for paper figures.

One canonical color per GPU, shared font sizes, and a single hardware-config
list reused by every figure script. To add a new GPU (e.g. L4), append an
entry to HW_CONFIGS and add the matching color in HW_COLORS below.
"""

from __future__ import annotations
from pathlib import Path

# ---------------------------------------------------------------- HW configs
# (aggregation_csv, raw_csv, display_name)
HW_CONFIGS: list[tuple[str, str, str]] = [
    ("../aggregation_results/mfu_aggregated_per_config_A100.csv",
     "../benchmark_results/mfu_benchmark_results_A100_128.csv",
     "NVIDIA A100"),
    ("../aggregation_results/mfu_aggregated_per_config_L40.csv",
     "../benchmark_results/mfu_benchmark_results_L40_128.csv",
     "NVIDIA L40"),
    # Slot reserved for the additional NVIDIA L4 once data is collected:
    # ("../aggregation_results/mfu_aggregated_per_config_L4.csv",
    #  "../benchmark_results/mfu_benchmark_results_L4_128.csv",
    #  "NVIDIA L4"),
    ("../aggregation_results/mfu_aggregated_per_config_GPU06.csv",
     "../benchmark_results/mfu_benchmark_results_GPU06_128.csv",
     "Quadro 5000"),
    ("../aggregation_results/mfu_aggregated_per_config_4070.csv",
     "../benchmark_results/mfu_benchmark_results_4070_128.csv",
     "RTX 4070 Ti"),
    ("../aggregation_results/mfu_aggregated_per_config_MI210.csv",
     "../benchmark_results/mfu_benchmark_results_MI210_128.csv",
     "AMD MI210"),
]

# Hardware to exclude from any GPU-Utilization aggregation/figure.
UTIL_EXCLUDE_HW: set[str] = {"AMD MI210"}

# Hardware for which external power-meter measurements were collected.
# Used by the validation figure (Appendix). Other GPUs were never validated
# externally and must NOT appear in that figure.
EXTERNAL_VALIDATED_HW: set[str] = {"NVIDIA L40", "Quadro 5000"}

# ---------------------------------------------------------------- Palette
# Color-blind-safe (Tol bright / Okabe-Ito blend). Keep stable across
# figures so reviewers can match colors across plots without a per-figure
# legend lookup.
HW_COLORS: dict[str, str] = {
    "NVIDIA A100":  "#332288",  # deep blue
    "NVIDIA L40":   "#117733",  # green
    "NVIDIA L4":    "#88CCEE",  # light blue (reserved)
    "Quadro 5000":  "#DDCC77",  # sand
    "RTX 4070 Ti":  "#CC6677",  # rose
    "AMD MI210":    "#AA4499",  # magenta
}

# Used for predictor-distinguishing colors when a figure is sliced by
# predictor instead of by hardware (e.g. prediction_error boxplots).
PREDICTOR_COLORS: dict[str, str] = {
    "MFU (%)":      "#1f77b4",
    "GPU Util (%)": "#ff7f0e",
}

# Marker shape per dtype (consistent across figures).
DTYPE_SYMBOLS: dict[str, str] = {
    "bfloat16": "circle",
    "float16":  "square",
    "float32":  "diamond",
}

# ---------------------------------------------------------------- Typography
FONT_LABEL  = 16
FONT_TICK   = 12
FONT_FACET  = 16
FONT_LEGEND = 12

# Figure dimensions chosen so that, after \includegraphics width-scaling
# in the paper, all figures end up with the SAME effective font size.
# Ratio is held at ~200 px per target inch (single col 3.4in, double col ~7in).
SINGLE_COL_W = 700
DOUBLE_COL_W = 1400

# ---------------------------------------------------------------- Output
RESULTS_DIR = Path("../results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
