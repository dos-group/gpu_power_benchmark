from __future__ import annotations
from pathlib import Path
from analysis.data_loader import PROJECT_ROOT

# ---------------------------------------------------------------- Palette
HW_COLORS: dict[str, str] = {
    "NVIDIA A100":  "#332288",
    "NVIDIA L40":   "#117733",
    "NVIDIA L4":    "#88CCEE",
    "Quadro 5000":  "#DDCC77",
    "RTX 4070 Ti":  "#CC6677",
    "AMD MI210":    "#AA4499",
}

PREDICTOR_COLORS: dict[str, str] = {
    "MFU (%)":      "#1f77b4",
    "GPU Util (%)": "#ff7f0e",
}

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

SINGLE_COL_W = 700
DOUBLE_COL_W = 1400

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
