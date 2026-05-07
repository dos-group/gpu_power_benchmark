"""Shared figure styling: palette, dimensions, paper rcParams."""

from __future__ import annotations
import matplotlib.pyplot as plt
from data import PROJECT_ROOT

# Hardware display order, with NVIDIA L4 reserved as a placeholder slot.
HW_ORDER = [
    "NVIDIA<br>A100",
    "NVIDIA<br>L40",
    "NVIDIA<br>L4",
    "NVIDIA<br>Quadro 5000",
    "NVIDIA<br>RTX 4070 Ti",
    "AMD<br>MI210",
]

HW_COLORS = {
    "NVIDIA<br>A100":          "#332288",
    "NVIDIA<br>L40":           "#117733",
    "NVIDIA<br>L4":            "#88CCEE",
    "NVIDIA<br>Quadro 5000":   "#DDCC77",
    "NVIDIA<br>RTX 4070 Ti":   "#CC6677",
    "AMD<br>MI210":            "#AA4499",
}

DTYPE_COLORS = {
    "float32":  "#1f77b4",
    "float16":  "#ff7f0e",
    "bfloat16": "#2ca02c",
}

PREDICTOR_COLORS = {
    "MFU (%)":      "#1f77b4",
    "GPU Util (%)": "#ff7f0e",
}

# IEEE conference geometry (inches). Keep both single-column figures at
# the same width so they print at the same effective font size.
SINGLE_COL_W = 3.4
DOUBLE_COL_W = 7.0
ROW_H        = 1.1   # per-row height in stacked single-column grids

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def label(hw: str, sep: str = "\n") -> str:
    """Render a hardware name for a plot label."""
    return hw.replace("<br>", sep)


def set_paper_style() -> None:
    plt.rcParams.update({
        "font.family":     "sans-serif",
        "font.size":       8,
        "axes.labelsize":  8,
        "axes.titlesize":  8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "lines.linewidth": 1.2,
        "axes.grid":       True,
        "grid.alpha":      0.3,
        "axes.axisbelow":  True,
        "pdf.fonttype":    42,
        "ps.fonttype":     42,
    })
