"""Regenerate all figures."""

import runpy
from pathlib import Path

scripts = [
    "external_vs_internal_power.py",
    "predicted_vs_measured_power.py",
    "prediction_error.py",
    "predictor_vs_power.py",
    "predictors_vs_batch.py",
]

here = Path(__file__).parent
for script in scripts:
    print(f"--- {script} ---")
    runpy.run_path(str(here / script), run_name="__main__")
