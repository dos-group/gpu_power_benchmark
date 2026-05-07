# GPU Power Benchmark

A framework for benchmarking GPU power draw and Model FLOPs Utilization (MFU) across different hardware and training configurations. Measures the relationship between MFU / GPU utilization and power consumption for causal language model training.

## Project Structure

```
benchmark.py           # Main benchmark runner (Hydra-based, entry point)
aggregate.py           # Aggregates raw CSVs into per-configuration means
analyze.py             # Statistical analysis and table generation (HTML + LaTeX)
data.py                # Shared data loader used by analyze.py and figures/
figures/
  style.py             # Shared matplotlib styling and color palettes
  prediction_error.py  # Violin plots of power prediction error
  predictor_vs_power.py          # MFU / GPU-util vs power scatter plots
  external_vs_internal_power.py  # External meter vs internal GPU power
benchmark_results/     # Raw per-sample CSVs output by benchmark.py
aggregation_results/   # Per-configuration aggregated CSVs output by aggregate.py
results/               # Figures (PDF) and tables (HTML, LaTeX) output
```

## Pipeline

Run the scripts in order:

```bash
# 1. Collect raw GPU measurements
python benchmark.py

# 2. Aggregate raw CSVs into per-configuration means
python aggregate.py

# 3. Generate analysis tables (HTML + LaTeX) — can run in parallel with step 4
python analyze.py

# 4. Generate figures — can run in parallel with step 3
python figures/prediction_error.py
python figures/predictor_vs_power.py
python figures/external_vs_internal_power.py
```

Steps 3 and 4 both read from `aggregation_results/` and `benchmark_results/` and are independent of each other.

## Configuration

### Benchmark parameters

Edit the `BenchmarkConfig` dataclass in `benchmark.py` to change:
- `models`: list of HuggingFace model IDs to benchmark
- `batch_sizes`: batch sizes to sweep
- `dtype`: precision (`float32`, `float16`, `bfloat16`)
- `context_window`: sequence length
- `gpu_index`: which GPU to benchmark

The Hydra sweep in `AppConfig` controls the cross-product of dtypes, cooldown, warmup, and context window run by default.

### Hardware configurations

`data.py` contains `HW_CONFIGS`, a list of `(aggregation_csv, raw_csv, display_name)` tuples. Add an entry here to include a new GPU in the analysis and figures.

### External power meter

`benchmark.py` can optionally query an external power meter via HTTP. Credentials are currently hardcoded in `PowerMeterConfig` — move the password to an environment variable (`POWER_METER_PASSWORD`) before deploying. The meter password has been committed to git history and should be rotated.

## License

MIT
