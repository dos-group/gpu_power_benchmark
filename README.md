# GPU Power Benchmark

A framework for benchmarking GPU power draw, GPU Utilization, and Model FLOPs Utilization (MFU) across different hardware and training configurations.

## Workflow

Run the scripts in order:

```bash
# 1. Collect raw GPU measurements
python benchmark.py

# 2. Aggregate raw CSVs into per-configuration means
python aggregate.py

# 3. Generate analysis tables (HTML + LaTeX)
python analyze.py

# 4. Generate figures
python create_figures/regenerate_all.py
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

`benchmark.py` can optionally query an external power meter via HTTP. 
Credentials are currently hardcoded in `PowerMeterConfig`.
