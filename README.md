# GPU Power Benchmark

A modular framework for benchmarking GPU power draw and Model FLOPs Utilization (MFU) across different hardware architectures and configurations.

## Project Structure

- `benchmark/`: Modules for running benchmarks.
  - `main.py`: Entry point for running benchmarks (Hydra-based).
  - `core.py`: Core benchmarking logic and MFU calculation.
  - `monitors.py`: GPU and external power meter monitoring.
- `analysis/`: Modules for data processing and analysis.
  - `data_loader.py`: Shared hardware configurations and data loading.
  - `aggregate.py`: Consolidates raw benchmark results.
  - `tables.py`: Generates statistical tables and HTML reports.
- `figures/`: Plotting scripts for paper figures.
  - `style.py`: Shared visual styling (colors, fonts).

## Usage

### Running Benchmarks
To run the benchmark suite using Hydra:
```bash
python -m benchmark.main
```

### Aggregating Results
To process raw CSVs from `benchmark_results/` into `aggregation_results/`:
```bash
python -m analysis.aggregate
```

### Generating Tables
To generate analysis tables in `results/`:
```bash
python -m analysis.tables
```

### Generating Figures
To export PDF figures to `results/`:
```bash
python -m figures.prediction_error
python -m figures.predictor_vs_power
python -m figures.external_vs_internal_power
```

## License
MIT
