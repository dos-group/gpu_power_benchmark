# Cross-Hardware Energy Prediction with MFU and GPU Utilization

This repository provides a complete pipeline to benchmark, process, and analyze GPU energy consumption across different hardware platforms and model configurations.

It includes:

* A benchmarking framework to measure training performance, GPU utilization, MFU (Model FLOP Utilization), and power consumption
* Data aggregation scripts to group raw measurements into configuration-level summaries
* Analysis tools to evaluate how well MFU and GPU utilization predict internal GPU power
* Scripts to generate publication-quality figures and tables for research papers

The goal of this project is to understand the relationship between model efficiency metrics (MFU, utilization) and energy usage, and to compare these relationships across GPUs, data types, and workloads.

The repository is designed to support reproducible experiments and generate insights for energy-efficient machine learning.
