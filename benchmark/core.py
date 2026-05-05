import csv
import logging
import time
import re
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from calflops import calculate_flops

# ==================== Configuration Classes ====================

@dataclass
class BenchmarkConfig:
    """Configuration for the MFU benchmark"""
    models: List[str] = field(default_factory=lambda: ['microsoft/DialoGPT-large', 'gpt2-xl', 'Qwen/Qwen2.5-3B'])
    batch_sizes: List[int] = field(default_factory=lambda: [1, 4, 8, 16, 32, 64, 128])
    sampling_interval: float = 1.0
    context_window: int = 1024
    dtype: str = 'float16'
    gpu_index: int = 0
    prompt: str = 'Write a detailed explanation about machine learning and artificial intelligence.'
    output_file: str = 'mfu_benchmark_results.csv'
    append_mode: bool = True
    baseline_samples: int = 50
    cooldown_seconds: float = 2.0
    peak_tflops: float = 0
    learning_rate: float = 1e-4
    warmup_iterations: int = 5
    ci_enable: bool = True
    ci_metric: str = "both"
    ci_rel_width: float = 0.05
    ci_min_samples: int = 50
    ci_max_samples: int = 500
    power_measurement_enabled: bool = True
    power_meter_url: str = "http://powermeter01.cit.tu-berlin.de/status.json"
    power_meter_timeout: float = 5.0
    attention_mechanism: str = "eager"

@dataclass
class AppConfig:
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    hydra: dict = field(default_factory=lambda: {
        "mode": "MULTIRUN",
        "job": {"chdir": False},
        "sweeper": {"params": {
            "benchmark.dtype": "bfloat16, float16, float32",
            "benchmark.cooldown_seconds": "0.0, 5",
            "benchmark.warmup_iterations": "0, 5",
            "benchmark.context_window": "512, 2048"
        }}
    })

@dataclass 
class ModelArchitecture:
    """Model architecture parameters for MFU calculation"""
    num_layers: int
    num_heads: int
    head_dim: int
    num_parameters: float
    
@dataclass
class BenchmarkSample:
    """Single benchmark measurement"""
    timestamp: str
    model_name: str
    batch_size: int
    tokens_per_second: float
    mfu_percentage: float
    mfu_percentage_calflops: float  
    sequence_length: int
    gpu_utilization: int
    memory_utilization: int
    memory_used_mb: int
    temperature_celsius: int
    power_draw_watts: float
    num_layers: int
    num_heads: int
    head_dim: int
    formula_flops_per_token: float  
    calflops_total: float  
    dtype: str
    learning_rate: float
    warmup_iterations: int
    cooldown_seconds: float
    context_window: int
    loss: float
    power_meter_available: bool = False
    power_meter_active_power_w: Optional[float] = None
    power_meter_reactive_power_var: Optional[float] = None
    power_meter_apparent_power_va: Optional[float] = None
    power_meter_timestamp: Optional[str] = None


# ==================== Core Components ====================

class ConvergenceChecker:
    def __init__(self, rel_width: float = 0.05, min_samples: int = 50, max_samples: int = 300, metric: str = "both"):
        self.rel_width = rel_width
        self.min_samples = min_samples
        self.max_samples = max_samples
        self.metric = metric
        self.mfu_values = []
        self.gpu_values = []

    def add_sample(self, mfu: float, gpu_util: int):
        self.mfu_values.append(mfu)
        self.gpu_values.append(gpu_util)

    def _check(self, values):
        n = len(values)
        if n < self.min_samples: return False, 0, 0, 0
        arr = np.array(values)
        mean = float(np.mean(arr))
        if mean < 1e-6: return False, mean, 0, 0
        sem = np.std(arr, ddof=1) / np.sqrt(n)
        ci_half_width = 1.96 * sem
        rel_hw = ci_half_width / mean
        return rel_hw <= self.rel_width, mean, ci_half_width, rel_hw

    def should_stop(self):
        n = len(self.mfu_values)
        if n >= self.max_samples: return True, {'reason': 'max_samples'}
        mfu_conv, mfu_mean, mfu_ci, mfu_rel = self._check(self.mfu_values)
        gpu_conv, gpu_mean, gpu_ci, gpu_rel = self._check(self.gpu_values)
        info = {'num_samples': n, 'mfu_mean': mfu_mean, 'mfu_rel': mfu_rel, 'gpu_mean': gpu_mean, 'gpu_rel': gpu_rel}
        if self.metric == "mfu": stop = mfu_conv
        elif self.metric == "gpu": stop = gpu_conv
        else: stop = mfu_conv and gpu_conv
        info['reason'] = 'converged' if stop else 'not_converged'
        return stop, info

    def reset(self):
        self.mfu_values.clear()
        self.gpu_values.clear()


class MFUCalculator:
    def __init__(self, peak_tflops: float):
        self.peak_flops = peak_tflops * 1e12
    
    def extract_architecture(self, config) -> ModelArchitecture:
        layers = getattr(config, 'num_hidden_layers', getattr(config, 'n_layer', 12))
        heads = getattr(config, 'num_attention_heads', getattr(config, 'n_head', 12))
        hidden = getattr(config, 'hidden_size', getattr(config, 'n_embd', 768))
        params = getattr(config, 'num_parameters', layers * hidden * hidden * 4)
        return ModelArchitecture(int(layers), int(heads), hidden // heads, float(params))

    def calculate_analytical(self, tps: float, arch: ModelArchitecture, seq_len: int) -> Tuple[float, float]:
        fpt = 6.0 * arch.num_parameters + 12.0 * arch.num_layers * arch.num_heads * arch.head_dim * seq_len
        tpt = 3.0 * fpt
        mfu = (tps * tpt) / self.peak_flops
        return min(mfu * 100, 100.0), tpt

    def calculate_empirical(self, model, inputs, tps: float) -> Tuple[float, float]:
        try:
            flops, _, _ = calculate_flops(model=model, kwargs=inputs, print_results=False)
            if isinstance(flops, str):
                match = re.search(r"([0-9]*\.?[0-9]+)", flops)
                val = float(match.group(1)) if match else 0.0
                unit = flops.lower()
                scale = 1e12 if "tflop" in unit else 1e9 if "gflop" in unit else 1e6 if "mflop" in unit else 1.0
                flops_val = val * scale
            else:
                flops_val = float(flops)
            training_flops = flops_val * 3.0
            total_tokens = inputs.get("input_ids").numel()
            fpt = training_flops / total_tokens if total_tokens > 0 else 0
            mfu = (tps * fpt) / self.peak_flops
            return mfu, training_flops
        except Exception:
            return 0.0, 0.0


class DataLogger:
    def __init__(self, output_file: str, append_mode: bool):
        self.path = Path(output_file)
        self.append = append_mode

    def log(self, samples: List[BenchmarkSample]):
        if not samples: return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        exists = self.path.exists() and self.append
        with open(self.path, 'a' if exists else 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(samples[0]).keys()))
            if not exists: writer.writeheader()
            for s in samples:
                row = asdict(s)
                # Simplified sanitization logic
                row = {k: ("" if v is None else v) for k, v in row.items()}
                writer.writerow(row)


class ModelTrainingBenchmark:
    def __init__(self, config: BenchmarkConfig, gpu_monitor, mfu_calculator, power_monitor):
        self.config = config
        self.gpu_monitor = gpu_monitor
        self.mfu_calculator = mfu_calculator
        self.power_monitor = power_monitor
        self.logger = logging.getLogger(__name__)
        self.amp_dtype = torch.float16 if config.dtype == 'float16' else torch.bfloat16 if config.dtype == 'bfloat16' else torch.float32
        self.use_amp = config.dtype in ("float16", "bfloat16")
        self.scaler = GradScaler(enabled=False)
        self.convergence_checker = ConvergenceChecker(
            rel_width=config.ci_rel_width, min_samples=config.ci_min_samples,
            max_samples=config.ci_max_samples, metric=config.ci_metric
        ) if config.ci_enable else None

    def load_model(self, name: str):
        device = torch.device(f'cuda:{self.config.gpu_index}')
        dtype = {'float16': torch.float16, 'bfloat16': torch.bfloat16, 'float32': torch.float32}[self.config.dtype]
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype, device_map={"": device})
        model.config.use_cache = False
        model.train()
        tokenizer = AutoTokenizer.from_pretrained(name)
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        arch = self.mfu_calculator.extract_architecture(model.config)
        opt = torch.optim.AdamW(model.parameters(), lr=self.config.learning_rate)
        return model, tokenizer, arch, opt

    def benchmark_model(self, name: str, batch_size: int) -> List[BenchmarkSample]:
        samples = []
        if self.convergence_checker: self.convergence_checker.reset()
        try:
            model, tokenizer, arch, opt = self.load_model(name)
            device = next(model.parameters()).device
            inputs = tokenizer([self.config.prompt] * batch_size, return_tensors="pt", padding=True, truncation=True, max_length=self.config.context_window)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = inputs['input_ids'].clone()
            
            step_start = torch.cuda.Event(enable_timing=True)
            step_end = torch.cuda.Event(enable_timing=True)
            ms_acc, steps_acc = 0.0, 0
            
            while True:
                step_start.record()
                opt.zero_grad(set_to_none=True)
                with autocast('cuda', dtype=self.amp_dtype, enabled=self.use_amp):
                    loss = model(**inputs, labels=labels).loss
                loss.backward()
                opt.step()
                step_end.record()
                torch.cuda.synchronize()
                
                elapsed = step_start.elapsed_time(step_end)
                ms_acc += elapsed
                steps_acc += 1
                
                if (ms_acc / 1000.0) >= self.config.sampling_interval:
                    tps = (batch_size * inputs['input_ids'].size(1) * steps_acc) / (ms_acc / 1000.0)
                    mfu, fpt = self.mfu_calculator.calculate_analytical(tps, arch, inputs['input_ids'].size(1))
                    mfu_emp, flops_total = self.mfu_calculator.calculate_empirical(model, inputs, tps)
                    
                    gpu_stats = self.gpu_monitor.get_gpu_stats()
                    power_stats = self.power_monitor.get_power_stats()
                    
                    sample = BenchmarkSample(
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        model_name=name, batch_size=batch_size, tokens_per_second=tps,
                        mfu_percentage=mfu, mfu_percentage_calflops=mfu_emp * 100,
                        sequence_length=inputs['input_ids'].size(1),
                        num_layers=arch.num_layers, num_heads=arch.num_heads, head_dim=arch.head_dim,
                        formula_flops_per_token=fpt, calflops_total=flops_total,
                        dtype=self.config.dtype, learning_rate=self.config.learning_rate,
                        warmup_iterations=self.config.warmup_iterations, cooldown_seconds=self.config.cooldown_seconds,
                        context_window=self.config.context_window, loss=float(loss.item()),
                        **gpu_stats, **power_stats
                    )
                    samples.append(sample)
                    
                    if self.convergence_checker:
                        self.convergence_checker.add_sample(mfu_emp * 100, gpu_stats['gpu_utilization'])
                        stop, _ = self.convergence_checker.should_stop()
                        if stop: break
                    
                    ms_acc, steps_acc = 0.0, 0
            
            del model, tokenizer, opt
            torch.cuda.empty_cache()
        except Exception as e:
            self.logger.error(f"Error: {e}")
        return samples


# ==================== Helpers ====================

def _norm_dtype(s: str) -> str:
    s = (s or "").lower()
    mapping = {"float": "float32", "fp32": "float32", "half": "float16", "fp16": "float16", "bf16": "bfloat16"}
    return mapping.get(s, s)

def pick_peak_tflops(device_name: str, dtype: str) -> float:
    name = (device_name or "").lower()
    dt = _norm_dtype(dtype)
    dt_key = {"float32": "fp32", "float16": "fp16", "bfloat16": "bf16"}.get(dt, "fp32")
    gpu_specs = {
        "rtx 4070 ti": {"fp32": 40.1, "fp16": 321.0, "bf16": 321.0},
        "nvidia l40": {"fp32": 90.5, "fp16": 181.05, "bf16": 181.05},
        "nvidia a100": {"fp32": 156, "fp16": 312, "bf16": 312},
        "amd instinct mi210": {"fp32": 22.6, "fp16": 181, "bf16": 181}
    }
    for key, specs in gpu_specs.items():
        if key in name: return specs.get(dt_key, specs["fp32"])
    return 0.0
