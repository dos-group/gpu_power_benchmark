#!/usr/bin/env python3

import csv
import json
import logging
import subprocess
import time
import traceback
import hydra
from dataclasses import dataclass, asdict, field
from omegaconf import DictConfig, OmegaConf
from hydra.core.config_store import ConfigStore
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, cast
import re
import shutil
import os

import numpy as np
import requests
from requests.exceptions import RequestException, Timeout

try:
    import pynvml as nvml
except Exception:
    nvml = None

import torch
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    BatchEncoding,
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
    ci_enable: bool = True                  # turn on/off early stopping
    ci_metric: str = "both"                 # "mfu", "gpu", or "both"
    ci_rel_width: float = 0.05              # 5% relative half-width
    ci_min_samples: int = 50                # don't check before this many samples
    ci_max_samples: int = 500               # hard cap to avoid very long runs

    power_measurement_enabled: bool = True
    power_meter_url: str = "http://powermeter01.cit.tu-berlin.de/status.json"
    power_meter_timeout: float = 5.0

    attention_mechanism: str = "eager"  # sdpa, eager, flash_attention

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
    

@dataclass
class AppConfig:
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)

    hydra: dict = field(default_factory=lambda: {
        "mode": "MULTIRUN",          # same as running with -m
        "job": {"chdir": False},     # keep CWD so files land where you launch from
        "sweeper": {
            "params": {
                # Comma-separated choices => Hydra takes the cross product
                "benchmark.dtype": "bfloat16, float16, float32",                  
                "benchmark.cooldown_seconds": "5, 5",                
                "benchmark.warmup_iterations": "5, 5",                   
                "benchmark.context_window": "512, 2048"
            }
        }
    })

cs = ConfigStore.instance()
cs.store(name="config", node=AppConfig())


# ==================== Inlined Modules ====================

@dataclass
class ConvergenceChecker:
    """Efficient convergence detection using confidence intervals"""

    rel_width: float = 0.05
    min_samples: int = 50
    max_samples: int = 300
    metric: str = "both"

    def __post_init__(self):
        self.mfu_values: List[float] = []
        self.gpu_values: List[float] = []

    def add_sample(self, mfu: float, gpu_util: int) -> None:
        self.mfu_values.append(mfu)
        self.gpu_values.append(gpu_util)

    def _check_convergence(self, values: List[float]) -> Tuple[bool, float, float, float]:
        n = len(values)
        if n < self.min_samples:
            return False, 0.0, 0.0, 0.0

        arr = np.array(values)
        mean = float(np.mean(arr))
        if mean < 1e-6:
            return False, mean, 0.0, 0.0

        sem = np.std(arr, ddof=1) / np.sqrt(n)
        ci_half_width = 1.96 * sem
        rel_hw = ci_half_width / mean
        converged = rel_hw <= self.rel_width
        return converged, mean, ci_half_width, rel_hw

    def should_stop(self) -> Tuple[bool, dict]:
        n = len(self.mfu_values)
        if n >= self.max_samples:
            return True, {'reason': 'max_samples_reached', 'num_samples': n}

        if n < self.min_samples:
            return False, {'reason': 'collecting_samples', 'num_samples': n, 'min_required': self.min_samples}

        mfu_conv, mfu_mean, mfu_ci, mfu_rel = self._check_convergence(self.mfu_values)
        gpu_conv, gpu_mean, gpu_ci, gpu_rel = self._check_convergence(self.gpu_values)

        info = {
            'num_samples': n,
            'mfu_mean': mfu_mean,
            'mfu_ci_half_width': mfu_ci,
            'mfu_rel_width': mfu_rel,
            'mfu_converged': mfu_conv,
            'gpu_mean': gpu_mean,
            'gpu_ci_half_width': gpu_ci,
            'gpu_rel_width': gpu_rel,
            'gpu_converged': gpu_conv,
        }

        if self.metric == "mfu":
            should_stop = mfu_conv
            info['reason'] = 'mfu_converged' if should_stop else 'mfu_not_converged'
        elif self.metric == "gpu":
            should_stop = gpu_conv
            info['reason'] = 'gpu_converged' if should_stop else 'gpu_not_converged'
        else:
            should_stop = mfu_conv and gpu_conv
            if should_stop:
                info['reason'] = 'both_converged'
            elif mfu_conv:
                info['reason'] = 'only_mfu_converged'
            elif gpu_conv:
                info['reason'] = 'only_gpu_converged'
            else:
                info['reason'] = 'neither_converged'

        return should_stop, info

    def reset(self) -> None:
        self.mfu_values.clear()
        self.gpu_values.clear()


@dataclass
class PowerMeterConfig:
    url: str = "http://powermeter01.cit.tu-berlin.de/status.json"
    user: str = "admin"
    password: str = "#Ofumdad12167"  # TODO: load from env (POWER_METER_PASSWORD)
    components: str = "9029395"
    referer: str = "http://powermeter01.cit.tu-berlin.de/dashboard.html"
    sessionrs: str = "5"
    sessionttl: str = "600"
    timeout_s: float = 5.0


class PowerMeter:
    def __init__(self, cfg: PowerMeterConfig):
        self.cfg = cfg
        self._auth = (cfg.user, cfg.password)
        self._headers = {"Referer": cfg.referer}
        self._cookies = {"sessionrs": cfg.sessionrs, "sessionttl": cfg.sessionttl}
        self._payload = {"components": cfg.components}

    def get_status(self) -> dict:
        try:
            r = requests.post(
                self.cfg.url,
                auth=self._auth,
                headers=self._headers,
                cookies=self._cookies,
                data=self._payload,
                timeout=self.cfg.timeout_s,
            )
            r.raise_for_status()
            data = r.json()
        except (RequestException, Timeout):
            return {
                "active_power_w": None,
                "reactive_power_var": None,
                "apparent_power_va": None,
                "timestamp": None,
            }

        try:
            ac = float(data["sensor_values"][1]["values"][2][4]["v"])
            re_ = float(data["sensor_values"][1]["values"][2][5]["v"])
            ap = float(data["sensor_values"][1]["values"][2][6]["v"])
        except Exception:
            ac, re_, ap = 0.0, 0.0, 0.0

        return {
            "active_power_w": ac,
            "reactive_power_var": re_,
            "apparent_power_va": ap,
            "timestamp": datetime.utcnow().isoformat(),
        }


# ==================== Utilities ====================
def _norm_dtype(s: str) -> str:
    """Normalize user/config dtype aliases to canonical names."""
    s = (s or "").lower()
    mapping = {
        "float": "float32", "fp32": "float32", "float32": "float32",
        "half": "float16", "fp16": "float16", "float16": "float16",
        "bf16": "bfloat16", "bfloat": "bfloat16", "bfloat16": "bfloat16",
    }
    return mapping.get(s, s)

def pick_peak_tflops(device_name: str, dtype: str) -> float:
    """Return peak TFLOPS for (device_name, dtype) used in MFU normalization."""
    name = (device_name or "").lower()
    dtype = _norm_dtype(dtype)
    dtype_key = {"float32": "fp32", "float16": "fp16", "bfloat16": "bf16"}.get(dtype, "fp32")
    supports_bf16 = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())

    gpu_specs = {
        "rtx 4070 ti": {"fp32": 40.1, "fp16": 80.2, "bf16": 80.2},
        "quadro rtx 5000": {"fp32": 89.2, "fp16": 89.2, "bf16": None},
        "nvidia l40": {"fp32": 90.5, "fp16": 181.05, "bf16": 181.05},
        "nvidia l4": {"fp32": 30.3, "fp16": 121.0, "bf16": 121.0},
        "nvidia a100": {"fp32": 156, "fp16": 312, "bf16": 312},
        "amd instinct mi210": {"fp32": 181, "fp16": 181, "bf16": 181}
    }

    for key, specs in gpu_specs.items():
        if key in name:
            if dtype_key == "bf16" and (not supports_bf16 or specs["bf16"] is None):
                logging.warning(f"BF16 requested on {device_name}: using FP32 peak for MFU.")
                return specs["fp32"]
            return specs.get(dtype_key, specs["fp32"]) or specs["fp32"]

    logging.warning(f"Unknown GPU '{device_name}', returning 0 TFLOPS.")
    return 0.0

# ==================== GPU Monitoring ====================

class GPUMonitor:
    """Handles GPU statistics collection for both NVIDIA and AMD GPUs."""

    def __init__(self, gpu_index: int = 0):
        self.gpu_index = gpu_index
        self.logger = logging.getLogger(__name__)
        self.mode: Optional[str] = None  # "nvml", "nvidia_smi", "rocm_smi"
        self.nvidia_smi_cmd: Optional[str] = None
        self.rocm_smi_cmd: Optional[str] = None
        self.handle = None
        self._energy_supported = False

        if nvml is not None:
            try:
                nvml.nvmlInit()
                self.handle = nvml.nvmlDeviceGetHandleByIndex(int(self.gpu_index))
                name = nvml.nvmlDeviceGetName(self.handle)
                self.mode = "nvml"
                self.logger.info(
                    "GPUMonitor: using NVML backend on GPU %d (%s)",
                    self.gpu_index,
                    name.decode("utf-8") if isinstance(name, bytes) else name,
                )
                try:
                    nvml.nvmlDeviceGetTotalEnergyConsumption(self.handle)
                    self._energy_supported = True
                except Exception:
                    self._energy_supported = False
                return
            except Exception as e:
                self.logger.warning(
                    "GPUMonitor: NVML init or handle failed (%s); falling back to CLI backends.",
                    e,
                )

        for candidate in [
            "nvidia-smi",
            "/usr/bin/nvidia-smi",
            "/usr/local/bin/nvidia-smi",
            "/usr/lib/nvidia-smi",
        ]:
            if not (shutil.which(candidate) or os.path.exists(candidate)):
                continue
            try:
                result = subprocess.run(
                    [candidate, "--query-gpu=name", "--format=csv,noheader,nounits"],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5, check=True,
                )
                if result.stdout.strip():
                    self.mode = "nvidia_smi"
                    self.nvidia_smi_cmd = candidate
                    self.logger.info(
                        "GPUMonitor: using nvidia-smi backend (%s) for GPU stats.",
                        candidate,
                    )
                    return
            except Exception:
                pass

        for candidate in [
            "rocm-smi",
            "amd-smi",
            "/opt/rocm/bin/rocm-smi",
            "/opt/rocm/bin/amd-smi",
        ]:
            if not (shutil.which(candidate) or os.path.exists(candidate)):
                continue
            try:
                result = subprocess.run(
                    [candidate, "--showuse", "--json"],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5, check=True,
                )
                if result.stdout.strip():
                    self.mode = "rocm_smi"
                    self.rocm_smi_cmd = candidate
                    self.logger.info(
                        "GPUMonitor: using %s backend for AMD GPU stats.",
                        candidate,
                    )
                    return
            except Exception:
                pass

        raise RuntimeError("GPUMonitor: no usable GPU monitoring backend found.")

    def supports_energy_counter(self) -> bool:
        return self.mode == "nvml" and self.handle is not None and self._energy_supported

    def read_total_energy_mj(self) -> Optional[int]:
        if not self.supports_energy_counter():
            return None
        try:
            return int(nvml.nvmlDeviceGetTotalEnergyConsumption(self.handle))       #type: ignore
        except Exception as e:
            self.logger.warning("GPUMonitor (NVML): failed to read total energy: %s", e)
            return None

    def shutdown(self) -> None:
        if self.mode == "nvml" and nvml is not None:
            try:
                nvml.nvmlShutdown()
            except Exception:
                pass

    def get_gpu_stats(self) -> Dict:
        stats = {
            "gpu_utilization": 0,
            "memory_utilization": 0,
            "memory_used_mb": 0,
            "temperature_celsius": 0,
            "power_draw_watts": 0.0,
        }

        if self.mode == "nvml" and self.handle is not None and nvml is not None:
            try:
                util = nvml.nvmlDeviceGetUtilizationRates(self.handle)
                mem_info = nvml.nvmlDeviceGetMemoryInfo(self.handle)
                temp = nvml.nvmlDeviceGetTemperature(self.handle, nvml.NVML_TEMPERATURE_GPU)
                power_mw = nvml.nvmlDeviceGetPowerUsage(self.handle)

                stats["gpu_utilization"] = int(util.gpu) if util.gpu is not None else 0
                if mem_info.total > 0:                                                                      #type: ignore
                    stats["memory_utilization"] = int(round(100.0 * mem_info.used / mem_info.total))        #type: ignore
                stats["memory_used_mb"] = int(mem_info.used / (1024 * 1024))                                #type: ignore
                stats["temperature_celsius"] = int(temp)
                stats["power_draw_watts"] = float(power_mw) / 1000.0
            except Exception as e:
                self.logger.warning(f"GPUMonitor (NVML): failed to read stats: {e}")
            return stats

        if self.mode == "nvidia_smi" and self.nvidia_smi_cmd:
            try:
                cmd = [
                    self.nvidia_smi_cmd, "-i", str(self.gpu_index),
                    "--query-gpu=utilization.gpu,utilization.memory,memory.used,temperature.gpu,power.draw",
                    "--format=csv,noheader,nounits",
                ]
                output = subprocess.check_output(cmd, text=True, timeout=5).strip()
                values = [val.strip() for val in output.split(",")]
                stats["gpu_utilization"] = int(values[0]) if values[0] != "[Not Supported]" else 0
                stats["memory_utilization"] = int(values[1]) if values[1] != "[Not Supported]" else 0
                stats["memory_used_mb"] = int(values[2]) if values[2] != "[Not Supported]" else 0
                stats["temperature_celsius"] = int(values[3]) if values[3] != "[Not Supported]" else 0
                stats["power_draw_watts"] = float(values[4]) if values[4] != "[Not Supported]" else 0.0
            except Exception as e:
                self.logger.warning(f"GPUMonitor (nvidia-smi): failed to read stats: {e}")
            return stats

        if self.mode == "rocm_smi" and self.rocm_smi_cmd:
            try:
                cmd = [
                    self.rocm_smi_cmd, "-d", str(self.gpu_index),
                    "--showuse", "--showmemuse", "--showtemp", "--showpower", "--json",
                ]
                output = subprocess.check_output(cmd, text=True, timeout=5)
                start = output.find("{")
                end = output.rfind("}")
                if start == -1 or end == -1 or end <= start:
                    return stats
                data = json.loads(output[start:end + 1])
                card_key = sorted(data.keys())[0]
                card = data[card_key]
                util_str = card.get("GPU use (%)") or card.get("Current GPU use") or card.get("GPU use")
                if util_str:
                    stats["gpu_utilization"] = int(float(str(util_str).strip().rstrip("%")))
                mem_util_str = card.get("GPU memory use (%)") or card.get("GPU Memory use (%)")
                if mem_util_str:
                    stats["memory_utilization"] = int(float(str(mem_util_str).strip().rstrip("%")))
                mem_used_str = card.get("GPU memory use") or card.get("GPU Memory use")
                if mem_used_str:
                    parts = str(mem_used_str).split()
                    if parts:
                        stats["memory_used_mb"] = int(float(parts[0]))
                temp_str = card.get("Temperature (Sensor #1) (C)") or card.get("Temperature (Sensor #1)") or card.get("Temperature")
                if temp_str:
                    stats["temperature_celsius"] = int(float(str(temp_str).split()[0]))
                power_str = card.get("Average Graphics Package Power (W)") or card.get("Average Graphics Package Power") or card.get("GPU power (W)")
                if power_str:
                    stats["power_draw_watts"] = float(str(power_str).replace("W", "").strip())
            except Exception as e:
                self.logger.warning(f"GPUMonitor (rocm-smi): failed to read stats: {e}")
            return stats

        raise RuntimeError("GPUMonitor: no mode set.")


# ==================== Power Measurement Monitor ====================


class ExternalPowerInterface:
    """Handles power measurement from external power meter"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.power_meter: Optional["PowerMeter"] = None
        self.is_available = False
        self.logger = logging.getLogger(__name__)
        
        if self.config.power_measurement_enabled:
            self._initialize_power_meter()
    
    def _initialize_power_meter(self) -> None:
        """Initialize power meter and check availability"""
        try:
            power_cfg = PowerMeterConfig(
                url=self.config.power_meter_url,
                timeout_s=self.config.power_meter_timeout
            )
            self.power_meter = PowerMeter(power_cfg)
            
            # Test connection
            test_status = self.power_meter.get_status()
            if test_status['active_power_w'] is not None:
                self.is_available = True
                self.logger.info(
                    f"Power meter connected successfully at {self.config.power_meter_url}"
                )
            else:
                self.is_available = False
                self.logger.warning(
                    "Power meter URL reachable but returned no data. "
                    "Power measurements will be empty."
                )
        except Exception as e:
            self.is_available = False
            self.logger.warning(
                f"Power meter not available at {self.config.power_meter_url}: {e}. "
                "Power measurements will be empty."
            )
    
    def get_power_stats(self) -> Dict:
        """Get current power statistics."""
        base_stats = {
            'power_meter_active_power_w': None,
            'power_meter_reactive_power_var': None,
            'power_meter_apparent_power_va': None,
            'power_meter_timestamp': None,
            'power_meter_available': False,
        }

        # Early exit if meter not enabled/available
        if not (self.config.power_measurement_enabled and self.is_available and self.power_meter):
            return base_stats

        try:
            status = self.power_meter.get_status()
            return {
                **base_stats,
                'power_meter_active_power_w': status.get('active_power_w'),
                'power_meter_reactive_power_var': status.get('reactive_power_var'),
                'power_meter_apparent_power_va': status.get('apparent_power_va'),
                'power_meter_timestamp': status.get('timestamp'),
                'power_meter_available': True,
            }
        except Exception as e:
            self.logger.warning(f"Failed to get power measurement: {e}")
            return base_stats


# ==================== MFU Calculator ====================

class MFUCalculator:
    """Calculates Model FLOP Utilization for Training"""
    
    def __init__(self, peak_tflops: float):
        self.peak_flops = peak_tflops * 1e12
    
    def extract_model_architecture(self, model_config) -> ModelArchitecture:
        """Extract architecture parameters from model config"""
        # Try different config attribute names across model families
        num_layers = (
            getattr(model_config, 'num_hidden_layers', None) or
            getattr(model_config, 'n_layer', None) or
            getattr(model_config, 'num_layers', None) or 12
        )
        
        num_heads = (
            getattr(model_config, 'num_attention_heads', None) or
            getattr(model_config, 'n_head', None) or
            getattr(model_config, 'num_heads', None) or 12
        )
        
        hidden_size = (
            getattr(model_config, 'hidden_size', None) or
            getattr(model_config, 'n_embd', None) or
            getattr(model_config, 'd_model', None) or 768
        )
        
        head_dim = hidden_size // num_heads
        
        # Get actual parameter count from model
        num_parameters = getattr(model_config, 'num_parameters', 0)
        if num_parameters == 0:
            # Rough estimation for transformer models
            num_parameters = num_layers * hidden_size * hidden_size * 4  # Approximate
        
        return ModelArchitecture(
            num_layers=int(num_layers),
            num_heads=int(num_heads), 
            head_dim=int(head_dim),
            num_parameters=float(num_parameters)
        )
    

    def calculate_analytical_mfu(self, tokens_per_second: float, architecture: ModelArchitecture, sequence_length: int) -> Tuple[float, float]:
        forward_flops_per_token = (
            6.0 * architecture.num_parameters +  # 6N term
            12.0 * architecture.num_layers * architecture.num_heads * 
            architecture.head_dim * sequence_length  # 12LHdT term
        )
        
        training_flops_per_token = 3.0 * forward_flops_per_token
        
        total_flops_per_second = tokens_per_second * training_flops_per_token
        mfu = total_flops_per_second / self.peak_flops
        
        return min(mfu * 100, 100.0), training_flops_per_token  # Return as percentage, cap at 100%
    

    def calculate_empirical_mfu(self, model, inputs, tokens_per_second: float) -> Tuple[float, float]:
        try:
            # Use calflops to measure model FLOPs
            # For training, we need to account for forward + backward pass
            flops, _, _ = calculate_flops(
                model=model,
                kwargs=inputs,
                print_results=False,
                print_detailed=False
            )
            
            flops_value = 0.0
            if isinstance(flops, str):
                match = re.search(r"([0-9]*\.?[0-9]+)", flops)
                if match:
                    flops_value = float(match.group(1))
                    unit = flops.lower()
                    scale = (
                        1e12 if "tflop" in unit else
                        1e9  if "gflop" in unit else
                        1e6  if "mflop" in unit else
                        1e3  if "kflop" in unit else
                        1.0
                    )
                    flops_value *= scale

            else:
                flops_value = float(flops)

            # calflops typically returns forward pass FLOPs
            # For training, multiply by ~3 (forward + backward + optimizer)
            training_flops_total = flops_value * 3.0
            
            total_tokens = inputs.get("input_ids").numel()          # numel give the total number of elements in the input tensor.
            
            flops_per_token = training_flops_total / total_tokens if total_tokens > 0 else 0
            
            # Calculate MFU
            total_flops_per_second = tokens_per_second * flops_per_token
            mfu = total_flops_per_second / self.peak_flops
            
            return mfu, training_flops_total
            
        except Exception as e:
            logging.warning(f"calflops calculation failed: {e}")
            return 0.0, 0.0


# ==================== Data Logger ====================

def sanitize_sample_values(sample: dict, align_tolerance_s: float = 0.5) -> dict:
    """Replace implausible numeric values with None and check time alignment."""
    bounds = {
        "gpu_utilization": (0, 100),
        "memory_utilization": (0, 100),
        "temperature_celsius": (0, 110),
        "power_draw_watts": (0, 1000),           
        "power_meter_active_power_w": (0, 2000), 
        "mfu_percentage": (0, 100),
        "mfu_percentage_calflops": (0, 100),
        "tokens_per_second": (0, 1e8),
    }

    # --- Basic numeric plausibility ---
    for k, (lo, hi) in bounds.items():
        v = sample.get(k)
        if isinstance(v, (int, float)):
            if v < lo or v > hi or (v != v):  # NaN check
                sample[k] = None

    # --- Timestamp alignment check ---
    bench_ts = sample.get("timestamp")
    pm_ts = sample.get("power_meter_timestamp")
    if bench_ts and pm_ts:
        try:
            b = datetime.fromisoformat(str(bench_ts))
            p = datetime.fromisoformat(str(pm_ts))
            if p.tzinfo is None:
                p = p.replace(tzinfo=timezone.utc)
            if b.tzinfo is None:
                b = b.replace(tzinfo=timezone.utc)
            dt = abs((b - p).total_seconds())
            if dt > align_tolerance_s:
                # misaligned -> drop power data
                sample["power_meter_active_power_w"] = None
                sample["power_meter_reactive_power_var"] = None
                sample["power_meter_apparent_power_va"] = None
        except Exception:
            # malformed timestamp -> drop power data
            sample["power_meter_active_power_w"] = None
            sample["power_meter_reactive_power_var"] = None
            sample["power_meter_apparent_power_va"] = None

    return sample

class DataLogger:


    def __init__(self, output_file: str, append_mode: bool, logger: Optional[logging.Logger] = None):
        self.output_path = Path(output_file)
        self.append_mode = append_mode
        self.logger = logger or logging.getLogger(__name__)

    def save_samples(self, samples: List[BenchmarkSample]) -> None:
        try:
            if not samples:
                self.logger.info("---------------Samples gone----------------")
                return

            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            file_exists = self.output_path.exists() and self.append_mode

            
            self.logger.info("CSV: writing %d samples to %s (append=%s)",
                            len(samples), str(self.output_path.resolve()), file_exists)

            with open(self.output_path, 'a' if file_exists else 'w', newline='') as csvfile:
                fieldnames = list(asdict(samples[0]).keys())
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                if not file_exists:
                    writer.writeheader()

                for sample in samples:
                    row = asdict(sample)
                    row = sanitize_sample_values(row)

                    if isinstance(row.get("loss"), torch.Tensor):
                        row["loss"] = float(row["loss"].detach().item())
                    row = {k: ("" if v is None else v) for k, v in row.items()}
                    writer.writerow(row)

            self.logger.info(f"Saved {len(samples)} samples to {self.output_path.resolve()}")
        except Exception as e:
            
            self.logger.exception("ERROR writing CSV: %s", e)

# ==================== Model Training Benchmark ====================
class ModelTrainingBenchmark:
    """Handles individual model training benchmarking"""
    
    def __init__(self, config: BenchmarkConfig, gpu_monitor: GPUMonitor, 
                 mfu_calculator: MFUCalculator, power_monitor: ExternalPowerInterface):
        self.config = config
        self.gpu_monitor = gpu_monitor
        self.mfu_calculator = mfu_calculator
        self.power_monitor = power_monitor
        self.logger = logging.getLogger(__name__)

        self.use_amp = self.config.dtype in ("float16", "bfloat16")
        if self.config.dtype == "float16":
            self.amp_dtype = torch.float16
        elif self.config.dtype == "bfloat16":
            self.amp_dtype = torch.bfloat16
        else:
            self.amp_dtype = torch.float32

        # Always disable GradScaler, regardless of dtype
        self.scaler = GradScaler(enabled=False)


        # Add convergence checker
        if self.config.ci_enable:
            self.convergence_checker = ConvergenceChecker(
                rel_width=self.config.ci_rel_width,
                min_samples=self.config.ci_min_samples,
                max_samples=self.config.ci_max_samples,
                metric=self.config.ci_metric
            )
        else:
            self.convergence_checker = None
        
    def load_model_and_tokenizer(self, model_name: str) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase, ModelArchitecture, torch.optim.Optimizer]:
        """Load model, tokenizer, extract architecture and setup optimizer"""
        
        dtype_map = {
            'float16': torch.float16,
            'bfloat16': torch.bfloat16, 
            'float32': torch.float32
        }
        
        device_str = f'cuda:{self.config.gpu_index}'
        device = torch.device(device_str)
        dtype = dtype_map[self.config.dtype]                                # pass in from_pretrained for real fp16 weights
        
        # Load model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map={"": device},                                                # don't pass a torch.device here for GPU load
        )
        model.to(device)                                                    # type: ignore[attr-defined]

        model.config.use_cache = False

        # Use TF32 on Ampere/Ada; numerically safe for training
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

        # Avoid huge fused SDPA kernels that can exceed launch limits (Windows/long seq)
        try:
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
        except Exception:
            pass

        try:
            model.config._attn_implementation = self.config.attention_mechanism  
        except Exception:
            pass
                
        # Set model to training mode
        model.train()
        
        tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        # Extract architecture
        architecture = self.mfu_calculator.extract_model_architecture(model.config)
        
        # Setup optimizer
        optimizer: torch.optim.Optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=self.config.learning_rate,
            weight_decay=0.01,
            fused=torch.cuda.is_available()
        )
        
        return model, tokenizer, architecture, optimizer
    
    def warmup_model(self, model, tokenizer, optimizer, device) -> None:
        """Warmup the model with a few training steps"""
        model.train()
        
        # Create dummy input for warmup
        warmup_text = "This is a warmup text for the model training benchmark."
        inputs = tokenizer(
            [warmup_text], 
            return_tensors='pt', 
            padding=True, 
            truncation=True,
            max_length=min(128, self.config.context_window)
        )

        # Non-blocking H→D copies (small win, but keeps pipeline GPU-centric)
        inputs = {k: v.pin_memory().to(device, non_blocking=True) for k, v in inputs.items()}
        labels = inputs["input_ids"]
        
        for _ in range(self.config.warmup_iterations):
            optimizer.zero_grad(set_to_none=True)

            if self.scaler.is_enabled():
                with autocast('cuda', dtype=self.amp_dtype, enabled=True):
                    outputs = model(**inputs, labels=labels)
                    loss = outputs.loss
                self.scaler.scale(loss).backward()
                # Optional: clip after unscale to avoid NaNs

                """ self.scaler.unscale_(optimizer) """

                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                with autocast('cuda', dtype=self.amp_dtype, enabled=self.use_amp):
                    outputs = model(**inputs, labels=labels)
                    loss = outputs.loss
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
        torch.cuda.synchronize()
    
    def benchmark_model(self, model_name: str, batch_size: int) -> List[BenchmarkSample]:
        """Benchmark a single model with given batch size for training"""
        samples = []

        # Reset convergence checker for new configuration
        if self.convergence_checker:
            self.convergence_checker.reset()
        
        try:
            model, tokenizer, architecture, optimizer = self.load_model_and_tokenizer(model_name)
            device = next(model.parameters()).device
            

            # Warmup
            self.warmup_model(model, tokenizer, optimizer, device)
            
            # Prepare training inputs
            encoding: BatchEncoding = tokenizer(
                [self.config.prompt] * batch_size,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.context_window,
            )

            inputs: Dict[str, torch.Tensor] = {k: v.pin_memory().to(device, non_blocking=True) for k, v in encoding.items()}
            
            # For training, we use the inputs as both input and labels
            labels = inputs['input_ids'].clone()
            
            sequence_length = inputs['input_ids'].size(1)
            
            self.logger.info(
                "RUN START | model=%s bs=%d dtype=%s lr=%g warmup=%d cooldown=%.1f context=%d",
                model_name, batch_size, self.config.dtype, self.config.learning_rate,
                self.config.warmup_iterations, self.config.cooldown_seconds,
                self.config.context_window
            )
            

            tokens_per_step = batch_size * sequence_length

            # Reusable CUDA events
            step_start_evt = torch.cuda.Event(enable_timing=True)
            step_end_evt   = torch.cuda.Event(enable_timing=True)

            # Accumulators for a sampling window measured in GPU time
            ms_step_acc = 0.0
            steps_acc   = 0
            util_ms_acc = 0.0

            cur_stream = torch.cuda.current_stream(device=device)

            energy_start_mj = self.gpu_monitor.read_total_energy_mj()

            while True:
                # ----- Start GPU-timed region: forward + backward + optimizer -----
                step_start_evt.record(cur_stream)

                # Zero grads once per iteration (start-of-iter)
                optimizer.zero_grad(set_to_none=True)

                if self.scaler.is_enabled():
                    with autocast('cuda', dtype=self.amp_dtype, enabled=True):
                        outputs = model(**inputs, labels=labels)
                        loss = outputs.loss
                    self.scaler.scale(loss).backward()
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    with autocast('cuda', dtype=self.amp_dtype, enabled=self.use_amp):
                        outputs = model(**inputs, labels=labels)
                        loss = outputs.loss
                    loss.backward()
                    optimizer.step()

                step_end_evt.record(cur_stream)
                torch.cuda.synchronize()
                elapsed_ms = step_start_evt.elapsed_time(step_end_evt)  # fwd+bwd+step
                util_stats = self.gpu_monitor.get_gpu_stats()
                util_now = util_stats.get("gpu_utilization", 0)
                util_ms_acc += float(util_now) * elapsed_ms

                # ---- accumulate GPU-timed window + power for this step ----
                ms_step_acc += elapsed_ms
                steps_acc   += 1

                # Emit a sample once enough GPU time has accumulated
                if (ms_step_acc / 1000.0) >= self.config.sampling_interval:
                    # Step-inclusive tokens/sec (forward + backward + optimizer)
                    tokens_per_second = (tokens_per_step * steps_acc) / (ms_step_acc / 1000.0)

                    # ---- MFU using step-inclusive TPS ----
                    mfu_percentage, flops_per_token = self.mfu_calculator.calculate_analytical_mfu(
                        tokens_per_second, architecture, sequence_length
                    )
                    mfu_calflops_frac, calflops_total = self.mfu_calculator.calculate_empirical_mfu(
                        model, inputs, tokens_per_second
                    )
                    mfu_calflops_percent = mfu_calflops_frac * 100.0

                    gpu_stats = self.gpu_monitor.get_gpu_stats()
                    power_stats = self.power_monitor.get_power_stats()

                    energy_end_mj = self.gpu_monitor.read_total_energy_mj()
                    window_s = ms_step_acc / 1000.0
                    if energy_start_mj is not None and energy_end_mj is not None and window_s > 0:
                        avg_power_w = ((energy_end_mj - energy_start_mj) / 1000.0) / max(1e-9, window_s)
                        gpu_stats['power_draw_watts'] = float(avg_power_w)
                        energy_start_mj = energy_end_mj

                    avg_gpu_util = (util_ms_acc / ms_step_acc) if ms_step_acc > 0 else 0.0
                    gpu_stats['gpu_utilization'] = int(round(avg_gpu_util))

                    loss_value = float(loss.detach().item()) if hasattr(loss, "item") else float(loss)

                    # Create sample
                    sample = BenchmarkSample(
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        model_name=model_name,
                        batch_size=batch_size,
                        tokens_per_second=tokens_per_second,
                        mfu_percentage=mfu_percentage,
                        sequence_length=sequence_length,
                        num_layers=architecture.num_layers,
                        num_heads=architecture.num_heads,
                        head_dim=architecture.head_dim,
                        calflops_total=calflops_total,
                        mfu_percentage_calflops=mfu_calflops_percent,
                        formula_flops_per_token=flops_per_token,
                        **gpu_stats,  # contains averaged power_draw_watts
                        dtype=self.config.dtype,
                        learning_rate=self.config.learning_rate,
                        warmup_iterations=self.config.warmup_iterations,
                        cooldown_seconds=self.config.cooldown_seconds,
                        context_window=self.config.context_window,
                        loss=loss_value,
                        **power_stats
                    )
                    samples.append(sample)

                    # NEW: Check convergence
                    if self.convergence_checker:
                        self.convergence_checker.add_sample(
                            mfu_calflops_percent, 
                            gpu_stats['gpu_utilization']
                        )
                        should_stop, info = self.convergence_checker.should_stop()
                        
                        if should_stop:
                            self.logger.info(
                                f"Converged: {info['reason']} | "
                                f"Samples={info['num_samples']} | "
                                f"MFU={info['mfu_mean']:.2f}±{info['mfu_ci_half_width']:.2f}% "
                                f"(rel_width={info['mfu_rel_width']:.4f}) | "
                                f"GPU={info['gpu_mean']:.1f}±{info['gpu_ci_half_width']:.1f}% "
                                f"(rel_width={info['gpu_rel_width']:.4f})"
                            )
                            break
                    
                    # Log progress
                    self.logger.info(
                        f"{model_name} | bs={batch_size} | steps in window={steps_acc} | dtype={self.config.dtype} |"
                        f"LR={self.config.learning_rate} | Warmup={self.config.warmup_iterations} | Cooldown={self.config.cooldown_seconds} |"
                        f"Context={self.config.context_window} | TPS (throughput)={tokens_per_second:.2f} | Formula MFU={mfu_percentage:.2f}% | "
                        f"Powermeter W={power_stats['power_meter_active_power_w']} | SMI W={sample.power_draw_watts} | "
                        f"Calfops Total (one training step)={calflops_total:.2f} | Calflops MFU={mfu_calflops_percent:.2f}% | "
                        f"loss={loss_value:.4f} | GPU={gpu_stats['gpu_utilization']}%"
                    )

                    # Reset window accumulators
                    ms_step_acc = 0.0
                    steps_acc   = 0
                    util_ms_acc = 0.0
            
            # Cleanup
            del model, tokenizer, optimizer
            torch.cuda.empty_cache()

            self.gpu_monitor.shutdown()

            
        except torch.cuda.OutOfMemoryError:
            self.logger.error(f"CUDA OOM for {model_name} with batch_size={batch_size}")
            torch.cuda.empty_cache()
        except Exception as e:
            self.logger.error(f"Error benchmarking {model_name}: {e}")
            traceback.print_exc()
        
        return samples


# ==================== Main Benchmark Runner ====================

class WorkloadController:
    """Main benchmark orchestrator"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.gpu_monitor = GPUMonitor(config.gpu_index)
        self.power_monitor = ExternalPowerInterface(config)
        self.mfu_calculator = MFUCalculator(config.peak_tflops)
        self.model_benchmark = ModelTrainingBenchmark(config, self.gpu_monitor, self.mfu_calculator, self.power_monitor)
        self.logger = self._setup_logging()
        self.interrupted = False
        self.data_logger = DataLogger(config.output_file, config.append_mode, self.logger)
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('mfu_training_benchmark.log')
            ]
        )
        return logging.getLogger(__name__)

    def collect_baseline_samples(self) -> List[BenchmarkSample]:
        samples: List[BenchmarkSample] = []
        self.logger.info("Collecting %d baseline samples", self.config.baseline_samples)

        try:
            for i in range(self.config.baseline_samples):
                try:
                    gpu_stats = self.gpu_monitor.get_gpu_stats()
                except Exception as e:
                    self.logger.exception("Baseline GPU stats failed: %s", e)
                    gpu_stats = {
                        'gpu_utilization': 0, 'memory_utilization': 0, 'memory_used_mb': 0,
                        'temperature_celsius': 0, 'power_draw_watts': 0.0
                    }

                try:
                    power_stats = self.power_monitor.get_power_stats()
                except Exception as e:
                    self.logger.exception("Baseline power stats failed: %s", e)
                    power_stats = {
                        'power_meter_active_power_w': None,
                        'power_meter_reactive_power_var': None,
                        'power_meter_apparent_power_va': None,
                        'power_meter_timestamp': None,
                        'power_meter_available': False
                    }

                sample = BenchmarkSample(
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    model_name="baseline",
                    batch_size=0,
                    tokens_per_second=0.0,
                    mfu_percentage=0.0,
                    mfu_percentage_calflops=0.0,
                    sequence_length=0,
                    num_layers=0, num_heads=0, head_dim=0,
                    formula_flops_per_token=0.0,
                    calflops_total=0.0,
                    dtype=self.config.dtype,
                    learning_rate=self.config.learning_rate,
                    warmup_iterations=self.config.warmup_iterations,
                    cooldown_seconds=self.config.cooldown_seconds,
                    context_window=self.config.context_window,
                    loss=0.0,
                    **gpu_stats,
                    **power_stats
                )
                samples.append(sample)

                self.logger.info(
                    "Baseline sample %d/%d: GPU=%d%% | Mem=%dMB | Power=%.2fW | PM_W=%s",
                    i + 1, self.config.baseline_samples,
                    gpu_stats['gpu_utilization'],
                    gpu_stats['memory_used_mb'],
                    gpu_stats['power_draw_watts'],
                    power_stats.get('power_meter_active_power_w')
                )

                # sleep between samples (except after the last one)
                if i + 1 < self.config.baseline_samples:
                    time.sleep(self.config.sampling_interval)

            # save once, after the loop
            self.data_logger.save_samples(samples)
            self.logger.info("Baseline collection complete: %d samples saved.", len(samples))

        except Exception as e:
            self.logger.exception("Baseline collection crashed: %s", e)

        return samples

    def run_benchmark(self) -> None:
        """Run the complete training benchmark suite"""
        all_samples: List[BenchmarkSample] = []

        # collect baseline first
        output_path = Path(self.config.output_file)
        new_file = not (output_path.exists() and self.config.append_mode)
        if new_file:
            baseline = self.collect_baseline_samples()
            all_samples.extend(baseline)
            self.logger.info("After baseline: total_samples=%d", len(all_samples))

        for model_name in self.config.models:
            if self.interrupted:
                break
                
            for batch_size in self.config.batch_sizes:
                if self.interrupted:
                    break
                
                samples = self.model_benchmark.benchmark_model(model_name, batch_size)
                all_samples.extend(samples)

                self.data_logger.save_samples(samples)
                
                # Cooldown between combinations
                if self.config.cooldown_seconds > 0 and not self.interrupted:
                    self.logger.info(f"Cooling down for {self.config.cooldown_seconds}s...")
                    time.sleep(self.config.cooldown_seconds)
        
        self.logger.info(f"\nTraining benchmark completed! Total samples: {len(all_samples)}")
        self.logger.info(f"Results saved to: {self.config.output_file}")

# ==================== Main Entry Point ====================

@hydra.main(version_base=None, config_name="config")
def main(cfg: DictConfig):
    appcfg = cast(AppConfig, OmegaConf.to_object(cfg))   # -> real dataclass
    bcfg: BenchmarkConfig = appcfg.benchmark
    dev_name = torch.cuda.get_device_name(bcfg.gpu_index)
    bcfg.peak_tflops = pick_peak_tflops(dev_name, bcfg.dtype)
    runner = WorkloadController(bcfg)

    runner.logger.info(
        "PEAK TFLOPS CONFIG | gpu='%s' | dtype=%s | peak_tflops=%.2f TFLOPS",
        dev_name, bcfg.dtype, bcfg.peak_tflops
    )

    try:
        runner.run_benchmark()
    except KeyboardInterrupt:
        print("\nBenchmark cancelled by User!")
        raise


if __name__ == '__main__':
    main()
