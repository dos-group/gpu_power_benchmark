import logging
import time
import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from hydra.core.config_store import ConfigStore
from typing import cast

from benchmark.core import (
    AppConfig, BenchmarkConfig, MFUCalculator, 
    ModelTrainingBenchmark, DataLogger, pick_peak_tflops
)
from benchmark.monitors import GPUMonitor, ExternalPowerInterface

cs = ConfigStore.instance()
cs.store(name="config", node=AppConfig())


class WorkloadController:
    """Main benchmark orchestrator"""
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.gpu_monitor = GPUMonitor(config.gpu_index)
        self.power_monitor = ExternalPowerInterface(
            enabled=config.power_measurement_enabled,
            url=config.power_meter_url,
            timeout=config.power_meter_timeout
        )
        self.mfu_calculator = MFUCalculator(config.peak_tflops)
        self.model_benchmark = ModelTrainingBenchmark(
            config, self.gpu_monitor, self.mfu_calculator, self.power_monitor
        )
        self.data_logger = DataLogger(config.output_file, config.append_mode)
        self.logger = logging.getLogger(__name__)

    def run_benchmark(self) -> None:
        """Run the complete training benchmark suite"""
        for model_name in self.config.models:
            for batch_size in self.config.batch_sizes:
                self.logger.info(f"Starting: {model_name} | BS={batch_size}")
                samples = self.model_benchmark.benchmark_model(model_name, batch_size)
                self.data_logger.log(samples)
                
                if self.config.cooldown_seconds > 0:
                    self.logger.info(f"Cooldown: {self.config.cooldown_seconds}s")
                    time.sleep(self.config.cooldown_seconds)
        
        self.gpu_monitor.shutdown()


@hydra.main(version_base=None, config_name="config")
def main(cfg: DictConfig):
    appcfg = cast(AppConfig, OmegaConf.to_object(cfg))
    bcfg: BenchmarkConfig = appcfg.benchmark
    
    # Auto-detect peak TFLOPS
    dev_name = torch.cuda.get_device_name(bcfg.gpu_index)
    bcfg.peak_tflops = pick_peak_tflops(dev_name, bcfg.dtype)
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info(f"GPU: {dev_name} | Peak TFLOPS: {bcfg.peak_tflops}")
    
    runner = WorkloadController(bcfg)
    runner.run_benchmark()


if __name__ == '__main__':
    main()
