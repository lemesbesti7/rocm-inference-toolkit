"""Configuration for ROCm Inference Toolkit."""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class InferenceConfig:
    """Inference configuration."""
    model_name: str = "bert-base-uncased"
    device: str = "rocm"
    batch_size: int = 32
    max_length: int = 512
    precision: str = "fp16"  # fp16, int8, fp32
    num_warmup: int = 10
    num_iterations: int = 100
    memory_fraction: float = 0.9
    use_flash_attention: bool = True
    quantize: bool = False
    output_dir: str = "./outputs"

    def __post_init__(self):
        if self.device == "rocm":
            os.environ["HSA_OVERRIDE_GFX_VERSION"] = os.environ.get(
                "HSA_OVERRIDE_GFX_VERSION", "10.3.0"
            )


@dataclass
class BenchmarkConfig:
    """Benchmark configuration."""
    models: list = field(default_factory=lambda: ["bert-base-uncased"])
    batch_sizes: list = field(default_factory=lambda: [1, 8, 16, 32])
    precisions: list = field(default_factory=lambda: ["fp16", "int8"])
    warmup_iterations: int = 20
    test_iterations: int = 200
    sequence_length: int = 128
    output_csv: str = "benchmark_results.csv"
