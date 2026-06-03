"""Configuration for ROCm Inference Toolkit.

Provides dataclass-based configuration for inference and benchmarking
on AMD ROCm GPUs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class InferenceConfig:
    """Configuration for the inference engine.

    Attributes:
        model_name: HuggingFace model identifier or local path.
        device: Target device — "rocm" for AMD GPU, "cpu" for fallback.
        batch_size: Number of inputs per inference call.
        max_length: Maximum token sequence length.
        precision: Numerical precision — "fp16", "fp32", or "int8".
        num_warmup: Warmup iterations before benchmarking.
        num_iterations: Total benchmark iterations.
        memory_fraction: Fraction of GPU memory to allocate (0.0–1.0).
        use_flash_attention: Enable FlashAttention if available.
        quantize: Apply INT8 dynamic quantization at load time.
        output_dir: Directory for saving outputs.
    """

    model_name: str = "bert-base-uncased"
    device: str = "rocm"
    batch_size: int = 32
    max_length: int = 512
    precision: str = "fp16"
    num_warmup: int = 10
    num_iterations: int = 100
    memory_fraction: float = 0.9
    use_flash_attention: bool = True
    quantize: bool = False
    output_dir: str = "./outputs"

    def __post_init__(self) -> None:
        """Set ROCm environment defaults if using GPU."""
        if self.device == "rocm":
            os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")

        valid_precisions = {"fp16", "fp32", "int8"}
        if self.precision not in valid_precisions:
            raise ValueError(
                f"Invalid precision '{self.precision}'. Must be one of {valid_precisions}"
            )

        if not 0.0 < self.memory_fraction <= 1.0:
            raise ValueError("memory_fraction must be between 0.0 and 1.0")


@dataclass
class BenchmarkConfig:
    """Configuration for the benchmark suite.

    Attributes:
        models: List of HuggingFace model identifiers to benchmark.
        batch_sizes: Batch sizes to test.
        precisions: Precision modes to test.
        warmup_iterations: Warmup iterations per configuration.
        test_iterations: Measurement iterations per configuration.
        sequence_length: Fixed sequence length for all tests.
        output_csv: Path for CSV results output.
    """

    models: list[str] = field(default_factory=lambda: ["bert-base-uncased"])
    batch_sizes: list[int] = field(default_factory=lambda: [1, 8, 16, 32])
    precisions: list[str] = field(default_factory=lambda: ["fp16", "int8"])
    warmup_iterations: int = 20
    test_iterations: int = 200
    sequence_length: int = 128
    output_csv: str = "benchmark_results.csv"
