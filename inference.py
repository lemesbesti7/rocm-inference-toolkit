"""Core inference engine for ROCm devices.

Provides high-performance transformer model inference optimized for
AMD ROCm GPUs with FP16/INT8 precision and batched processing.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Union

import numpy as np
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from config import InferenceConfig
from memory_manager import ROCmMemoryManager

logger = logging.getLogger(__name__)


class ROCmInferenceEngine:
    """High-performance inference engine optimized for AMD ROCm GPUs.

    Supports any HuggingFace transformer model with automatic device placement,
    mixed precision, and INT8 quantization.

    Example:
        config = InferenceConfig(model_name="bert-base-uncased", precision="fp16")
        engine = ROCmInferenceEngine(config)
        engine.load_model()
        result = engine.infer("Hello world")
    """

    def __init__(self, config: InferenceConfig | None = None) -> None:
        """Initialize the inference engine.

        Args:
            config: Inference configuration. Uses defaults if None.
        """
        self.config = config or InferenceConfig()
        self.device = torch.device(
            self.config.device if torch.cuda.is_available() else "cpu"
        )
        self.memory_manager = ROCmMemoryManager(
            memory_fraction=self.config.memory_fraction
        )
        self.model: torch.nn.Module | None = None
        self.tokenizer: Any = None

    def load_model(self, model_name: str | None = None) -> ROCmInferenceEngine:
        """Load a HuggingFace model onto the ROCm device.

        Args:
            model_name: Model identifier or path. Uses config default if None.

        Returns:
            Self for method chaining.
        """
        name = model_name or self.config.model_name
        logger.info("Loading model: %s (precision=%s)", name, self.config.precision)

        self.tokenizer = AutoTokenizer.from_pretrained(name)
        dtype = torch.float16 if self.config.precision == "fp16" else torch.float32

        # Try causal LM first, fall back to base model
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                name, torch_dtype=dtype, device_map="auto"
            )
        except (ValueError, OSError):
            self.model = AutoModel.from_pretrained(
                name, torch_dtype=dtype, device_map="auto"
            )

        self.model.eval()

        # Apply INT8 dynamic quantization
        if self.config.precision == "int8":
            self.model = torch.quantization.quantize_dynamic(
                self.model, {torch.nn.Linear}, dtype=torch.qint8
            )
            logger.info("Applied INT8 dynamic quantization")

        stats = self.memory_manager.get_memory_stats()
        logger.info(
            "Model loaded on %s | GPU memory: %.0f MB allocated",
            self.device,
            stats.get("allocated_mb", 0),
        )
        return self

    def infer(self, inputs: Union[str, list[str]], **kwargs: Any) -> dict[str, Any]:
        """Run inference on input text(s).

        Args:
            inputs: Single string or list of strings.
            **kwargs: Additional tokenizer arguments.

        Returns:
            Dict with latency_ms, output_shape, batch_size, device.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if isinstance(inputs, str):
            inputs = [inputs]

        start = time.perf_counter()
        encoded = self.tokenizer(
            inputs,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad(), torch.cuda.amp.autocast(
            enabled=self.config.precision == "fp16"
        ):
            outputs = self.model(**encoded)

        latency_ms = (time.perf_counter() - start) * 1000

        return {
            "latency_ms": round(latency_ms, 2),
            "output_shape": tuple(outputs.last_hidden_state.shape),
            "batch_size": len(inputs),
            "device": str(self.device),
        }

    def benchmark(self, sample_text: str = "This is a benchmark test.") -> dict[str, Any]:
        """Run warmup + benchmark iterations and return statistics.

        Args:
            sample_text: Text to use for benchmarking.

        Returns:
            Dict with mean/p50/p95/p99 latency, throughput, and metadata.
        """
        latencies: list[float] = []

        # Warmup
        logger.info("Warming up (%d iterations)...", self.config.num_warmup)
        for _ in range(self.config.num_warmup):
            self.infer(sample_text)

        # Benchmark
        logger.info("Benchmarking (%d iterations)...", self.config.num_iterations)
        for _ in range(self.config.num_iterations):
            result = self.infer(sample_text)
            latencies.append(result["latency_ms"])

        arr = np.array(latencies)
        return {
            "mean_latency_ms": round(float(np.mean(arr)), 2),
            "p50_latency_ms": round(float(np.percentile(arr, 50)), 2),
            "p95_latency_ms": round(float(np.percentile(arr, 95)), 2),
            "p99_latency_ms": round(float(np.percentile(arr, 99)), 2),
            "throughput_inferences_per_sec": round(1000 / float(np.mean(arr)), 2),
            "iterations": self.config.num_iterations,
            "device": str(self.device),
        }


def main() -> None:
    """CLI entry point for inference and benchmarking."""
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="ROCm Inference Toolkit — run transformer inference on AMD GPUs"
    )
    parser.add_argument("--model", default="bert-base-uncased", help="HuggingFace model name")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--device", default="rocm", choices=["rocm", "cpu"], help="Device")
    parser.add_argument("--precision", default="fp16", choices=["fp16", "fp32", "int8"])
    parser.add_argument("--iterations", type=int, default=100, help="Benchmark iterations")
    parser.add_argument("--max-length", type=int, default=512, help="Max sequence length")
    args = parser.parse_args()

    config = InferenceConfig(
        model_name=args.model,
        device=args.device,
        batch_size=args.batch_size,
        precision=args.precision,
        num_iterations=args.iterations,
        max_length=args.max_length,
    )

    engine = ROCmInferenceEngine(config)
    engine.load_model()

    results = engine.benchmark()
    print("\nBenchmark Results:")
    for key, value in results.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
