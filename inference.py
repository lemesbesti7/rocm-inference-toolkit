"""Core inference engine for ROCm devices."""

import time
from typing import Optional, Union

import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM

from config import InferenceConfig
from memory_manager import ROCmMemoryManager


class ROCmInferenceEngine:
    """High-performance inference engine optimized for AMD ROCm GPUs."""

    def __init__(self, config: Optional[InferenceConfig] = None):
        self.config = config or InferenceConfig()
        self.device = torch.device(self.config.device if torch.cuda.is_available() else "cpu")
        self.memory_manager = ROCmMemoryManager(
            memory_fraction=self.config.memory_fraction
        )
        self.model = None
        self.tokenizer = None

    def load_model(self, model_name: Optional[str] = None):
        """Load a HuggingFace model onto the ROCm device."""
        name = model_name or self.config.model_name
        print(f"Loading model: {name}")

        self.tokenizer = AutoTokenizer.from_pretrained(name)

        dtype = torch.float16 if self.config.precision == "fp16" else torch.float32

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                name, torch_dtype=dtype, device_map="auto"
            )
        except Exception:
            self.model = AutoModel.from_pretrained(
                name, torch_dtype=dtype, device_map="auto"
            )

        self.model.eval()

        if self.config.precision == "int8":
            self.model = torch.quantization.quantize_dynamic(
                self.model, {torch.nn.Linear}, dtype=torch.qint8
            )

        stats = self.memory_manager.get_memory_stats()
        print(f"Model loaded. GPU memory: {stats.get('allocated_mb', 0):.0f} MB allocated")
        return self

    def infer(self, inputs: Union[str, list[str]], **kwargs) -> dict:
        """Run inference on input text(s)."""
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

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.config.precision == "fp16"):
            outputs = self.model(**encoded)

        latency = (time.perf_counter() - start) * 1000

        return {
            "latency_ms": round(latency, 2),
            "output_shape": outputs.last_hidden_state.shape,
            "batch_size": len(inputs),
            "device": str(self.device),
        }

    def benchmark(self, sample_text: str = "This is a benchmark test.") -> dict:
        """Run warmup + benchmark iterations."""
        latencies = []

        # Warmup
        for _ in range(self.config.num_warmup):
            self.infer(sample_text)

        # Benchmark
        for _ in range(self.config.num_iterations):
            result = self.infer(sample_text)
            latencies.append(result["latency_ms"])

        return {
            "mean_latency_ms": round(np.mean(latencies), 2),
            "p50_latency_ms": round(np.percentile(latencies, 50), 2),
            "p95_latency_ms": round(np.percentile(latencies, 95), 2),
            "p99_latency_ms": round(np.percentile(latencies, 99), 2),
            "throughput_inferences_per_sec": round(1000 / np.mean(latencies), 2),
            "iterations": self.config.num_iterations,
            "device": str(self.device),
        }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="ROCm Inference Toolkit")
    parser.add_argument("--model", default="bert-base-uncased", help="Model name")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="rocm", choices=["rocm", "cpu"])
    parser.add_argument("--precision", default="fp16", choices=["fp16", "fp32", "int8"])
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--max-length", type=int, default=512)
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
    print(f"\nBenchmark Results:")
    for k, v in results.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
