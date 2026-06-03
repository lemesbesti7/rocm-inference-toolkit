#!/usr/bin/env python3
"""Basic inference example using ROCm Inference Toolkit."""

from inference import ROCmInferenceEngine
from config import InferenceConfig


def main():
    # Configure for ROCm GPU
    config = InferenceConfig(
        model_name="bert-base-uncased",
        device="rocm",
        precision="fp16",
        batch_size=16,
        max_length=256,
    )

    # Initialize engine
    engine = ROCmInferenceEngine(config)
    engine.load_model()

    # Single inference
    result = engine.infer("AMD ROCm enables high-performance AI inference on AMD GPUs.")
    print(f"Latency: {result['latency_ms']}ms | Shape: {result['output_shape']}")

    # Batch inference
    texts = [
        "PyTorch with ROCm runs natively on AMD hardware.",
        "Model quantization reduces memory usage by 4x.",
        "MI250X delivers excellent ML training performance.",
    ]
    batch_result = engine.infer(texts)
    print(f"Batch latency: {batch_result['latency_ms']}ms | Size: {batch_result['batch_size']}")


if __name__ == "__main__":
    main()
