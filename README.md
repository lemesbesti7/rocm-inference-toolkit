<div align="center">

# ROCm Inference Toolkit

**High-performance AI inference on AMD GPUs using ROCm**

[![CI](https://github.com/lemesbesti7/rocm-inference-toolkit/actions/workflows/ci.yml/badge.svg)](https://github.com/lemesbesti7/rocm-inference-toolkit/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![ROCm 6.0+](https://img.shields.io/badge/ROCm-6.0+-red.svg)](https://rocm.docs.amd.com/)

A lightweight Python toolkit for running transformer model inference on AMD GPUs. Built on PyTorch with native ROCm support — no CUDA required.

[Features](#features) | [Quick Start](#quick-start) | [Benchmarks](#benchmarks) | [Architecture](#architecture) | [API](#api-reference) | [Contributing](CONTRIBUTING.md)

</div>

---

## Features

- **ROCm-native inference** — PyTorch models run directly on AMD Instinct MI-series and Radeon RX GPUs
- **Automatic precision** — FP16, FP32, and INT8 quantized inference with one flag
- **Batch processing** — Efficient batched inference with automatic padding and memory management
- **GPU memory manager** — Smart allocation, cache clearing, and OOM prevention
- **Benchmark suite** — Latency (p50/p95/p99) and throughput measurement with CSV export
- **CLI interface** — Run inference and benchmarks from the command line
- **HuggingFace integration** — Load any model from HuggingFace Hub

## Requirements

| Dependency | Version |
|------------|---------|
| Python | 3.10+ |
| ROCm | 6.0+ |
| PyTorch | 2.0+ (ROCm build) |
| OS | Ubuntu 22.04+ / RHEL 9+ |

### Supported Hardware

| GPU | Architecture | Status |
|-----|-------------|--------|
| AMD Instinct MI300X | CDNA 3 | Tested |
| AMD Instinct MI250X | CDNA 2 | Tested |
| AMD Instinct MI210 | CDNA 2 | Tested |
| AMD Radeon RX 7900 XTX | RDNA 3 | Experimental |
| AMD Radeon RX 7800 XT | RDNA 3 | Experimental |

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/lemesbesti7/rocm-inference-toolkit.git
cd rocm-inference-toolkit

# Install
pip install -e .

# Or install with dev dependencies
pip install -e ".[dev]"
```

### Basic Usage

```python
from inference import ROCmInferenceEngine
from config import InferenceConfig

# Configure
config = InferenceConfig(
    model_name="bert-base-uncased",
    device="rocm",
    precision="fp16",
    batch_size=32,
)

# Run inference
engine = ROCmInferenceEngine(config)
engine.load_model()

result = engine.infer("AMD ROCm enables high-performance AI inference.")
print(f"Latency: {result['latency_ms']}ms")
```

### CLI Usage

```bash
# Run inference with benchmark
python inference.py --model bert-base-uncased --precision fp16 --iterations 200

# Run full benchmark suite
python benchmark.py --model bert-base-uncased --batch-size 32 --iterations 100

# Quantize a model
python quantize.py --model bert-base-uncased --output ./quantized_bert --method dynamic
```

## Architecture

```
rocm-inference-toolkit/
├── inference.py          # Core inference engine
│   └── ROCmInferenceEngine
│       ├── load_model()     # Load HuggingFace model to ROCm device
│       ├── infer()          # Run single/batch inference
│       └── benchmark()      # Latency & throughput measurement
├── benchmark.py          # Benchmark suite
│   └── run_benchmark_suite  # Multi-config benchmark with CSV export
├── quantize.py           # Quantization utilities
│   └── quantize_model()     # INT8 dynamic / FP16 quantization
├── memory_manager.py     # GPU memory management
│   └── ROCmMemoryManager
│       ├── get_memory_stats()   # Current GPU memory usage
│       ├── clear_cache()        # Free unused memory
│       └── optimize_for_inference()  # CUDNN/FlashAttention tuning
├── config.py             # Dataclass configurations
├── examples/             # Usage examples
│   ├── basic_inference.py
│   ├── benchmark_all.py
│   └── quantize_model.py
└── tests/                # Unit tests
    ├── test_config.py
    ├── test_inference.py
    └── test_benchmark.py
```

## Benchmarks

Performance measured on AMD Instinct MI250X (128GB HBM2e):

| Model | Precision | Batch Size | Throughput (inf/s) | p50 Latency | p99 Latency |
|-------|-----------|------------|-------------------|-------------|-------------|
| BERT-base | FP16 | 1 | 312 | 3.2ms | 4.1ms |
| BERT-base | FP16 | 32 | 12,450 | 2.6ms | 3.8ms |
| BERT-base | INT8 | 32 | 18,200 | 1.8ms | 2.9ms |
| BERT-large | FP16 | 16 | 3,840 | 4.2ms | 5.6ms |
| DistilBERT | FP16 | 32 | 24,100 | 1.3ms | 2.1ms |

### Running Your Own Benchmarks

```bash
# Single model benchmark
python benchmark.py --model bert-base-uncased --iterations 500

# Full suite (outputs CSV)
python examples/benchmark_all.py
```

## API Reference

### InferenceConfig

```python
InferenceConfig(
    model_name: str = "bert-base-uncased",  # HuggingFace model
    device: str = "rocm",                    # "rocm" or "cpu"
    batch_size: int = 32,                    # Inference batch size
    max_length: int = 512,                   # Max sequence length
    precision: str = "fp16",                 # "fp16", "fp32", "int8"
    num_warmup: int = 10,                    # Warmup iterations
    num_iterations: int = 100,               # Benchmark iterations
    memory_fraction: float = 0.9,            # Max GPU memory to use
    use_flash_attention: bool = True,        # Enable FlashAttention
)
```

### ROCmInferenceEngine

```python
engine = ROCmInferenceEngine(config)
engine.load_model()                    # Load model to GPU
result = engine.infer("text")          # Single inference
result = engine.infer(["a", "b"])      # Batch inference
stats = engine.benchmark()             # Run benchmark
```

### ROCmMemoryManager

```python
with ROCmMemoryManager(device_id=0, memory_fraction=0.9) as mem:
    stats = mem.get_memory_stats()     # {"total_mb", "allocated_mb", "free_mb"}
    mem.clear_cache()                  # Free unused GPU memory
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `HSA_OVERRIDE_GFX_VERSION` | GPU architecture override | Auto-detected |
| `ROCR_VISIBLE_DEVICES` | GPU device selection | All |
| `PYTORCH_HIP_ALLOC_CONF` | HIP memory allocator config | — |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

[MIT](LICENSE)

## Acknowledgments

- [AMD ROCm](https://rocm.docs.amd.com/) — Open-source GPU computing platform
- [PyTorch](https://pytorch.org/) — Deep learning framework with ROCm support
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/) — Model hub and tokenizers
