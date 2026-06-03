# ROCm Inference Toolkit

A lightweight Python toolkit for running AI model inference on AMD GPUs using ROCm. Optimized for AMD Instinct MI-series accelerators.

## Features

- **ROCm-native inference** — runs PyTorch models on AMD GPUs without CUDA
- **Batch processing** — efficient batched inference with automatic memory management
- **Model quantization** — INT8/FP16 quantization for faster inference on ROCm
- **Benchmark suite** — built-in benchmarking for latency and throughput testing
- **Multi-GPU support** — data parallelism across multiple AMD GPUs

## Requirements

- AMD GPU with ROCm support (MI210, MI250, MI250X, MI300, or RX 7900 series)
- ROCm 6.0+
- Python 3.10+
- PyTorch 2.0+ (ROCm build)

## Quick Start

```bash
pip install -r requirements.txt

# Run inference on a sample model
python inference.py --model bert-base --batch-size 32 --device rocm

# Run benchmarks
python benchmark.py --model bert-base --iterations 1000
```

## Project Structure

```
rocm-inference-toolkit/
├── inference.py          # Core inference engine
├── benchmark.py          # Performance benchmarking
├── quantize.py           # Model quantization utilities
├── memory_manager.py     # GPU memory management
├── config.py             # Configuration
├── requirements.txt
└── tests/
    ├── test_inference.py
    └── test_benchmark.py
```

## Performance (AMD MI250X)

| Model | Precision | Batch Size | Throughput (tokens/s) | Latency (ms) |
|-------|-----------|------------|----------------------|---------------|
| BERT-base | FP16 | 32 | 12,450 | 2.6 |
| BERT-base | INT8 | 32 | 18,200 | 1.8 |
| LLaMA-7B | FP16 | 1 | 42.3 | 23.6 |

## License

MIT
