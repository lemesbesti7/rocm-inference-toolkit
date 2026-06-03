# Contributing to ROCm Inference Toolkit

Contributions are welcome! This project aims to make AMD GPU inference accessible to everyone.

## Getting Started

1. Fork the repository
2. Clone your fork
3. Create a feature branch: `git checkout -b feature/my-feature`
4. Install dev dependencies: `pip install -e ".[dev]"`

## Development Setup

```bash
git clone https://github.com/lemesbesti7/rocm-inference-toolkit.git
cd rocm-inference-toolkit
pip install -e ".[dev]"
```

## Code Quality

We use `ruff` for linting and formatting:

```bash
ruff check .          # Lint
ruff format .         # Format
ruff check --fix .    # Auto-fix
```

## Testing

```bash
pytest                          # Run all tests
pytest --cov=. --cov-report=html  # With coverage
pytest tests/test_inference.py    # Single file
```

## Pull Request Process

1. Ensure all tests pass
2. Add tests for new functionality
3. Update README if needed
4. Keep PRs focused (one feature per PR)

## Areas for Contribution

- ROCm-specific optimizations (hipBLAS, MIOpen)
- Additional model architectures
- Multi-GPU data parallelism
- Documentation improvements
- Benchmark comparisons across AMD GPU generations
