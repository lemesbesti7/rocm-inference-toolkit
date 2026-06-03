"""Benchmark suite for ROCm inference performance."""

import csv
import time
from itertools import product

from config import BenchmarkConfig, InferenceConfig
from inference import ROCmInferenceEngine


def run_benchmark_suite(config: BenchmarkConfig):
    """Run full benchmark across models, batch sizes, and precisions."""
    results = []
    total = len(config.models) * len(config.batch_sizes) * len(config.precisions)
    current = 0

    for model_name, batch_size, precision in product(
        config.models, config.batch_sizes, config.precisions
    ):
        current += 1
        print(f"[{current}/{total}] {model_name} | batch={batch_size} | {precision}")

        inf_config = InferenceConfig(
            model_name=model_name,
            batch_size=batch_size,
            precision=precision,
            num_warmup=config.warmup_iterations,
            num_iterations=config.test_iterations,
            max_length=config.sequence_length,
        )

        try:
            engine = ROCmInferenceEngine(inf_config)
            engine.load_model()
            result = engine.benchmark()
            result.update({
                "model": model_name,
                "batch_size": batch_size,
                "precision": precision,
                "sequence_length": config.sequence_length,
            })
            results.append(result)
            engine.memory_manager.clear_cache()
        except Exception as e:
            print(f"  Error: {e}")
            results.append({
                "model": model_name,
                "batch_size": batch_size,
                "precision": precision,
                "error": str(e),
            })

    if config.output_csv and results:
        keys = results[0].keys()
        with open(config.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults saved to {config.output_csv}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="bert-base-uncased")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    config = BenchmarkConfig(
        models=[args.model],
        test_iterations=args.iterations,
    )
    run_benchmark_suite(config)
