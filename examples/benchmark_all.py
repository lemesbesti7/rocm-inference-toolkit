#!/usr/bin/env python3
"""Run benchmarks across multiple models and configurations."""

from config import BenchmarkConfig
from benchmark import run_benchmark_suite


def main():
    config = BenchmarkConfig(
        models=[
            "bert-base-uncased",
            "bert-large-uncased",
            "distilbert-base-uncased",
        ],
        batch_sizes=[1, 8, 16, 32],
        precisions=["fp16", "int8"],
        warmup_iterations=20,
        test_iterations=100,
        sequence_length=128,
        output_csv="benchmark_results.csv",
    )

    results = run_benchmark_suite(config)

    # Print summary
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    for r in results:
        if "error" not in r:
            print(
                f"  {r['model']:30} | {r['precision']:5} | "
                f"batch={r['batch_size']:3} | "
                f"p50={r['p50_latency_ms']:8.2f}ms | "
                f"throughput={r['throughput_inferences_per_sec']:8.1f} inf/s"
            )


if __name__ == "__main__":
    main()
