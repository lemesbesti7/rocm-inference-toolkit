"""Tests for configuration module."""

import unittest
from config import InferenceConfig, BenchmarkConfig


class TestInferenceConfig(unittest.TestCase):
    def test_defaults(self):
        config = InferenceConfig()
        self.assertEqual(config.model_name, "bert-base-uncased")
        self.assertEqual(config.device, "rocm")
        self.assertEqual(config.batch_size, 32)
        self.assertEqual(config.precision, "fp16")
        self.assertEqual(config.max_length, 512)
        self.assertEqual(config.num_warmup, 10)
        self.assertEqual(config.num_iterations, 100)
        self.assertAlmostEqual(config.memory_fraction, 0.9)

    def test_custom_values(self):
        config = InferenceConfig(
            model_name="gpt2",
            device="cpu",
            batch_size=16,
            precision="fp32",
            max_length=256,
        )
        self.assertEqual(config.model_name, "gpt2")
        self.assertEqual(config.device, "cpu")
        self.assertEqual(config.batch_size, 16)
        self.assertEqual(config.precision, "fp32")
        self.assertEqual(config.max_length, 256)


class TestBenchmarkConfig(unittest.TestCase):
    def test_defaults(self):
        config = BenchmarkConfig()
        self.assertIn("bert-base-uncased", config.models)
        self.assertEqual(config.batch_sizes, [1, 8, 16, 32])
        self.assertEqual(config.precisions, ["fp16", "int8"])
        self.assertEqual(config.warmup_iterations, 20)
        self.assertEqual(config.test_iterations, 200)
        self.assertEqual(config.sequence_length, 128)

    def test_custom_values(self):
        config = BenchmarkConfig(
            models=["gpt2", "distilbert-base-uncased"],
            batch_sizes=[1, 4],
            precisions=["fp16"],
        )
        self.assertEqual(len(config.models), 2)
        self.assertEqual(len(config.batch_sizes), 2)
        self.assertEqual(config.precisions, ["fp16"])


if __name__ == "__main__":
    unittest.main()
