"""Tests for benchmark suite."""

import unittest
from config import BenchmarkConfig


class TestBenchmarkConfig(unittest.TestCase):
    def test_default_config(self):
        config = BenchmarkConfig()
        self.assertIn("bert-base-uncased", config.models)
        self.assertEqual(config.test_iterations, 200)

    def test_custom_config(self):
        config = BenchmarkConfig(models=["gpt2"], batch_sizes=[1, 4])
        self.assertEqual(config.models, ["gpt2"])
        self.assertEqual(len(config.batch_sizes), 2)


if __name__ == "__main__":
    unittest.main()
