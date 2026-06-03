"""Tests for inference engine."""

import unittest
from config import InferenceConfig


class TestInferenceConfig(unittest.TestCase):
    def test_default_config(self):
        config = InferenceConfig()
        self.assertEqual(config.device, "rocm")
        self.assertEqual(config.batch_size, 32)
        self.assertEqual(config.precision, "fp16")
        self.assertTrue(config.use_flash_attention)

    def test_custom_config(self):
        config = InferenceConfig(model_name="gpt2", batch_size=16, precision="fp32")
        self.assertEqual(config.model_name, "gpt2")
        self.assertEqual(config.batch_size, 16)
        self.assertEqual(config.precision, "fp32")

    def test_memory_fraction(self):
        config = InferenceConfig(memory_fraction=0.8)
        self.assertAlmostEqual(config.memory_fraction, 0.8)


class TestImports(unittest.TestCase):
    def test_import_inference(self):
        from inference import ROCmInferenceEngine
        self.assertIsNotNone(ROCmInferenceEngine)

    def test_import_memory_manager(self):
        from memory_manager import ROCmMemoryManager
        self.assertIsNotNone(ROCmMemoryManager)

    def test_import_config(self):
        from config import InferenceConfig, BenchmarkConfig
        self.assertIsNotNone(InferenceConfig)
        self.assertIsNotNone(BenchmarkConfig)


if __name__ == "__main__":
    unittest.main()
