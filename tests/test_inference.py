"""Tests for inference engine."""

import unittest
from config import InferenceConfig


class TestInferenceConfig(unittest.TestCase):
    def test_default_config(self):
        config = InferenceConfig()
        self.assertEqual(config.device, "rocm")
        self.assertEqual(config.batch_size, 32)
        self.assertEqual(config.precision, "fp16")

    def test_custom_config(self):
        config = InferenceConfig(model_name="gpt2", batch_size=16, precision="fp32")
        self.assertEqual(config.model_name, "gpt2")
        self.assertEqual(config.batch_size, 16)
        self.assertEqual(config.precision, "fp32")


class TestMemoryManager(unittest.TestCase):
    def test_import(self):
        from memory_manager import ROCmMemoryManager
        self.assertIsNotNone(ROCmMemoryManager)


if __name__ == "__main__":
    unittest.main()
