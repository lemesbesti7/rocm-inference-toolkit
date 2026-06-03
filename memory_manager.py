"""GPU memory management for ROCm devices."""

import gc
from typing import Optional
import torch


class ROCmMemoryManager:
    """Manages GPU memory allocation and cleanup for ROCm devices."""

    def __init__(self, device_id: int = 0, memory_fraction: float = 0.9):
        self.device_id = device_id
        self.memory_fraction = memory_fraction
        self.device = torch.device(f"cuda:{device_id}")
        self._setup_memory()

    def _setup_memory(self):
        """Configure ROCm memory allocation."""
        if torch.cuda.is_available():
            torch.cuda.set_device(self.device_id)
            total_mem = torch.cuda.get_device_properties(self.device_id).total_mem
            self.reserved_mem = int(total_mem * self.memory_fraction)
            torch.cuda.set_per_process_memory_fraction(
                self.memory_fraction, self.device_id
            )

    def get_memory_stats(self) -> dict:
        """Get current GPU memory statistics."""
        if not torch.cuda.is_available():
            return {"error": "No GPU available"}

        return {
            "total_mb": torch.cuda.get_device_properties(self.device_id).total_mem / 1e6,
            "allocated_mb": torch.cuda.memory_allocated(self.device_id) / 1e6,
            "cached_mb": torch.cuda.memory_reserved(self.device_id) / 1e6,
            "free_mb": (
                torch.cuda.get_device_properties(self.device_id).total_mem
                - torch.cuda.memory_allocated(self.device_id)
            ) / 1e6,
        }

    def clear_cache(self):
        """Free unused GPU memory."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize(self.device)

    def optimize_for_inference(self):
        """Apply inference-time memory optimizations."""
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(True)

    def __enter__(self):
        self.optimize_for_inference()
        return self

    def __exit__(self, *args):
        self.clear_cache()
