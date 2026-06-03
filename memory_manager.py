"""GPU memory management for ROCm devices.

Provides context-managed GPU memory allocation, monitoring, and cleanup
optimized for AMD ROCm/hipBLAS workloads.
"""

from __future__ import annotations

import gc
import logging
from typing import Any

logger = logging.getLogger(__name__)


class ROCmMemoryManager:
    """Manages GPU memory allocation and cleanup for ROCm devices.

    Usage:
        with ROCmMemoryManager(device_id=0) as mem:
            stats = mem.get_memory_stats()
            # ... run inference ...
            mem.clear_cache()
    """

    def __init__(self, device_id: int = 0, memory_fraction: float = 0.9) -> None:
        """Initialize memory manager.

        Args:
            device_id: ROCm device index.
            memory_fraction: Maximum fraction of GPU memory to use.
        """
        import torch

        self.device_id = device_id
        self.memory_fraction = memory_fraction
        self.device = torch.device(f"cuda:{device_id}")
        self._torch = torch
        self._setup_memory()

    def _setup_memory(self) -> None:
        """Configure ROCm memory allocation limits."""
        torch = self._torch
        if torch.cuda.is_available():
            torch.cuda.set_device(self.device_id)
            props = torch.cuda.get_device_properties(self.device_id)
            total_mb = props.total_mem / 1e6
            torch.cuda.set_per_process_memory_fraction(
                self.memory_fraction, self.device_id
            )
            logger.info(
                "GPU %d: %s (%.0f MB), limit=%.0f%%",
                self.device_id,
                props.name,
                total_mb,
                self.memory_fraction * 100,
            )
        else:
            logger.warning("No CUDA/ROCm GPU available, memory manager in passive mode")

    def get_memory_stats(self) -> dict[str, Any]:
        """Get current GPU memory statistics.

        Returns:
            Dict with keys: total_mb, allocated_mb, cached_mb, free_mb.
        """
        torch = self._torch
        if not torch.cuda.is_available():
            return {"error": "No GPU available", "total_mb": 0, "allocated_mb": 0,
                    "cached_mb": 0, "free_mb": 0}

        props = torch.cuda.get_device_properties(self.device_id)
        allocated = torch.cuda.memory_allocated(self.device_id)
        cached = torch.cuda.memory_reserved(self.device_id)

        return {
            "total_mb": round(props.total_mem / 1e6, 1),
            "allocated_mb": round(allocated / 1e6, 1),
            "cached_mb": round(cached / 1e6, 1),
            "free_mb": round((props.total_mem - allocated) / 1e6, 1),
        }

    def clear_cache(self) -> None:
        """Free unused GPU memory — clears PyTorch cache and runs GC."""
        gc.collect()
        torch = self._torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize(self.device)
            logger.debug("GPU %d cache cleared", self.device_id)

    def optimize_for_inference(self) -> None:
        """Apply inference-time performance optimizations."""
        torch = self._torch
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(True)
            logger.debug("FlashAttention SDP enabled")

    def __enter__(self) -> ROCmMemoryManager:
        self.optimize_for_inference()
        return self

    def __exit__(self, *args: Any) -> None:
        self.clear_cache()
