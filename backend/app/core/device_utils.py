"""Utility for detecting available PyTorch devices at runtime."""
from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def get_available_devices() -> list[str]:
    """Return the list of available PyTorch devices.

    Always includes "cpu". Adds "cuda" if torch.cuda.is_available(),
    "mps" if torch.backends.mps.is_available(). If torch is not installed,
    only "cpu" is returned.
    """
    devices = ["cpu"]
    try:
        import torch

        if torch.cuda.is_available():
            devices.append("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            devices.append("mps")
    except ImportError:
        pass
    return devices
