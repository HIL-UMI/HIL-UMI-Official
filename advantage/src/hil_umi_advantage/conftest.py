# Modified for HIL-UMI.
"""Keep ordinary unit tests independent of GPU/NVML availability."""

import os

# GPU integration runs can explicitly select JAX_PLATFORMS before pytest starts.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
