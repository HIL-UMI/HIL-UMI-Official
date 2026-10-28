# Copyright 2025 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
# Modified for HIL-UMI.

from .accelerator import (
    Accelerator,
    AcceleratorManager,
    AcceleratorType,
    AcceleratorUtil,
    ProfileConfig,
)
from .amd_gpu import AMDGPUManager, RocprofSysConfig  # Must be imported to register
from .ascend_npu import AscendNPUManager
from .intel_gpu import IntelGPUManager
from .kunlun_xpu import KunlunXPUManager
from .musa_gpu import MUSAGPUManager
from .nvidia_gpu import NsightConfig, NvidiaGPUManager

__all__ = [
    "AcceleratorManager",
    "AcceleratorUtil",
    "Accelerator",
    "AcceleratorType",
    "ProfileConfig",
    "AMDGPUManager",
    "RocprofSysConfig",
    "AscendNPUManager",
    "IntelGPUManager",
    "NvidiaGPUManager",
    "NsightConfig",
    "MUSAGPUManager",
    "KunlunXPUManager",
]
