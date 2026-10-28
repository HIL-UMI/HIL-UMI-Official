# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
# Modified for HIL-UMI.

import torch
import torch.nn.functional as F


@torch.compile
def gelu_glu(gate_input: torch.Tensor, value_input: torch.Tensor) -> torch.Tensor:
    """Fused GELU-GLU activation: ``gelu(gate_input) * value_input``."""
    return F.gelu(gate_input) * value_input


def _str_to_dtype(dtype_str: str) -> torch.dtype:
    """Convert string dtype to torch dtype."""
    # if dtype_str == "mp_bfloat16":
    #     assert False
    mapping = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "mp_bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    return mapping[dtype_str]
