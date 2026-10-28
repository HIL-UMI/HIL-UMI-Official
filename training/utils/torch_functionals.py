# Copyright 2025 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
# Modified for HIL-UMI.

import torch.nn.functional as F


def pad_tensor_to_length(tensors, max_seq_len, pad_token_id, left_pad=False):
    """
    pad a 2D tensors (e.g. responses, logprobs) in the last dim to max_seq_length.
    input shape: [bs, seq_length]
    output shape: [bs, max_seq_length]
    (0, max_seq_len - tensors.shape[-1]) means right pad to max_seq_length and no left pad
    """
    if tensors.shape[-1] >= max_seq_len:
        return tensors
    pad_tuple = (
        (max_seq_len - tensors.shape[-1], 0)
        if left_pad
        else (0, max_seq_len - tensors.shape[-1])
    )
    return F.pad(tensors, pad_tuple, "constant", pad_token_id)
