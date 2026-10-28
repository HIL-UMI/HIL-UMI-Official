# Copyright 2026 HIL-UMI contributors.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the saved-training-state to deployment-tensor boundary on CPU."""

import json
import subprocess
import sys

import pytest
import torch
from safetensors.torch import load_file

from training.utils.ckpt_convertor.openpi._core import strip_wrapper_prefix


def test_saved_sft_tensors_convert_without_namespace_or_stats_loss(tmp_path):
    checkpoint = tmp_path / "global_step_1"
    state_path = checkpoint / "actor/model_state_dict/full_weights.pt"
    state_path.parent.mkdir(parents=True)
    weights = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    indices = torch.arange(3, dtype=torch.int64)
    torch.save(
        {
            "state_dict": {
                "module._fsdp_wrapped_module.model.action_in_proj.weight": weights,
                "_orig_mod.model.position_ids": indices,
            }
        },
        state_path,
    )
    stats = tmp_path / "norm_stats.json"
    stats.write_text('{"norm_stats": {"state": {"mean": [0.0]}}}\n')
    output = tmp_path / "deploy"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "training.utils.ckpt_convertor.openpi.convert",
            "--mode",
            "sft2new",
            "--ckpt",
            str(checkpoint),
            "--input-norm-stats",
            str(stats),
            "--output-model",
            str(output),
            "--output-norm-stats",
            str(output / "norm_stats/norm_stats.json"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    actual = load_file(str(output / "model.safetensors"))
    assert set(actual) == {"action_in_proj.weight", "position_ids"}
    assert torch.equal(actual["action_in_proj.weight"], weights.bfloat16())
    assert torch.equal(actual["position_ids"], indices)
    assert (output / "norm_stats/norm_stats.json").read_bytes() == stats.read_bytes()
    assert json.loads((output / "config.json").read_text())["pi05"] is True


def test_duplicate_checkpoint_keys_fail_before_a_tensor_can_be_dropped():
    with pytest.raises(ValueError, match="duplicate bare key"):
        strip_wrapper_prefix({"model.weight": torch.ones(1), "weight": torch.zeros(1)})
