"""Configuration smoke tests for the complete Advantage source environment."""

import dataclasses
import sys

import pytest

from hil_umi_advantage.training import config


@pytest.mark.parametrize(
    ("old_name", "public_name"),
    [
        ("ADVANTAGE_TORCH_KAI0_TOWEL_FOLD", "ADVANTAGE_TORCH_TOWEL_FOLD"),
        ("ADVANTAGE_TORCH_KAI0_UMI", "ADVANTAGE_TORCH_UMI"),
    ],
)
def test_old_config_names_resolve_to_the_same_config(old_name, public_name):
    assert config.get_config(old_name) is config.get_config(public_name)


def test_generic_umi_config_preserves_training_defaults():
    towel = config.get_config("ADVANTAGE_TORCH_TOWEL_FOLD")
    generic = config.get_config("ADVANTAGE_TORCH_UMI")
    assert generic == dataclasses.replace(
        towel, name="ADVANTAGE_TORCH_UMI", data=dataclasses.replace(towel.data, default_prompt=None)
    )


def test_cli_accepts_an_existing_config_name(monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["train_pytorch.py", "ADVANTAGE_TORCH_KAI0_TOWEL_FOLD", "--exp-name", "compatibility_test"]
    )
    result = config.cli()
    assert result.name == "ADVANTAGE_TORCH_TOWEL_FOLD"
    assert result.exp_name == "compatibility_test"


def test_unknown_config_is_rejected():
    with pytest.raises(ValueError, match="not found"):
        config.get_config("missing_config")
