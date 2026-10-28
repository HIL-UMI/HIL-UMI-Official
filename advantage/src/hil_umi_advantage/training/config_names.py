"""Compatibility names for existing Advantage checkpoints.

Only names are translated. Checkpoint contents and training behavior are unchanged.
"""

_LEGACY_CONFIG_NAMES = {
    "ADVANTAGE_TORCH_KAI0_TOWEL_FOLD": "ADVANTAGE_TORCH_TOWEL_FOLD",
    "ADVANTAGE_TORCH_KAI0_UMI": "ADVANTAGE_TORCH_UMI",
}


def canonicalize_config_name(name: str) -> str:
    """Return the public name while accepting historical checkpoint names."""
    return _LEGACY_CONFIG_NAMES.get(name, name)
