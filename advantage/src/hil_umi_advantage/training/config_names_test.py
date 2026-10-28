"""Names remain compatible without modifying a serialized checkpoint."""

from hil_umi_advantage.training.config_names import canonicalize_config_name


def test_historical_names_resolve_to_public_names():
    assert canonicalize_config_name("ADVANTAGE_TORCH_KAI0_TOWEL_FOLD") == "ADVANTAGE_TORCH_TOWEL_FOLD"
    assert canonicalize_config_name("ADVANTAGE_TORCH_KAI0_UMI") == "ADVANTAGE_TORCH_UMI"


def test_canonical_names_and_unknown_names_are_preserved():
    for name in ("ADVANTAGE_TORCH_TOWEL_FOLD", "ADVANTAGE_TORCH_UMI", "unknown"):
        assert canonicalize_config_name(name) == name
