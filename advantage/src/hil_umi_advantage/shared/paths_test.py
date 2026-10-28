# Modified for HIL-UMI.
from pathlib import Path

from hil_umi_advantage.shared import paths


def test_repo_relative_defaults(monkeypatch) -> None:
    for name in (
        "ADVANTAGE_ROOT",
        "ADVANTAGE_DATA_ROOT",
        "ADVANTAGE_CHECKPOINT_ROOT",
        "ADVANTAGE_PI05_BASE",
        "ADVANTAGE_OUTPUT_ROOT",
        "ADVANTAGE_CACHE_ROOT",
    ):
        monkeypatch.delenv(name, raising=False)

    root = paths.repo_root()
    assert paths.data_root() == root / "data"
    assert paths.checkpoint_root() == root / "checkpoints"
    assert paths.pi05_base() == root / "checkpoints/pi05_base"
    assert paths.output_root() == root / "outputs"
    assert paths.cache_root() == root / ".cache"


def test_environment_overrides(monkeypatch, tmp_path: Path) -> None:
    values = {
        "ADVANTAGE_ROOT": tmp_path / "root",
        "ADVANTAGE_DATA_ROOT": tmp_path / "datasets",
        "ADVANTAGE_CHECKPOINT_ROOT": tmp_path / "models",
        "ADVANTAGE_PI05_BASE": tmp_path / "external/pi05_base",
        "ADVANTAGE_OUTPUT_ROOT": tmp_path / "results",
        "ADVANTAGE_CACHE_ROOT": tmp_path / "cache",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, str(value))

    assert paths.repo_root() == values["ADVANTAGE_ROOT"]
    assert paths.data_root() == values["ADVANTAGE_DATA_ROOT"]
    assert paths.checkpoint_root() == values["ADVANTAGE_CHECKPOINT_ROOT"]
    assert paths.pi05_base() == values["ADVANTAGE_PI05_BASE"]
    assert paths.output_root() == values["ADVANTAGE_OUTPUT_ROOT"]
    assert paths.cache_root() == values["ADVANTAGE_CACHE_ROOT"]
