# Copyright 2026 HIL-UMI contributors.
# SPDX-License-Identifier: Apache-2.0

"""Policy and Advantage must resolve distinct Python namespaces."""

from __future__ import annotations

import ast
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ADVANTAGE = ROOT / "advantage"


def test_advantage_distribution_cannot_publish_the_policy_namespace() -> None:
    manifest = tomllib.loads((ADVANTAGE / "pyproject.toml").read_text())
    assert manifest["project"]["name"] == "hil-umi-advantage"
    assert (ADVANTAGE / "src/hil_umi_advantage/__init__.py").is_file()
    assert not (ADVANTAGE / "src/openpi").exists()
    # Keep the real policy dependency and upstream source identity intact.
    policy = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert policy["tool"]["uv"]["sources"]["openpi"]["git"] == (
        "https://github.com/RLinf/openpi"
    )


def test_advantage_runtime_has_no_imports_from_the_policy_fork() -> None:
    paths = [
        *(ADVANTAGE / "src/hil_umi_advantage").rglob("*.py"),
        *(ADVANTAGE / "scripts").rglob("*.py"),
        ROOT / "collection/ood/advantage_model_client.py",
    ]
    assert paths
    for path in paths:
        for node in ast.walk(ast.parse(path.read_text())):
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                modules = [node.module or ""]
            assert not any(
                module == "openpi" or module.startswith("openpi.")
                for module in modules
            ), f"{path.relative_to(ROOT)}:{node.lineno}: policy-fork import"


def test_policy_and_advantage_imports_do_not_shadow_each_other(tmp_path: Path) -> None:
    # Simulate the external policy dependency without downloading its GPU stack.
    policy = tmp_path / "openpi"
    policy.mkdir()
    (policy / "__init__.py").write_text("policy_marker = 'policy fork'\n")
    source = ADVANTAGE / "src"
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            "import sys; "
            f"sys.path[:0] = {[str(source), str(tmp_path)]!r}; "
            "import openpi; "
            "from hil_umi_advantage.training.config_names import canonicalize_config_name; "
            "from hil_umi_advantage.shared import paths; "
            "assert openpi.policy_marker == 'policy fork'; "
            "assert canonicalize_config_name('ADVANTAGE_TORCH_TOWEL_FOLD') "
            "== 'ADVANTAGE_TORCH_TOWEL_FOLD'",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
