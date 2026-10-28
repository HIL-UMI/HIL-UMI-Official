"""The Advantage package is importable without shadowing the policy's OpenPI."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]


def test_advantage_namespace_coexists_with_policy_openpi(tmp_path):
    policy_package = tmp_path / "openpi"
    policy_package.mkdir()
    (policy_package / "__init__.py").write_text('ROLE = "policy"\n')
    env = os.environ.copy()
    env["ADVANTAGE_PYTHON"] = sys.executable
    env.pop("ADVANTAGE_ROOT", None)
    env["PYTHONPATH"] = str(tmp_path)
    result = subprocess.run(
        [
            "bash",
            str(ROOT / "advantage"),
            "python",
            "-c",
            (
                "import json, openpi, hil_umi_advantage; "
                "from hil_umi_advantage.shared import paths; "
                "from hil_umi_advantage.training.config_names import canonicalize_config_name; "
                "print(json.dumps([openpi.ROLE, hil_umi_advantage.__name__, str(paths.repo_root()), "
                "canonicalize_config_name('ADVANTAGE_TORCH_KAI0_UMI')]))"
            ),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout) == ["policy", "hil_umi_advantage", str(ROOT), "ADVANTAGE_TORCH_UMI"]


def test_build_metadata_uses_distinct_advantage_distribution_and_package():
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert metadata["project"]["name"] == "hil-umi-advantage"
    assert metadata["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"] == ["src/hil_umi_advantage"]
    assert (ROOT / "src/hil_umi_advantage/__init__.py").is_file()
    assert not (ROOT / "src/openpi").exists()
    assert metadata["tool"]["uv"]["sources"]["openpi-client"] == {"workspace": True}
