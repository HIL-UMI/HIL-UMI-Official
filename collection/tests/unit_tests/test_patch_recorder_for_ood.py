# Copyright 2026 The HIL-UMI Authors.
# Modified for HIL-UMI.
# SPDX-License-Identifier: Apache-2.0

"""Patch contracts; proprietary recorder source is never a test fixture."""

from __future__ import annotations

import os
from pathlib import Path
from zipfile import ZipFile

import pytest
from scripts.patch_recorder_for_ood import (
    patch_camera_worker_py,
    patch_recorder_py,
    patch_wheel,
)


def test_unknown_recorder_layout_fails_without_modifying_source(tmp_path):
    source = tmp_path / "recorder.py"
    source.write_text("# Our deliberately unsupported test module\n", encoding="utf-8")
    original = source.read_bytes()
    with pytest.raises(RuntimeError, match="setup marker"):
        patch_recorder_py(str(source))
    assert source.read_bytes() == original


def test_unknown_camera_layout_fails_without_modifying_source(tmp_path):
    source = tmp_path / "camera_worker.py"
    source.write_text("# Our deliberately unsupported test module\n", encoding="utf-8")
    original = source.read_bytes()
    with pytest.raises(RuntimeError, match="capture gate"):
        patch_camera_worker_py(str(source))
    assert source.read_bytes() == original


def test_patch_cannot_replace_input_or_existing_output(tmp_path):
    source = tmp_path / "data_recorder-0.1.21-py3-none-any.whl"
    source.write_bytes(b"our-test-input")
    with pytest.raises(ValueError, match="differ from the input"):
        patch_wheel(source, tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    (output / source.name).write_bytes(b"our-existing-result")
    with pytest.raises(FileExistsError, match="Refusing to replace"):
        patch_wheel(source, output)
    assert source.read_bytes() == b"our-test-input"
    assert (output / source.name).read_bytes() == b"our-existing-result"


def test_patch_rejects_other_recorder_versions(tmp_path):
    source = tmp_path / "data_recorder-0.1.20-py3-none-any.whl"
    source.write_bytes(b"our-test-input")
    with pytest.raises(ValueError, match="0.1.21"):
        patch_wheel(source, tmp_path / "output")


@pytest.mark.integration
def test_external_recorder_patch_is_idempotent(tmp_path):
    configured = os.environ.get("COLLECTION_TEST_RECORDER_WHEEL")
    if not configured:
        pytest.skip("Set COLLECTION_TEST_RECORDER_WHEEL to your external 0.1.21 wheel")
    source = Path(configured).expanduser().resolve()
    if not source.is_file():
        pytest.fail(f"Explicit external recorder wheel not found: {source}")
    paths = [tmp_path / "recorder.py", tmp_path / "camera_worker.py"]
    with ZipFile(source) as wheel:
        for path in paths:
            path.write_bytes(wheel.read(f"data_recorder/{path.name}"))
    for patch, path in zip((patch_recorder_py, patch_camera_worker_py), paths):
        patch(str(path))
        first = path.read_text(encoding="utf-8")
        patch(str(path))
        assert path.read_text(encoding="utf-8") == first
        compile(first, str(path), "exec")
    recorder = paths[0].read_text(encoding="utf-8")
    camera = paths[1].read_text(encoding="utf-8")
    assert "if is_new_oculus_sample:" in recorder
    assert "if recording and is_new_oculus_sample:" not in recorder
    assert '"ood_ready": bool(_ood_ready)' in recorder
    assert "if capture_active and depth_frame is not None" in camera
