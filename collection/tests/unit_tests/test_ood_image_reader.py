# Copyright 2026 The RLinf Authors.
# Modified for HIL-UMI.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys

import numpy as np
import ood.image_reader as image_reader_module
import pytest
from ood.image_reader import CameraConfig, SharedMemoryRingReader


class _FakeSharedMemory:
    def __init__(self, name: str) -> None:
        self._name = name


def test_reader_reattaches_when_old_ring_frame_id_stalls(monkeypatch):
    clock = {"now": 0.0}
    generation = {"current": 0}
    images = [
        np.full((1, 2, 3), 7, dtype=np.uint8),
        np.full((1, 2, 3), 55, dtype=np.uint8),
    ]
    metadata = [
        np.array([[2, 99]], dtype=np.uint64),
        np.array([[2, 0]], dtype=np.uint64),
    ]

    class FakeRing:
        def __init__(self, **_kwargs) -> None:
            self.generation = generation["current"]
            self._meta_arr = metadata[self.generation]
            self.shm = _FakeSharedMemory(f"rgb-{self.generation}")
            self.meta_shm = _FakeSharedMemory(f"meta-{self.generation}")
            self.closed = False

        @property
        def supports_deterministic_validation(self) -> bool:
            return True

        def read_verified(self, _slot, _shape, _dtype, expected_frame_id=None):
            if int(self._meta_arr[0, 1]) != expected_frame_id:
                return None
            return images[self.generation].copy()

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(
        image_reader_module.resource_tracker,
        "unregister",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        image_reader_module.time,
        "monotonic",
        lambda: clock["now"],
    )
    reader = SharedMemoryRingReader(
        CameraConfig(
            name="front",
            ring_name="test-front",
            width=2,
            height=1,
            ring_slots=1,
            stale_reconnect_s=1.0,
        ),
        ring_factory=FakeRing,
    )

    first = reader.read_latest_frame()
    assert first is not None
    assert first.frame_id == 99
    assert np.all(first.image == 7)

    # Recorder has unlinked generation 0 and created generation 1 under the
    # same shared-memory name. The old attachment remains readable but frozen.
    generation["current"] = 1
    clock["now"] = 1.1
    replacement = reader.read_latest_frame()

    assert replacement is not None
    assert replacement.frame_id == 0
    assert np.all(replacement.image == 55)


def test_reader_ignores_never_written_zero_metadata(monkeypatch):
    class EmptyRing:
        supports_deterministic_validation = True
        _meta_arr = np.array([[0, 0]], dtype=np.uint64)
        shm = _FakeSharedMemory("empty-rgb")
        meta_shm = _FakeSharedMemory("empty-meta")

        def __init__(self, **_kwargs) -> None:
            pass

        def read_verified(self, *_args, **_kwargs):
            raise AssertionError("an unwritten slot must not be read")

        def close(self) -> None:
            pass

    monkeypatch.setattr(
        image_reader_module.resource_tracker,
        "unregister",
        lambda *_args: None,
    )
    reader = SharedMemoryRingReader(
        CameraConfig("front", "empty-front", width=2, height=1, ring_slots=1),
        ring_factory=EmptyRing,
    )

    assert reader.read_latest_frame() is None


def test_reader_returns_only_the_exact_requested_historic_frame(monkeypatch):
    image = np.full((1, 2, 3), 19, dtype=np.uint8)

    class HistoricRing:
        supports_deterministic_validation = True
        _meta_arr = np.array([[2, 8], [2, 9], [2, 10]], dtype=np.uint64)
        shm = _FakeSharedMemory("historic-rgb")
        meta_shm = _FakeSharedMemory("historic-meta")

        def __init__(self, **_kwargs) -> None:
            pass

        def read_verified(self, slot, _shape, _dtype, expected_frame_id=None):
            if slot != expected_frame_id % 3 or expected_frame_id != 10:
                return None
            return image.copy()

        def close(self) -> None:
            pass

    monkeypatch.setattr(
        image_reader_module.resource_tracker, "unregister", lambda *_args: None
    )
    reader = SharedMemoryRingReader(
        CameraConfig("front", "historic-front", width=2, height=1, ring_slots=3),
        ring_factory=HistoricRing,
    )

    retained = reader.read_frame_id(10)
    assert retained is not None
    assert retained.frame_id == 10
    assert np.all(retained.image == 19)
    assert reader.read_frame_id(7) is None


def test_missing_external_recorder_has_clear_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "data_recorder", None)
    monkeypatch.delitem(sys.modules, "data_recorder.shared_memory_ring", raising=False)
    reader = SharedMemoryRingReader(
        CameraConfig("front", "not-created", width=2, height=1, ring_slots=1)
    )
    with pytest.raises(RuntimeError, match="externally supplied data_recorder 0.1.21"):
        reader.connect()


def test_external_ring_loads_only_explicit_verified_source(tmp_path, monkeypatch):
    import hashlib

    # Original test code, not a copied recorder implementation.
    source = b"class SharedMemoryRing: pass\n"
    external = tmp_path / "external-recorder" / "shared_memory_ring.py"
    external.parent.mkdir()
    external.write_bytes(source)
    monkeypatch.setenv("COLLECTION_RECORDER_RING_MODULE", str(external))
    monkeypatch.setattr(
        image_reader_module, "_RECORDER_RING_SHA256", hashlib.sha256(source).hexdigest()
    )
    factory = image_reader_module.recorder_ring_factory()
    assert factory.__name__ == "SharedMemoryRing"
    assert factory.__module__ == "_hil_umi_external_recorder_ring"


def test_external_ring_rejects_unknown_recorder_source(tmp_path, monkeypatch):
    external = tmp_path / "shared_memory_ring.py"
    external.write_text("raise AssertionError('unverified code must not execute')\n")
    monkeypatch.setenv("COLLECTION_RECORDER_RING_MODULE", str(external))
    with pytest.raises(RuntimeError, match="does not match the verified"):
        image_reader_module.recorder_ring_factory()
