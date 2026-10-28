# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
# Modified for HIL-UMI.

"""Named, reusable scene-reference sets for real-world evaluation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

_SAFE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


@dataclass(frozen=True)
class SceneReferenceSource:
    """One contiguous range of recorded videos used for offline calibration."""

    epoch_start: int
    count: int
    video_dir: Path
    video_suffix: str


@dataclass(frozen=True)
class SceneReferenceSet:
    """A named collection of per-epoch reference images and source videos."""

    name: str
    root_dir: Path
    view_name: str
    width: int
    height: int
    num_references: int
    references_dir: Path
    sources: tuple[SceneReferenceSource, ...]

    @classmethod
    def load(cls, preset_dir: str | Path) -> SceneReferenceSet:
        """Load and validate ``manifest.json`` under ``preset_dir``."""
        root_dir = Path(preset_dir).expanduser().resolve()
        manifest_path = root_dir / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Scene-reference manifest does not exist: {manifest_path}"
            )
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid scene-reference manifest JSON: {manifest_path}"
            ) from exc
        if manifest.get("version") != 1:
            raise ValueError("Scene-reference manifest version must be 1.")

        name = str(manifest.get("name", ""))
        if not _SAFE_NAME_PATTERN.fullmatch(name):
            raise ValueError(
                "Scene-reference name must contain only letters, digits, '-' and "
                f"'_', got {name!r}."
            )
        view_name = str(manifest.get("view_name", "")).strip()
        if not _SAFE_NAME_PATTERN.fullmatch(view_name):
            raise ValueError(
                "Scene-reference manifest view_name must contain only letters, "
                "digits, '-' and '_'."
            )
        resolution = manifest.get("resolution")
        if not (
            isinstance(resolution, list)
            and len(resolution) == 2
            and all(isinstance(value, int) and value > 0 for value in resolution)
        ):
            raise ValueError(
                "Scene-reference manifest resolution must be [width, height]."
            )
        num_references = manifest.get("num_references")
        if not isinstance(num_references, int) or num_references <= 0:
            raise ValueError(
                "Scene-reference manifest num_references must be positive."
            )

        references_dir = cls._resolve_path(
            root_dir,
            manifest.get("references_dir", "references"),
            field_name="references_dir",
        )
        raw_sources = manifest.get("sources", [])
        if not isinstance(raw_sources, list):
            raise ValueError("Scene-reference manifest sources must be a list.")
        sources = []
        covered_epochs: set[int] = set()
        for raw_source in raw_sources:
            if not isinstance(raw_source, dict):
                raise ValueError("Each scene-reference source must be an object.")
            epoch_start = raw_source.get("epoch_start")
            count = raw_source.get("count")
            if not isinstance(epoch_start, int) or epoch_start < 0:
                raise ValueError("Source epoch_start must be non-negative.")
            if not isinstance(count, int) or count <= 0:
                raise ValueError("Source count must be positive.")
            video_suffix = str(raw_source.get("video_suffix", "wrist_2")).strip()
            if not video_suffix:
                raise ValueError("Source video_suffix must not be empty.")
            video_dir = cls._resolve_path(
                root_dir,
                raw_source.get("video_dir"),
                field_name="sources[].video_dir",
            )
            source_epochs = set(range(epoch_start, epoch_start + count))
            overlap = covered_epochs & source_epochs
            if overlap:
                raise ValueError(
                    f"Scene-reference source epoch ranges overlap at {sorted(overlap)}."
                )
            covered_epochs.update(source_epochs)
            sources.append(
                SceneReferenceSource(
                    epoch_start=epoch_start,
                    count=count,
                    video_dir=video_dir,
                    video_suffix=video_suffix,
                )
            )
        expected_epochs = set(range(num_references))
        if covered_epochs != expected_epochs:
            missing = sorted(expected_epochs - covered_epochs)
            unexpected = sorted(covered_epochs - expected_epochs)
            raise ValueError(
                "Scene-reference sources must cover every epoch exactly once; "
                f"missing={missing}, unexpected={unexpected}."
            )

        return cls(
            name=name,
            root_dir=root_dir,
            view_name=view_name,
            width=resolution[0],
            height=resolution[1],
            num_references=num_references,
            references_dir=references_dir,
            sources=tuple(sources),
        )

    @staticmethod
    def _resolve_path(root_dir: Path, value: object, *, field_name: str) -> Path:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Scene-reference {field_name} must be a path string.")
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = root_dir / path
        return path.resolve()

    def reference_image_path(self, epoch_index: int) -> Path:
        """Return the persisted reference image path for ``epoch_index``."""
        self._validate_epoch(epoch_index)
        return self.references_dir / f"{epoch_index}_{self.view_name}_reference.jpg"

    def reference_metadata_path(self, epoch_index: int) -> Path:
        """Return the persisted selection metadata path for ``epoch_index``."""
        self._validate_epoch(epoch_index)
        return self.references_dir / f"{epoch_index}_{self.view_name}_reference.json"

    def source_video(self, epoch_index: int) -> tuple[Path, int]:
        """Return ``(video_path, local_index)`` for one global epoch."""
        self._validate_epoch(epoch_index)
        for source in self.sources:
            if source.epoch_start <= epoch_index < source.epoch_start + source.count:
                local_index = epoch_index - source.epoch_start
                return (
                    source.video_dir / f"{local_index}_{source.video_suffix}.mp4",
                    local_index,
                )
        raise RuntimeError(f"No source video mapping for epoch {epoch_index}.")

    def validate_saved_references(self) -> None:
        """Require an image and metadata file for every epoch."""
        missing = []
        for epoch_index in range(self.num_references):
            if not self.reference_image_path(epoch_index).is_file() or not (
                self.reference_metadata_path(epoch_index).is_file()
            ):
                missing.append(epoch_index)
        if missing:
            raise FileNotFoundError(
                f"Scene-reference preset {self.name!r} is not fully calibrated; "
                f"missing epochs: {missing}."
            )

    def video_paths(self) -> dict[int, Path]:
        """Return all global-epoch to source-video mappings."""
        return {
            epoch_index: self.source_video(epoch_index)[0]
            for epoch_index in range(self.num_references)
        }

    def source_epoch_indices(self) -> dict[int, int]:
        """Return all global-epoch to source-local-index mappings."""
        return {
            epoch_index: self.source_video(epoch_index)[1]
            for epoch_index in range(self.num_references)
        }

    def _validate_epoch(self, epoch_index: int) -> None:
        if not 0 <= int(epoch_index) < self.num_references:
            raise IndexError(
                f"Scene-reference epoch must be in [0, {self.num_references - 1}], "
                f"got {epoch_index}."
            )


def default_scene_reference_root() -> Path:
    """Return this checkout's default named-reference directory."""
    return Path(__file__).resolve().parents[4] / "assets" / "scene_references"


def resolve_scene_reference_preset(
    name: str,
    *,
    root_dir: str | Path | None = None,
) -> Path:
    """Resolve a safe preset name below the configured reference root."""
    if not _SAFE_NAME_PATTERN.fullmatch(name):
        raise ValueError(
            "Scene-reference name must contain only letters, digits, '-' and '_'."
        )
    root = (
        Path(root_dir).expanduser().resolve()
        if root_dir is not None
        else default_scene_reference_root()
    )
    preset_dir = root / name
    preset = SceneReferenceSet.load(preset_dir)
    if preset.name != name:
        raise ValueError(
            f"Scene-reference manifest name {preset.name!r} does not match {name!r}."
        )
    return preset_dir
