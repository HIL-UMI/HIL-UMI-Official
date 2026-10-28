#!/usr/bin/env python3
# Modified for HIL-UMI.
"""Build the canonical fold-towel dataset hierarchy without modifying sources."""

from __future__ import annotations

import argparse
import copy
import dataclasses
import datetime as dt
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

LOGGER = logging.getLogger("organize_fold_towel_datasets")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_ORIGIN = Path(os.environ.get("ADVANTAGE_DATA_ROOT", PROJECT_ROOT / "data")) / "source/base_origin"
DEFAULT_BASE_LABELED = Path(os.environ.get("ADVANTAGE_DATA_ROOT", PROJECT_ROOT / "data")) / "source/base_labeled"
DEFAULT_STAGE1_SOURCE_ROOT = Path(os.environ.get("ADVANTAGE_DATA_ROOT", PROJECT_ROOT / "data")) / "source/stage1"
DEFAULT_TARGET_ROOT = Path(os.environ.get("ADVANTAGE_DATA_ROOT", PROJECT_ROOT / "data")) / "fold_towel"
DEFAULT_TASK_PROMPT = "fold the towel on the table and put it into the basket"


@dataclasses.dataclass(frozen=True)
class FileInventory:
    entries: dict[str, tuple[str, int, str]]
    file_count: int
    total_bytes: int
    tree_sha256: str


@dataclasses.dataclass(frozen=True)
class SourceEpisode:
    run_dir: Path
    source_episode_index: int
    length: int
    parquet_path: Path
    video_paths: dict[str, Path]
    stats: dict[str, Any]


@dataclasses.dataclass(frozen=True)
class Stage1Preflight:
    source_info: dict[str, Any]
    video_keys: tuple[str, ...]
    episodes: tuple[SourceEpisode, ...]
    total_frames: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory_tree(root: Path) -> FileInventory:
    if not root.is_dir():
        raise FileNotFoundError(f"Inventory root is not a directory: {root}")
    entries: dict[str, tuple[str, int, str]] = {}
    total_bytes = 0
    for path in sorted(root.rglob("*")):
        relative_path = path.relative_to(root).as_posix()
        if path.is_symlink():
            target = str(path.readlink())
            entries[relative_path] = ("symlink", len(target.encode()), hashlib.sha256(target.encode()).hexdigest())
        elif path.is_file():
            size = path.stat().st_size
            entries[relative_path] = ("file", size, sha256_file(path))
            total_bytes += size

    digest = hashlib.sha256()
    for relative_path, record in entries.items():
        digest.update(json.dumps([relative_path, *record], separators=(",", ":")).encode())
        digest.update(b"\n")
    return FileInventory(
        entries=entries,
        file_count=len(entries),
        total_bytes=total_bytes,
        tree_sha256=digest.hexdigest(),
    )


def inventory_summary(root: Path, inventory: FileInventory) -> dict[str, Any]:
    return {
        "path": str(root.resolve()),
        "file_count": inventory.file_count,
        "total_bytes": inventory.total_bytes,
        "tree_sha256": inventory.tree_sha256,
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required JSONL file is missing: {path}")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def read_dataset_info(dataset_dir: Path) -> dict[str, Any]:
    info_path = dataset_dir / "meta/info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"LeRobot info.json is missing: {info_path}")
    return json.loads(info_path.read_text())


def format_dataset_path(
    template: str,
    *,
    chunk_index: int,
    episode_index: int,
    video_key: str | None = None,
) -> Path:
    return Path(
        template.format(
            chunk_index=chunk_index,
            episode_chunk=chunk_index,
            episode_index=episode_index,
            video_key=video_key,
        )
    )


def ffprobe_frame_count(path: Path) -> int:
    result = subprocess.run(
        [
            "/usr/bin/ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_frames",
            "-of",
            "default=nokey=1:noprint_wrappers=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    value = result.stdout.strip()
    if not value.isdigit():
        raise ValueError(f"ffprobe did not return an integer frame count for {path}: {value!r}")
    return int(value)


def _validate_base_dataset(dataset_dir: Path, *, labeled: bool) -> None:
    info = read_dataset_info(dataset_dir)
    required = {"data", "meta", "videos"}
    missing = sorted(name for name in required if not (dataset_dir / name).exists())
    if missing:
        raise FileNotFoundError(f"Dataset {dataset_dir} is missing required entries: {missing}")
    if int(info.get("total_episodes", -1)) <= 0 or int(info.get("total_frames", -1)) <= 0:
        raise ValueError(f"Dataset has invalid totals: {dataset_dir}")
    features = set(info.get("features", {}))
    advantage_features = {"absolute_advantage", "absolute_value", "relative_advantage", "stage_index"}
    if labeled and not advantage_features.issubset(features):
        raise ValueError(f"Labeled base dataset is missing advantage features: {dataset_dir}")
    if not labeled and features.intersection(advantage_features):
        raise ValueError(f"Origin base dataset unexpectedly contains advantage features: {dataset_dir}")


def _assert_source_info_compatible(reference: dict[str, Any], candidate: dict[str, Any], run_dir: Path) -> None:
    keys = ("codebase_version", "robot_type", "fps", "chunks_size", "features")
    differences = [key for key in keys if reference.get(key) != candidate.get(key)]
    if differences:
        raise ValueError(f"Stage1 source metadata differs for {run_dir}; incompatible keys={differences}")


def _validate_episode_indices(table: pa.Table, *, episode_index: int, global_start: int) -> None:
    num_rows = table.num_rows
    expected = {
        "episode_index": np.full(num_rows, episode_index, dtype=np.int64),
        "frame_index": np.arange(num_rows, dtype=np.int64),
        "index": np.arange(global_start, global_start + num_rows, dtype=np.int64),
        "task_index": np.zeros(num_rows, dtype=np.int64),
    }
    for name, values in expected.items():
        if name not in table.column_names:
            raise ValueError(f"Required parquet column is missing: {name}")
        actual = np.asarray(table[name].to_pylist(), dtype=np.int64)
        if not np.array_equal(actual, values):
            raise ValueError(f"Invalid {name} values for source episode {episode_index}")


def preflight_stage1_sources(
    source_root: Path,
    *,
    verify_video_frames: bool = True,
) -> Stage1Preflight:
    run_dirs = sorted(path for path in source_root.glob("run_*") if path.is_dir())
    if not run_dirs:
        raise FileNotFoundError(f"No run_* source datasets found under {source_root}")

    reference_info: dict[str, Any] | None = None
    reference_schema: pa.Schema | None = None
    video_keys: tuple[str, ...] | None = None
    episodes: list[SourceEpisode] = []
    total_frames = 0
    for run_dir in run_dirs:
        info = read_dataset_info(run_dir)
        if reference_info is None:
            reference_info = info
            video_keys = tuple(
                key for key, feature in info.get("features", {}).items() if feature.get("dtype") == "video"
            )
            if not video_keys:
                raise ValueError(f"No video features found in {run_dir}")
        else:
            _assert_source_info_compatible(reference_info, info, run_dir)

        episode_records = read_jsonl(run_dir / "meta/episodes.jsonl")
        episode_stats_records = read_jsonl(run_dir / "meta/episodes_stats.jsonl")
        episode_stats = {int(record["episode_index"]): record.get("stats", {}) for record in episode_stats_records}
        total_episodes = int(info["total_episodes"])
        if len(episode_records) != total_episodes or len(episode_stats) != total_episodes:
            raise ValueError(f"Episode metadata count mismatch in {run_dir}")
        if [int(record["episode_index"]) for record in episode_records] != list(range(total_episodes)):
            raise ValueError(f"Episode indices are not contiguous in {run_dir}")

        run_global_start = 0
        for episode_record in episode_records:
            source_episode_index = int(episode_record["episode_index"])
            source_chunk_index = source_episode_index // int(info["chunks_size"])
            parquet_path = run_dir / format_dataset_path(
                info["data_path"],
                chunk_index=source_chunk_index,
                episode_index=source_episode_index,
            )
            if not parquet_path.is_file():
                raise FileNotFoundError(f"Source parquet is missing: {parquet_path}")
            table = pq.read_table(parquet_path)
            length = int(episode_record["length"])
            if table.num_rows != length:
                raise ValueError(f"Parquet row count mismatch for {parquet_path}: {table.num_rows} != {length}")
            _validate_episode_indices(table, episode_index=source_episode_index, global_start=run_global_start)
            if reference_schema is None:
                reference_schema = table.schema
            elif not reference_schema.equals(table.schema):
                raise ValueError(f"Parquet schema mismatch: {parquet_path}")

            source_videos: dict[str, Path] = {}
            assert video_keys is not None
            for video_key in video_keys:
                video_path = run_dir / format_dataset_path(
                    info["video_path"],
                    chunk_index=source_chunk_index,
                    episode_index=source_episode_index,
                    video_key=video_key,
                )
                if not video_path.is_file():
                    raise FileNotFoundError(f"Source video is missing: {video_path}")
                if verify_video_frames and ffprobe_frame_count(video_path) != length:
                    raise ValueError(f"Video frame count does not match parquet rows: {video_path}")
                source_videos[video_key] = video_path

            episodes.append(
                SourceEpisode(
                    run_dir=run_dir,
                    source_episode_index=source_episode_index,
                    length=length,
                    parquet_path=parquet_path,
                    video_paths=source_videos,
                    stats=episode_stats[source_episode_index],
                )
            )
            run_global_start += length
            total_frames += length
        if run_global_start != int(info["total_frames"]):
            raise ValueError(f"Frame total mismatch in {run_dir}: {run_global_start} != {info['total_frames']}")

    assert reference_info is not None
    assert video_keys is not None
    return Stage1Preflight(
        source_info=reference_info,
        video_keys=video_keys,
        episodes=tuple(episodes),
        total_frames=total_frames,
    )


def _replace_integer_column(table: pa.Table, name: str, values: np.ndarray) -> pa.Table:
    column_index = table.schema.get_field_index(name)
    if column_index < 0:
        raise ValueError(f"Required column is missing: {name}")
    arrow_type = table.schema.field(column_index).type
    return table.set_column(column_index, name, pa.array(values, type=arrow_type))


def compute_array_stats(values: np.ndarray) -> dict[str, list[Any]]:
    values = np.asarray(values)
    if values.ndim == 0 or len(values) == 0:
        raise ValueError("Statistics require a non-empty array with a frame dimension")
    if values.ndim == 1:
        values = values[:, None]
    return {
        "min": np.min(values, axis=0).tolist(),
        "max": np.max(values, axis=0).tolist(),
        "mean": np.mean(values, axis=0).tolist(),
        "std": np.std(values, axis=0).tolist(),
        "count": [len(values)],
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q10": np.quantile(values, 0.10, axis=0).tolist(),
        "q50": np.quantile(values, 0.50, axis=0).tolist(),
        "q90": np.quantile(values, 0.90, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
    }


def aggregate_episode_stats(episode_stats: list[dict[str, Any]]) -> dict[str, Any]:
    feature_names = sorted({name for stats in episode_stats for name in stats})
    output: dict[str, Any] = {}
    for feature_name in feature_names:
        records = [stats[feature_name] for stats in episode_stats if feature_name in stats]
        required = {"min", "max", "mean", "std", "count"}
        if any(not required.issubset(record) for record in records):
            continue
        means = np.stack([np.asarray(record["mean"], dtype=np.float64) for record in records])
        variances = np.stack([np.asarray(record["std"], dtype=np.float64) ** 2 for record in records])
        counts = np.asarray([int(record["count"][0]) for record in records], dtype=np.float64)
        expanded_counts = counts.reshape((len(counts),) + (1,) * (means.ndim - 1))
        total_count = float(np.sum(counts))
        mean = np.sum(means * expanded_counts, axis=0) / total_count
        variance = np.sum((variances + (means - mean) ** 2) * expanded_counts, axis=0) / total_count
        output[feature_name] = {
            "min": np.min(np.stack([np.asarray(record["min"]) for record in records]), axis=0).tolist(),
            "max": np.max(np.stack([np.asarray(record["max"]) for record in records]), axis=0).tolist(),
            "mean": mean.tolist(),
            "std": np.sqrt(variance).tolist(),
            "count": [int(total_count)],
        }
    return output


def merge_stage1_sources(
    preflight: Stage1Preflight,
    target_dir: Path,
    *,
    task_prompt: str,
    source_inventories: dict[Path, FileInventory],
) -> dict[str, Any]:
    if target_dir.exists():
        raise FileExistsError(f"Refusing to merge into an existing target: {target_dir}")
    (target_dir / "data").mkdir(parents=True)
    (target_dir / "videos").mkdir()
    (target_dir / "meta").mkdir()

    info = copy.deepcopy(preflight.source_info)
    total_episodes = len(preflight.episodes)
    chunks_size = int(info["chunks_size"])
    info.update(
        {
            "total_episodes": total_episodes,
            "total_frames": preflight.total_frames,
            "total_tasks": 1,
            "total_videos": total_episodes * len(preflight.video_keys),
            "total_chunks": (total_episodes + chunks_size - 1) // chunks_size,
            "splits": {"train": f"0:{total_episodes}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        }
    )

    episodes_records: list[dict[str, Any]] = []
    episode_stats_records: list[dict[str, Any]] = []
    mapping: list[dict[str, Any]] = []
    global_index = 0
    for target_episode_index, source_episode in enumerate(preflight.episodes):
        target_chunk_index = target_episode_index // chunks_size
        # Recorder exports carry Hugging Face metadata with the retired `_type=List` encoding.
        # Arrow data and field types are valid, so discard only that stale schema metadata.
        table = pq.read_table(source_episode.parquet_path).replace_schema_metadata(None)
        table = _replace_integer_column(
            table,
            "episode_index",
            np.full(source_episode.length, target_episode_index, dtype=np.int64),
        )
        table = _replace_integer_column(
            table,
            "index",
            np.arange(global_index, global_index + source_episode.length, dtype=np.int64),
        )
        table = _replace_integer_column(
            table,
            "task_index",
            np.zeros(source_episode.length, dtype=np.int64),
        )
        target_parquet = target_dir / format_dataset_path(
            info["data_path"],
            chunk_index=target_chunk_index,
            episode_index=target_episode_index,
        )
        target_parquet.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, target_parquet, compression="zstd")

        target_video_hashes: dict[str, str] = {}
        for video_key, source_video in source_episode.video_paths.items():
            target_video = target_dir / format_dataset_path(
                info["video_path"],
                chunk_index=target_chunk_index,
                episode_index=target_episode_index,
                video_key=video_key,
            )
            target_video.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_video, target_video)
            source_hash = sha256_file(source_video)
            target_hash = sha256_file(target_video)
            if source_hash != target_hash:
                raise RuntimeError(f"Copied video checksum mismatch: {source_video} -> {target_video}")
            target_video_hashes[video_key] = target_hash

        stats = copy.deepcopy(source_episode.stats)
        for column_name in ("episode_index", "index", "task_index"):
            stats[column_name] = compute_array_stats(np.asarray(table[column_name].to_pylist()))
        episodes_records.append(
            {"episode_index": target_episode_index, "tasks": [task_prompt], "length": source_episode.length}
        )
        episode_stats_records.append({"episode_index": target_episode_index, "stats": stats})
        mapping.append(
            {
                "target_episode_index": target_episode_index,
                "source_run": source_episode.run_dir.name,
                "source_dataset": str(source_episode.run_dir.resolve()),
                "source_episode_index": source_episode.source_episode_index,
                "length": source_episode.length,
                "source_parquet_sha256": sha256_file(source_episode.parquet_path),
                "target_parquet_sha256": sha256_file(target_parquet),
                "video_sha256": target_video_hashes,
            }
        )
        global_index += source_episode.length

    write_json(target_dir / "meta/info.json", info)
    write_jsonl(target_dir / "meta/tasks.jsonl", [{"task_index": 0, "task": task_prompt}])
    write_jsonl(target_dir / "meta/episodes.jsonl", episodes_records)
    write_jsonl(target_dir / "meta/episodes_stats.jsonl", episode_stats_records)
    write_json(
        target_dir / "meta/stats.json",
        aggregate_episode_stats([record["stats"] for record in episode_stats_records]),
    )
    manifest = {
        "schema_version": 1,
        "task": "fold_towel",
        "stage": "stage1",
        "variant": "origin",
        "task_prompt": task_prompt,
        "created_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "merge_order": "source run name ascending, then source episode_index ascending",
        "total_episodes": total_episodes,
        "total_frames": preflight.total_frames,
        "source_inventories": [
            inventory_summary(run_dir, source_inventories[run_dir]) for run_dir in sorted(source_inventories)
        ],
        "episodes": mapping,
    }
    write_json(target_dir / "meta/merge_manifest.json", manifest)
    return manifest


def strip_stale_parquet_metadata(dataset_dir: Path) -> int:
    """Repair a generated merge by removing stale Hugging Face schema metadata."""
    info = read_dataset_info(dataset_dir)
    manifest_path = dataset_dir / "meta/merge_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    repaired = 0
    for episode in manifest["episodes"]:
        episode_index = int(episode["target_episode_index"])
        chunk_index = episode_index // int(info["chunks_size"])
        parquet_path = dataset_dir / format_dataset_path(
            info["data_path"], chunk_index=chunk_index, episode_index=episode_index
        )
        table = pq.read_table(parquet_path)
        if table.schema.metadata:
            temporary_path = parquet_path.with_name(f".{parquet_path.name}.metadata-repair.tmp")
            pq.write_table(table.replace_schema_metadata(None), temporary_path, compression="zstd")
            os.replace(temporary_path, parquet_path)
            repaired += 1
        episode["target_parquet_sha256"] = sha256_file(parquet_path)
    write_json(manifest_path, manifest)
    return repaired


def _rewrite_base_labeled_provenance(labeled_dir: Path, final_target_root: Path) -> None:
    origin_dir = (final_target_root / "base/origin").resolve()
    final_labeled_dir = (final_target_root / "base/labeled").resolve()
    annotations = origin_dir / "augmentation_metadata.json"
    updates = {
        "meta/advantage_prediction.json": {
            "source_dataset": str(origin_dir),
            "annotations": str(annotations),
        },
        "meta/acbc_labeling.json": {
            "source_dataset": str(origin_dir),
            "output_dataset": str(final_labeled_dir),
            "annotations": str(annotations),
        },
        "meta/acbc_source.json": {
            "source_dataset": str(origin_dir),
            "annotation_file": str(annotations),
        },
    }
    for relative_path, values in updates.items():
        path = labeled_dir / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"Expected labeled provenance file is missing: {path}")
        payload = json.loads(path.read_text())
        payload.update(values)
        write_json(path, payload)


def _write_layout_documentation(target_root: Path, task_prompt: str) -> None:
    (target_root / "README.md").write_text(
        f"""# Fold Towel datasets

Canonical hierarchy: `task/stage/variant`, where variants are `origin` (no positive/negative ACBC label) and
`labeled` (positive/negative ACBC label available).

Task prompt: `{task_prompt}`

| Stage | Origin | Labeled |
|---|---|---|
| base | 51 episodes / 47,683 frames | 51 episodes / 47,683 frames, four-stage top-30 ACBC labels |
| stage1 | 44 episodes / 12,520 frames, merged from `fold_towel_umi_adv` | pending |
| stage2 | pending | pending |
| stage3 | pending | pending |

The legacy source directories are intentionally retained unchanged. Stage1 recorder sidecars (`ood_replay`, crash
logs, and recorder checkpoints) remain in `data/source/stage1`; `stage1/origin/meta/merge_manifest.json`
records the episode-level provenance.
"""
    )
    pending = {
        "stage1/labeled": "No trustworthy per-frame positive/negative labels have been generated yet.\n",
        "stage2/origin": "No stage2 origin dataset has been collected yet.\n",
        "stage2/labeled": "No stage2 labeled dataset has been generated yet.\n",
        "stage3/origin": "No stage3 origin dataset has been collected yet.\n",
        "stage3/labeled": "No stage3 labeled dataset has been generated yet.\n",
    }
    for relative_dir, message in pending.items():
        directory = target_root / relative_dir
        directory.mkdir(parents=True)
        (directory / "STATUS.md").write_text(message)


def _validate_merged_stage1(dataset_dir: Path, expected: Stage1Preflight, task_prompt: str) -> None:
    info = read_dataset_info(dataset_dir)
    if int(info["total_episodes"]) != len(expected.episodes) or int(info["total_frames"]) != expected.total_frames:
        raise ValueError("Merged stage1 totals do not match the preflight result")
    if "{episode_chunk" not in info["data_path"] or "{episode_chunk" not in info["video_path"]:
        raise ValueError("Merged stage1 did not normalize its LeRobot path templates")
    tasks = read_jsonl(dataset_dir / "meta/tasks.jsonl")
    if tasks != [{"task_index": 0, "task": task_prompt}]:
        raise ValueError("Merged stage1 task metadata is incorrect")

    global_index = 0
    for episode_index, source_episode in enumerate(expected.episodes):
        chunk_index = episode_index // int(info["chunks_size"])
        parquet_path = dataset_dir / format_dataset_path(
            info["data_path"], chunk_index=chunk_index, episode_index=episode_index
        )
        table = pq.read_table(parquet_path)
        _validate_episode_indices(table, episode_index=episode_index, global_start=global_index)
        if table.num_rows != source_episode.length:
            raise ValueError(f"Merged parquet length mismatch: {parquet_path}")
        for video_key, source_video in source_episode.video_paths.items():
            target_video = dataset_dir / format_dataset_path(
                info["video_path"],
                chunk_index=chunk_index,
                episode_index=episode_index,
                video_key=video_key,
            )
            if sha256_file(source_video) != sha256_file(target_video):
                raise ValueError(f"Merged video differs from its source: {target_video}")
        global_index += table.num_rows


def _smoke_load_dataset(dataset_dir: Path, repo_id: str, expected_episodes: int, expected_frames: int) -> None:
    cache_dir = Path("/tmp/advantage-fold-towel-hf-cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_DATASETS_CACHE"] = str(cache_dir)
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(repo_id=repo_id, root=dataset_dir)
    if dataset.num_episodes != expected_episodes or dataset.num_frames != expected_frames:
        raise ValueError(
            f"LeRobot load returned incorrect totals for {dataset_dir}: {dataset.num_episodes}/{dataset.num_frames}"
        )


def organize_fold_towel_datasets(
    *,
    base_origin: Path,
    base_labeled: Path,
    stage1_source_root: Path,
    target_root: Path,
    task_prompt: str,
    dry_run: bool,
    verify_video_frames: bool = True,
) -> dict[str, Any]:
    paths = [base_origin, base_labeled, stage1_source_root, target_root]
    base_origin, base_labeled, stage1_source_root, target_root = [path.expanduser().resolve() for path in paths]
    if target_root.exists():
        raise FileExistsError(f"Refusing to overwrite an existing target: {target_root}")
    _validate_base_dataset(base_origin, labeled=False)
    _validate_base_dataset(base_labeled, labeled=True)
    base_origin_info = read_dataset_info(base_origin)
    base_labeled_info = read_dataset_info(base_labeled)
    stage1 = preflight_stage1_sources(stage1_source_root, verify_video_frames=verify_video_frames)

    run_dirs = sorted({episode.run_dir for episode in stage1.episodes})
    source_roots = [base_origin, base_labeled, *run_dirs]
    source_inventories = {source: inventory_tree(source) for source in source_roots}
    summary = {
        "target_root": str(target_root),
        "task_prompt": task_prompt,
        "base_origin": {
            "total_episodes": int(base_origin_info["total_episodes"]),
            "total_frames": int(base_origin_info["total_frames"]),
        },
        "base_labeled": {
            "total_episodes": int(base_labeled_info["total_episodes"]),
            "total_frames": int(base_labeled_info["total_frames"]),
        },
        "stage1_origin": {"total_episodes": len(stage1.episodes), "total_frames": stage1.total_frames},
        "sources": [inventory_summary(source, source_inventories[source]) for source in source_roots],
    }
    if dry_run:
        LOGGER.info("Dry-run preflight succeeded:\n%s", json.dumps(summary, indent=2, sort_keys=True))
        return summary

    timestamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    staging_root = target_root.with_name(f".{target_root.name}.building-{timestamp}")
    if staging_root.exists():
        raise FileExistsError(f"Staging directory already exists: {staging_root}")
    LOGGER.info("Building canonical hierarchy in %s", staging_root)
    staging_root.mkdir(parents=True)
    shutil.copytree(base_origin, staging_root / "base/origin", symlinks=False)
    shutil.copytree(base_labeled, staging_root / "base/labeled", symlinks=False)
    if (staging_root / "base/labeled/videos").is_symlink():
        raise RuntimeError("base/labeled/videos must be materialized, not left as a source-dependent symlink")
    _rewrite_base_labeled_provenance(staging_root / "base/labeled", target_root)
    merge_stage1_sources(
        stage1,
        staging_root / "stage1/origin",
        task_prompt=task_prompt,
        source_inventories={run_dir: source_inventories[run_dir] for run_dir in run_dirs},
    )
    _write_layout_documentation(staging_root, task_prompt)
    _validate_merged_stage1(staging_root / "stage1/origin", stage1, task_prompt)
    _smoke_load_dataset(
        staging_root / "base/origin",
        "fold_towel_base_origin",
        int(base_origin_info["total_episodes"]),
        int(base_origin_info["total_frames"]),
    )
    _smoke_load_dataset(
        staging_root / "base/labeled",
        "fold_towel_base_labeled",
        int(base_labeled_info["total_episodes"]),
        int(base_labeled_info["total_frames"]),
    )
    _smoke_load_dataset(
        staging_root / "stage1/origin",
        "fold_towel_stage1_origin",
        len(stage1.episodes),
        stage1.total_frames,
    )

    for source, before in source_inventories.items():
        after = inventory_tree(source)
        if before.entries != after.entries:
            raise RuntimeError(f"Source data changed while organizing; refusing to publish: {source}")
    staging_root.rename(target_root)
    LOGGER.info("Published canonical fold-towel hierarchy at %s", target_root)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-origin", type=Path, default=DEFAULT_BASE_ORIGIN)
    parser.add_argument("--base-labeled", type=Path, default=DEFAULT_BASE_LABELED)
    parser.add_argument("--stage1-source-root", type=Path, default=DEFAULT_STAGE1_SOURCE_ROOT)
    parser.add_argument("--target-root", type=Path, default=DEFAULT_TARGET_ROOT)
    parser.add_argument("--task-prompt", default=DEFAULT_TASK_PROMPT)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    organize_fold_towel_datasets(
        base_origin=args.base_origin,
        base_labeled=args.base_labeled,
        stage1_source_root=args.stage1_source_root,
        target_root=args.target_root,
        task_prompt=args.task_prompt,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
