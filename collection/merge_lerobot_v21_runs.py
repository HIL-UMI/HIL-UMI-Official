#!/usr/bin/env python3
# Modified for HIL-UMI.
# Copyright 2026 The HIL-UMI Authors.
# SPDX-License-Identifier: Apache-2.0

"""Merge compatible LeRobot v2.1 runs without modifying the source runs.

The merger keeps the legacy v2.1 episode-per-file layout, reindexes episode
and global frame indices, copies per-episode videos without re-encoding, and
rebuilds both per-episode and dataset-level statistics.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from lerobot.datasets.compute_stats import aggregate_stats

CONSISTENT_INFO_FIELDS = (
    "codebase_version",
    "robot_type",
    "fps",
    "chunks_size",
    "data_files_size_in_mb",
    "video_files_size_in_mb",
    "data_path",
    "video_path",
    "features",
)
LOCATION_STATS = ("min", "max", "mean", "q01", "q10", "q50", "q90", "q99")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected a JSON object in {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RuntimeError(f"Expected an object at {path}:{line_number}")
            rows.append(value)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(value), handle, indent=4, ensure_ascii=False)
        handle.write("\n")
    tmp.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(to_jsonable(row), ensure_ascii=False))
            handle.write("\n")
    tmp.replace(path)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def stats_to_numpy(stats: dict[str, Any]) -> dict[str, dict[str, np.ndarray]]:
    return {
        feature: {
            stat_name: np.asarray(stat_value)
            for stat_name, stat_value in feature_stats.items()
        }
        for feature, feature_stats in stats.items()
    }


def formatted_path(
    root: Path,
    pattern: str,
    *,
    episode_index: int,
    chunks_size: int,
    video_key: str | None = None,
) -> Path:
    chunk_index = episode_index // chunks_size
    return root / pattern.format(
        chunk_index=chunk_index,
        episode_chunk=chunk_index,
        episode_index=episode_index,
        video_key=video_key,
    )


def replace_int64_column(table: pa.Table, name: str, values: np.ndarray) -> pa.Table:
    if name not in table.column_names:
        raise RuntimeError(f"Parquet table is missing required column {name!r}")
    column_index = table.column_names.index(name)
    field = table.schema.field(column_index)
    if not pa.types.is_integer(field.type):
        raise RuntimeError(f"Expected integer column {name!r}, got {field.type}")
    array = pa.array(values, type=field.type)
    return table.set_column(column_index, field, array)


def shifted_stat_values(values: Any, offset: int) -> list[Any]:
    if not isinstance(values, list):
        values = [values]
    return [value + offset for value in values]


def update_episode_stats(
    original: dict[str, Any],
    *,
    new_episode_index: int,
    new_frame_start: int,
    old_frame_start: int,
    frame_count: int,
) -> dict[str, Any]:
    result = copy.deepcopy(original)
    episode_stats = result.get("episode_index")
    index_stats = result.get("index")
    if not isinstance(episode_stats, dict) or not isinstance(index_stats, dict):
        raise RuntimeError("episodes_stats row is missing episode_index or index stats")

    for stat_name in LOCATION_STATS:
        if stat_name in episode_stats:
            episode_stats[stat_name] = [new_episode_index]
    episode_stats["std"] = [0.0]
    episode_stats["count"] = [frame_count]

    frame_offset = new_frame_start - old_frame_start
    for stat_name in LOCATION_STATS:
        if stat_name in index_stats:
            index_stats[stat_name] = shifted_stat_values(
                index_stats[stat_name], frame_offset
            )
    index_stats["count"] = [frame_count]
    return result


def validate_source(
    root: Path, *, skip_missing_episodes: bool = False
) -> dict[str, Any]:
    required = (
        root / "meta" / "info.json",
        root / "meta" / "episodes.jsonl",
        root / "meta" / "episodes_stats.jsonl",
        root / "meta" / "tasks.jsonl",
    )
    for path in required:
        if not path.is_file():
            raise RuntimeError(f"Missing required source file: {path}")

    info = read_json(root / "meta" / "info.json")
    if info.get("codebase_version") != "v2.1":
        raise RuntimeError(
            f"{root} uses {info.get('codebase_version')!r}, expected LeRobot v2.1"
        )

    episodes = sorted(
        read_jsonl(root / "meta" / "episodes.jsonl"),
        key=lambda row: int(row["episode_index"]),
    )
    episode_stats = sorted(
        read_jsonl(root / "meta" / "episodes_stats.jsonl"),
        key=lambda row: int(row["episode_index"]),
    )
    tasks = sorted(
        read_jsonl(root / "meta" / "tasks.jsonl"),
        key=lambda row: int(row["task_index"]),
    )

    expected_indices = list(range(len(episodes)))
    if [int(row["episode_index"]) for row in episodes] != expected_indices:
        raise RuntimeError(f"{root}: episodes.jsonl indices are not contiguous")
    if [int(row["episode_index"]) for row in episode_stats] != expected_indices:
        raise RuntimeError(f"{root}: episodes_stats.jsonl indices are not contiguous")
    if int(info.get("total_episodes", -1)) != len(episodes):
        raise RuntimeError(
            f"{root}: info.total_episodes does not match episodes.jsonl"
        )

    skipped_episode_indices: list[int] = []
    if skip_missing_episodes:
        chunks_size = int(info["chunks_size"])
        data_pattern = str(info["data_path"])
        video_pattern = str(info["video_path"])
        video_keys = sorted(
            key
            for key, feature in info["features"].items()
            if feature.get("dtype") == "video"
        )
        stats_by_episode = {
            int(row["episode_index"]): row for row in episode_stats
        }
        usable_episodes: list[dict[str, Any]] = []
        usable_episode_stats: list[dict[str, Any]] = []
        for episode_row in episodes:
            episode_index = int(episode_row["episode_index"])
            required_paths = [
                formatted_path(
                    root,
                    data_pattern,
                    episode_index=episode_index,
                    chunks_size=chunks_size,
                )
            ]
            required_paths.extend(
                formatted_path(
                    root,
                    video_pattern,
                    episode_index=episode_index,
                    chunks_size=chunks_size,
                    video_key=video_key,
                )
                for video_key in video_keys
            )
            if not all(path.is_file() for path in required_paths):
                skipped_episode_indices.append(episode_index)
                continue
            usable_episodes.append(episode_row)
            usable_episode_stats.append(stats_by_episode[episode_index])

        episodes = usable_episodes
        episode_stats = usable_episode_stats
        if skipped_episode_indices:
            print(
                f"Skipping incomplete episodes in {root}: "
                + ", ".join(map(str, skipped_episode_indices))
            )

    return {
        "root": root,
        "info": info,
        "episodes": episodes,
        "episode_stats": episode_stats,
        "tasks": tasks,
        "skipped_episode_indices": skipped_episode_indices,
    }


def validate_compatibility(sources: list[dict[str, Any]]) -> None:
    base = sources[0]
    for source in sources[1:]:
        for field in CONSISTENT_INFO_FIELDS:
            if source["info"].get(field) != base["info"].get(field):
                raise RuntimeError(
                    f"Incompatible {field}: {base['root']} != {source['root']}"
                )
        if source["tasks"] != base["tasks"]:
            raise RuntimeError(
                f"Incompatible tasks metadata: {base['root']} != {source['root']}"
            )


def merge_runs(
    source_roots: list[Path],
    output_root: Path,
    *,
    skip_missing_episodes: bool = False,
) -> None:
    if output_root.exists():
        raise RuntimeError(f"Refusing to overwrite existing output: {output_root}")
    if len(source_roots) < 2:
        raise RuntimeError("At least two source runs are required")

    sources = [
        validate_source(
            root.resolve(), skip_missing_episodes=skip_missing_episodes
        )
        for root in source_roots
    ]
    validate_compatibility(sources)

    base_info = sources[0]["info"]
    chunks_size = int(base_info["chunks_size"])
    data_pattern = str(base_info["data_path"])
    video_pattern = str(base_info["video_path"])
    video_keys = sorted(
        key
        for key, feature in base_info["features"].items()
        if feature.get("dtype") == "video"
    )

    output_root.mkdir(parents=True)
    merged_episodes: list[dict[str, Any]] = []
    merged_episode_stats: list[dict[str, Any]] = []
    reference_schema: pa.Schema | None = None
    global_episode_index = 0
    global_frame_index = 0
    copied_videos = 0

    for source in sources:
        source_root = source["root"]
        source_frame_count = 0
        for episode_row, stats_row in zip(
            source["episodes"], source["episode_stats"], strict=True
        ):
            old_episode_index = int(episode_row["episode_index"])
            src_parquet = formatted_path(
                source_root,
                data_pattern,
                episode_index=old_episode_index,
                chunks_size=chunks_size,
            )
            if not src_parquet.is_file():
                raise RuntimeError(f"Missing source parquet: {src_parquet}")

            table = pq.read_table(src_parquet)
            if reference_schema is None:
                reference_schema = table.schema.remove_metadata()
            elif not table.schema.remove_metadata().equals(reference_schema):
                raise RuntimeError(f"Parquet schema differs: {src_parquet}")

            frame_count = table.num_rows
            if frame_count != int(episode_row["length"]):
                raise RuntimeError(
                    f"{src_parquet}: rows={frame_count}, "
                    f"episodes.length={episode_row['length']}"
                )

            old_episode_values = table.column("episode_index").to_pylist()
            old_frame_values = table.column("frame_index").to_pylist()
            old_index_values = table.column("index").to_pylist()
            if set(old_episode_values) != {old_episode_index}:
                raise RuntimeError(f"{src_parquet}: bad episode_index column")
            if old_frame_values != list(range(frame_count)):
                raise RuntimeError(f"{src_parquet}: frame_index is not contiguous")
            expected_source_start = (
                old_index_values[0]
                if source["skipped_episode_indices"]
                else source_frame_count
            )
            if old_index_values != list(
                range(expected_source_start, expected_source_start + frame_count)
            ):
                raise RuntimeError(f"{src_parquet}: source index is not contiguous")

            table = replace_int64_column(
                table,
                "episode_index",
                np.full(frame_count, global_episode_index, dtype=np.int64),
            )
            table = replace_int64_column(
                table,
                "index",
                np.arange(
                    global_frame_index,
                    global_frame_index + frame_count,
                    dtype=np.int64,
                ),
            )

            dst_parquet = formatted_path(
                output_root,
                data_pattern,
                episode_index=global_episode_index,
                chunks_size=chunks_size,
            )
            dst_parquet.parent.mkdir(parents=True, exist_ok=True)
            tmp_parquet = dst_parquet.with_suffix(".parquet.tmp")
            pq.write_table(table, tmp_parquet, compression="snappy")
            tmp_parquet.replace(dst_parquet)

            output_episode = copy.deepcopy(episode_row)
            output_episode["episode_index"] = global_episode_index
            merged_episodes.append(output_episode)

            updated_stats = update_episode_stats(
                stats_row["stats"],
                new_episode_index=global_episode_index,
                new_frame_start=global_frame_index,
                old_frame_start=old_index_values[0],
                frame_count=frame_count,
            )
            merged_episode_stats.append(
                {
                    "episode_index": global_episode_index,
                    "stats": updated_stats,
                }
            )

            for video_key in video_keys:
                src_video = formatted_path(
                    source_root,
                    video_pattern,
                    episode_index=old_episode_index,
                    chunks_size=chunks_size,
                    video_key=video_key,
                )
                if not src_video.is_file():
                    raise RuntimeError(f"Missing source video: {src_video}")
                dst_video = formatted_path(
                    output_root,
                    video_pattern,
                    episode_index=global_episode_index,
                    chunks_size=chunks_size,
                    video_key=video_key,
                )
                dst_video.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_video, dst_video)
                copied_videos += 1

            source_frame_count += frame_count
            global_frame_index += frame_count
            global_episode_index += 1

        if (
            not source["skipped_episode_indices"]
            and source_frame_count != int(source["info"]["total_frames"])
        ):
            raise RuntimeError(
                f"{source_root}: observed {source_frame_count} frames, "
                f"expected {source['info']['total_frames']}"
            )

    merged_info = copy.deepcopy(base_info)
    merged_info.update(
        {
            "total_episodes": global_episode_index,
            "total_frames": global_frame_index,
            "total_tasks": len(sources[0]["tasks"]),
            "total_videos": len(video_keys),
            "total_chunks": math.ceil(global_episode_index / chunks_size),
            "splits": {"train": f"0:{global_episode_index}"},
        }
    )

    aggregate_input = [
        stats_to_numpy(row["stats"]) for row in merged_episode_stats
    ]
    merged_stats = aggregate_stats(aggregate_input)

    write_json(output_root / "meta" / "info.json", merged_info)
    write_json(output_root / "meta" / "stats.json", merged_stats)
    write_jsonl(output_root / "meta" / "episodes.jsonl", merged_episodes)
    write_jsonl(
        output_root / "meta" / "episodes_stats.jsonl", merged_episode_stats
    )
    write_jsonl(output_root / "meta" / "tasks.jsonl", sources[0]["tasks"])

    expected_videos = global_episode_index * len(video_keys)
    if copied_videos != expected_videos:
        raise RuntimeError(
            f"Copied {copied_videos} videos, expected {expected_videos}"
        )

    print(
        "Merged "
        f"{len(sources)} runs -> {output_root}: "
        f"episodes={global_episode_index}, frames={global_frame_index}, "
        f"videos={copied_videos}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        type=Path,
        help="Source LeRobot v2.1 run. Pass once per run.",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--skip-missing-episodes",
        action="store_true",
        help=(
            "Skip metadata episodes whose parquet or any expected video is "
            "missing, then reindex the remaining complete episodes."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    merge_runs(
        args.source,
        args.output.resolve(),
        skip_missing_episodes=args.skip_missing_episodes,
    )


if __name__ == "__main__":
    main()
