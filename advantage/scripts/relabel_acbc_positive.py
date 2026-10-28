#!/usr/bin/env python3
"""Copy a LeRobot dataset and make it positive except for one anchor frame.

The negative anchor is the final frame of episode 0. Keeping this one frame
preserves both observed ACBC bins while all other frames use the positive
prompt. Inputs may already contain binary ACBC labels or may be an unlabeled
LeRobot dataset. For an unlabeled input, the task prompt is inferred from
``meta/tasks.jsonl`` or supplied with ``--task-prompt``.

The source dataset is never modified and the output must not exist.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

_ADVANTAGE_PROMPT = re.compile(r"^(?P<task>.+), Advantage: (?P<label>negative|positive)$")
_LABEL_OVERRIDE = "all_positive_except_episode_000000_last_frame_negative"
_THRESHOLD_SEMANTICS = (
    "all dataset frames forced positive except episode_000000 final frame forced negative; "
    "stored threshold retained for provenance but not applied"
)
_RAW_INPUT_FORMAT = "raw_lerobot"
_ACBC_INPUT_FORMAT = "binary_acbc"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not all(isinstance(record, dict) for record in records):
        raise ValueError(f"Expected JSON objects in {path}")
    return records


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary_path, path)


def _atomic_write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    os.replace(temporary_path, path)


def _atomic_write_parquet(path: Path, table: pa.Table) -> None:
    temporary_path = path.with_name(f".{path.name}.tmp")
    pq.write_table(table, temporary_path)
    os.replace(temporary_path, path)


def _compute_scalar_stats(values: np.ndarray) -> dict[str, list[float] | list[int]]:
    values = np.asarray(values)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("task_index statistics require a non-empty 1D array")
    return {
        "min": [float(np.min(values))],
        "max": [float(np.max(values))],
        "mean": [float(np.mean(values))],
        "std": [float(np.std(values))],
        "count": [len(values)],
        "q01": [float(np.quantile(values, 0.01))],
        "q10": [float(np.quantile(values, 0.10))],
        "q50": [float(np.quantile(values, 0.50))],
        "q90": [float(np.quantile(values, 0.90))],
        "q99": [float(np.quantile(values, 0.99))],
    }


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _validate_paths(source: Path, output: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(f"Source dataset does not exist: {source}")
    if output.exists():
        raise FileExistsError(f"Output already exists; refusing to overwrite it: {output}")
    if source == output or _is_within(output, source) or _is_within(source, output):
        raise ValueError(f"Source and output datasets must not overlap: source={source}, output={output}")


def _validate_task_prompt(task_prompt: str) -> str:
    prompt = task_prompt.strip()
    if not prompt or "\n" in prompt or "\r" in prompt:
        raise ValueError("Task prompt must be a non-empty, single-line string")
    if _ADVANTAGE_PROMPT.fullmatch(prompt) is not None or ", Advantage:" in prompt:
        raise ValueError("Task prompt must not include an Advantage label suffix")
    return prompt


def _load_or_create_prompt_mapping(
    dataset_root: Path,
    task_prompt: str | None,
) -> tuple[int, str, int, str, str]:
    tasks_path = dataset_root / "meta/tasks.jsonl"
    tasks = _read_jsonl(tasks_path)
    if not tasks:
        raise ValueError(f"Expected at least one task mapping in {tasks_path}")

    prompts: dict[str, tuple[int, str, str]] = {}
    raw_prompts: set[str] = set()
    contains_advantage_prompt = False
    for record in tasks:
        try:
            task_index = int(record["task_index"])
            prompt = record["task"]
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid task mapping in {tasks_path}: {record!r}") from error
        if not isinstance(prompt, str):
            raise ValueError(f"Task prompt must be a string in {tasks_path}: {prompt!r}")
        prompt = prompt.strip()
        if not prompt or "\n" in prompt or "\r" in prompt:
            raise ValueError(f"Task prompt must be a non-empty, single-line string in {tasks_path}: {prompt!r}")

        match = _ADVANTAGE_PROMPT.fullmatch(prompt)
        if match is None:
            if ", Advantage:" in prompt:
                raise ValueError(f"Malformed Advantage prompt in {tasks_path}: {prompt!r}")
            raw_prompts.add(prompt)
            continue

        contains_advantage_prompt = True
        label = match.group("label")
        if label in prompts:
            raise ValueError(f"Expected exactly one {label} prompt in {tasks_path}")
        prompts[label] = (task_index, prompt, match.group("task"))
        raw_prompts.add(match.group("task"))

    if len(tasks) == 2 and set(prompts) == {"negative", "positive"}:
        negative_index, negative_prompt, negative_task = prompts["negative"]
        positive_index, positive_prompt, positive_task = prompts["positive"]
        if negative_task != positive_task:
            raise ValueError("Negative and positive prompts must describe the same task")
        if task_prompt is not None and _validate_task_prompt(task_prompt) != negative_task:
            raise ValueError(
                f"--task-prompt does not match the task already recorded in the binary ACBC dataset: {negative_task!r}"
            )
        return negative_index, negative_prompt, positive_index, positive_prompt, _ACBC_INPUT_FORMAT

    if contains_advantage_prompt and prompts and len(raw_prompts) != 1 and task_prompt is None:
        raise ValueError(
            f"Cannot infer one base task from incomplete/mixed Advantage mappings in {tasks_path}; "
            "pass --task-prompt explicitly"
        )

    if task_prompt is not None:
        base_prompt = _validate_task_prompt(task_prompt)
    elif len(raw_prompts) == 1:
        base_prompt = _validate_task_prompt(next(iter(raw_prompts)))
    else:
        raise ValueError(
            f"Cannot infer one task prompt from {len(raw_prompts)} tasks in {tasks_path}; pass --task-prompt explicitly"
        )

    negative_index = 0
    positive_index = 1
    negative_prompt = f"{base_prompt}, Advantage: negative"
    positive_prompt = f"{base_prompt}, Advantage: positive"
    _atomic_write_jsonl(
        tasks_path,
        [
            {"task_index": negative_index, "task": negative_prompt},
            {"task_index": positive_index, "task": positive_prompt},
        ],
    )
    return negative_index, negative_prompt, positive_index, positive_prompt, _RAW_INPUT_FORMAT


def _parquet_path(dataset_root: Path, info: dict[str, Any], episode_index: int) -> Path:
    chunk_size = int(info.get("chunks_size", 1000))
    template = info.get("data_path")
    if not isinstance(template, str) or not template:
        raise ValueError("meta/info.json must contain a data_path template")
    chunk_index = episode_index // chunk_size
    try:
        relative_path = Path(
            template.format(
                episode_chunk=chunk_index,
                chunk_index=chunk_index,
                episode_index=episode_index,
            )
        )
    except (KeyError, ValueError) as error:
        raise ValueError(f"Unsupported data_path template in meta/info.json: {template!r}") from error
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"Unsafe parquet path in meta/info.json: {relative_path}")
    return dataset_root / relative_path


def _relabel_in_place(dataset_root: Path, task_prompt: str | None = None) -> dict[str, Any]:
    meta_root = dataset_root / "meta"
    info = _read_json(meta_root / "info.json")
    total_episodes = int(info["total_episodes"])
    expected_frames = int(info["total_frames"])
    if total_episodes <= 0 or expected_frames <= 0:
        raise ValueError("Dataset must contain at least one episode and one frame")

    negative_index, negative_prompt, positive_index, positive_prompt, input_format = _load_or_create_prompt_mapping(
        dataset_root,
        task_prompt,
    )
    if negative_index == positive_index:
        raise ValueError("Negative and positive prompts must use different task_index values")

    episodes_path = meta_root / "episodes.jsonl"
    episode_records = _read_jsonl(episodes_path)
    episode_by_index = {int(record["episode_index"]): record for record in episode_records}
    if set(episode_by_index) != set(range(total_episodes)) or len(episode_records) != total_episodes:
        raise ValueError("meta/episodes.jsonl must contain contiguous, unique episode indices")

    all_task_indices: list[np.ndarray] = []
    task_indices_by_episode: dict[int, np.ndarray] = {}
    allowed_indices = {negative_index, positive_index}
    for episode_index in range(total_episodes):
        parquet_path = _parquet_path(dataset_root, info, episode_index)
        if not parquet_path.is_file():
            raise FileNotFoundError(f"Missing episode parquet: {parquet_path}")
        table = pq.read_table(parquet_path)
        task_column_index = table.schema.get_field_index("task_index")
        if table.num_rows <= 0:
            raise ValueError(f"Episode contains no frames: {parquet_path}")
        if task_column_index < 0:
            if input_format == _ACBC_INPUT_FORMAT:
                raise ValueError(f"Missing task_index column: {parquet_path}")
            task_type = pa.int64()
        else:
            task_type = table.schema.field(task_column_index).type
            if not pa.types.is_integer(task_type):
                raise ValueError(f"task_index must be integer in {parquet_path}, got {task_type}")
            if input_format == _ACBC_INPUT_FORMAT:
                existing_indices = {int(value) for value in table["task_index"].to_pylist()}
                if not existing_indices <= allowed_indices:
                    raise ValueError(
                        f"Unexpected task_index values in {parquet_path}: {sorted(existing_indices - allowed_indices)}"
                    )

        task_indices = np.full(table.num_rows, positive_index, dtype=np.int64)
        if episode_index == 0:
            task_indices[-1] = negative_index
        if input_format == _RAW_INPUT_FORMAT:
            task_type = pa.int64()
        task_array = pa.array(task_indices, type=task_type)
        if task_column_index < 0:
            rewritten = table.append_column("task_index", task_array)
        else:
            rewritten = table.set_column(task_column_index, "task_index", task_array)
        _atomic_write_parquet(parquet_path, rewritten)
        task_indices_by_episode[episode_index] = task_indices
        all_task_indices.append(task_indices)

        episode_by_index[episode_index]["tasks"] = (
            [negative_prompt, positive_prompt] if episode_index == 0 else [positive_prompt]
        )

    total_frames = sum(len(values) for values in all_task_indices)
    if total_frames != expected_frames:
        raise ValueError(f"meta/info.json reports {expected_frames} frames but parquet data contains {total_frames}")
    _atomic_write_jsonl(episodes_path, episode_records)

    if input_format == _RAW_INPUT_FORMAT:
        info["total_tasks"] = 2
        info.setdefault("features", {})["task_index"] = {
            "dtype": "int64",
            "shape": [1],
            "names": ["task_index"],
        }
        _atomic_write_json(meta_root / "info.json", info)

    episode_stats_path = meta_root / "episodes_stats.jsonl"
    if episode_stats_path.is_file():
        episode_stats = _read_jsonl(episode_stats_path)
        stats_by_episode = {int(record["episode_index"]): record for record in episode_stats}
        if set(stats_by_episode) != set(range(total_episodes)) or len(episode_stats) != total_episodes:
            raise ValueError("meta/episodes_stats.jsonl must contain contiguous, unique episode indices")
        for episode_index, values in task_indices_by_episode.items():
            stats_by_episode[episode_index].setdefault("stats", {})["task_index"] = _compute_scalar_stats(values)
        _atomic_write_jsonl(episode_stats_path, episode_stats)

    stats_path = meta_root / "stats.json"
    stats = _read_json(stats_path) if stats_path.is_file() else {}
    combined_indices = np.concatenate(all_task_indices)
    stats["task_index"] = _compute_scalar_stats(combined_indices)
    _atomic_write_json(stats_path, stats)

    manifest_path = meta_root / "mix_acbc_labeling.json"
    positive_count = int(np.sum(combined_indices == positive_index))
    if manifest_path.is_file():
        manifest = _read_json(manifest_path)
        group_summary = manifest.get("group_summary")
        if group_summary is not None:
            if not isinstance(group_summary, dict) or len(group_summary) != 1:
                raise ValueError(
                    "Cannot safely update a multi-group mix_acbc_labeling.json; "
                    "the positive override currently supports one aggregate group"
                )
            group_record = next(iter(group_summary.values()))
            if not isinstance(group_record, dict):
                raise ValueError("Invalid group_summary record in mix_acbc_labeling.json")
            group_record["total"] = total_frames
            group_record["positive"] = positive_count
            group_record["positive_fraction"] = positive_count / total_frames
        manifest["label_override"] = _LABEL_OVERRIDE
        manifest["threshold_semantics"] = _THRESHOLD_SEMANTICS
        manifest["input_format"] = input_format
        _atomic_write_json(manifest_path, manifest)
    elif input_format == _RAW_INPUT_FORMAT:
        _atomic_write_json(
            manifest_path,
            {
                "advantage_source": None,
                "group_count": 1,
                "group_summary": {
                    "0": {
                        "positive": positive_count,
                        "positive_fraction": positive_count / total_frames,
                        "threshold": None,
                        "total": total_frames,
                    }
                },
                "grouping": "whole_dataset",
                "input_format": input_format,
                "label_override": _LABEL_OVERRIDE,
                "threshold_semantics": _THRESHOLD_SEMANTICS,
            },
        )

    return {
        "input_format": input_format,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "negative": total_frames - positive_count,
        "positive": positive_count,
        "negative_prompt": negative_prompt,
        "positive_prompt": positive_prompt,
    }


def relabel_dataset(
    source_dataset: Path,
    output_dataset: Path,
    *,
    task_prompt: str | None = None,
) -> dict[str, Any]:
    """Copy and relabel one dataset without modifying the source."""
    source = source_dataset.expanduser().resolve()
    output = output_dataset.expanduser().resolve()
    _validate_paths(source, output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_name(f".{output.name}.relabeling.tmp")
    if temporary_output.exists():
        raise FileExistsError(f"Temporary output already exists: {temporary_output}")

    try:
        shutil.copytree(source, temporary_output, symlinks=True)
        summary = _relabel_in_place(temporary_output, task_prompt=task_prompt)
        os.replace(temporary_output, output)
    except Exception:
        if temporary_output.exists():
            shutil.rmtree(temporary_output)
        raise
    return {**summary, "source_dataset": str(source), "output_dataset": str(output)}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dataset", type=Path, help="LeRobot dataset to copy; binary ACBC labels are optional.")
    parser.add_argument("output_dataset", type=Path, help="New output path; it must not already exist.")
    parser.add_argument(
        "--task-prompt",
        help="Base task prompt for an unlabeled/multi-task input; inferred from meta/tasks.jsonl when unambiguous.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = relabel_dataset(args.source_dataset, args.output_dataset, task_prompt=args.task_prompt)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
