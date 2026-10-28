import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts import relabel_acbc_positive as relabeling

NEGATIVE_PROMPT = "new task, Advantage: negative"
POSITIVE_PROMPT = "new task, Advantage: positive"


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def _make_dataset(root: Path) -> Path:
    meta_root = root / "meta"
    data_root = root / "data/chunk-000"
    video_root = root / "videos/chunk-000/image"
    meta_root.mkdir(parents=True)
    data_root.mkdir(parents=True)
    video_root.mkdir(parents=True)
    info = {
        "chunks_size": 1000,
        "total_episodes": 2,
        "total_frames": 5,
        "total_tasks": 2,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    }
    (meta_root / "info.json").write_text(json.dumps(info), encoding="utf-8")
    _write_jsonl(
        meta_root / "tasks.jsonl",
        [
            {"task_index": 0, "task": NEGATIVE_PROMPT},
            {"task_index": 1, "task": POSITIVE_PROMPT},
        ],
    )
    _write_jsonl(
        meta_root / "episodes.jsonl",
        [
            {"episode_index": 0, "length": 3, "tasks": [NEGATIVE_PROMPT, POSITIVE_PROMPT]},
            {"episode_index": 1, "length": 2, "tasks": [NEGATIVE_PROMPT, POSITIVE_PROMPT]},
        ],
    )
    _write_jsonl(
        meta_root / "episodes_stats.jsonl",
        [
            {"episode_index": 0, "stats": {"state": {"count": [3]}, "task_index": {"mean": [0.0]}}},
            {"episode_index": 1, "stats": {"state": {"count": [2]}, "task_index": {"mean": [0.5]}}},
        ],
    )
    (meta_root / "stats.json").write_text(
        json.dumps({"state": {"count": [5]}, "task_index": {"mean": [0.4]}}),
        encoding="utf-8",
    )
    (meta_root / "mix_acbc_labeling.json").write_text(
        json.dumps(
            {
                "grouping": "whole_dataset",
                "group_summary": {"0": {"positive": 2, "positive_fraction": 0.4, "threshold": 0.25, "total": 5}},
                "threshold_semantics": "old threshold semantics",
            }
        ),
        encoding="utf-8",
    )
    task_indices = ([0, 1, 0], [1, 0])
    global_index = 0
    for episode_index, indices in enumerate(task_indices):
        table = pa.table(
            {
                "state": pa.array(np.arange(len(indices), dtype=np.float32)),
                "episode_index": pa.array([episode_index] * len(indices), type=pa.int64()),
                "frame_index": pa.array(range(len(indices)), type=pa.int64()),
                "index": pa.array(range(global_index, global_index + len(indices)), type=pa.int64()),
                "task_index": pa.array(indices, type=pa.int64()),
            }
        )
        pq.write_table(table, data_root / f"episode_{episode_index:06d}.parquet")
        (video_root / f"episode_{episode_index:06d}.mp4").write_bytes(f"video-{episode_index}".encode())
        global_index += len(indices)
    return root


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _make_raw_dataset(root: Path, *, prompts: list[str] | None = None, keep_task_index: bool = True) -> Path:
    dataset = _make_dataset(root)
    prompts = prompts or ["pick up the stamp and press it into the rectangular area on the paper"]
    _write_jsonl(
        dataset / "meta/tasks.jsonl",
        [{"task_index": index, "task": prompt} for index, prompt in enumerate(prompts)],
    )
    episodes = _read_jsonl(dataset / "meta/episodes.jsonl")
    for episode in episodes:
        episode["tasks"] = prompts
    _write_jsonl(dataset / "meta/episodes.jsonl", episodes)
    info_path = dataset / "meta/info.json"
    info = json.loads(info_path.read_text())
    info["total_tasks"] = len(prompts)
    info_path.write_text(json.dumps(info), encoding="utf-8")
    (dataset / "meta/mix_acbc_labeling.json").unlink()

    for parquet_path in sorted((dataset / "data").rglob("*.parquet")):
        table = pq.read_table(parquet_path)
        task_column_index = table.schema.get_field_index("task_index")
        if keep_task_index:
            raw_indices = pa.array(np.zeros(table.num_rows, dtype=np.int64), type=pa.int64())
            table = table.set_column(task_column_index, "task_index", raw_indices)
        else:
            table = table.remove_column(task_column_index)
        pq.write_table(table, parquet_path)
    return dataset


def test_relabel_dataset_copies_source_and_forces_positive_labels(tmp_path: Path):
    source = _make_dataset(tmp_path / "source")
    output = tmp_path / "output"

    summary = relabeling.relabel_dataset(source, output)

    source_episode_0 = pq.read_table(source / "data/chunk-000/episode_000000.parquet")
    output_episode_0 = pq.read_table(output / "data/chunk-000/episode_000000.parquet")
    output_episode_1 = pq.read_table(output / "data/chunk-000/episode_000001.parquet")
    assert source_episode_0["task_index"].to_pylist() == [0, 1, 0]
    assert output_episode_0["task_index"].to_pylist() == [1, 1, 0]
    assert output_episode_1["task_index"].to_pylist() == [1, 1]
    assert source_episode_0["state"].equals(output_episode_0["state"])

    assert (output / "meta/tasks.jsonl").read_bytes() == (source / "meta/tasks.jsonl").read_bytes()
    assert (output / "meta/info.json").read_bytes() == (source / "meta/info.json").read_bytes()
    episodes = _read_jsonl(output / "meta/episodes.jsonl")
    assert episodes[0]["tasks"] == [NEGATIVE_PROMPT, POSITIVE_PROMPT]
    assert episodes[1]["tasks"] == [POSITIVE_PROMPT]

    episode_stats = _read_jsonl(output / "meta/episodes_stats.jsonl")
    assert episode_stats[0]["stats"]["task_index"]["mean"] == [2 / 3]
    assert episode_stats[1]["stats"]["task_index"]["mean"] == [1.0]
    assert episode_stats[0]["stats"]["state"] == {"count": [3]}
    stats = json.loads((output / "meta/stats.json").read_text())
    assert stats["task_index"]["mean"] == [0.8]
    assert stats["state"] == {"count": [5]}

    manifest = json.loads((output / "meta/mix_acbc_labeling.json").read_text())
    assert manifest["group_summary"]["0"]["positive"] == 4
    assert manifest["group_summary"]["0"]["positive_fraction"] == 0.8
    assert manifest["group_summary"]["0"]["threshold"] == 0.25
    assert manifest["label_override"] == "all_positive_except_episode_000000_last_frame_negative"
    assert (output / "videos/chunk-000/image/episode_000000.mp4").read_bytes() == b"video-0"
    assert summary["negative"] == 1
    assert summary["positive"] == 4


def test_relabel_dataset_refuses_to_overwrite_output(tmp_path: Path):
    source = _make_dataset(tmp_path / "source")
    output = tmp_path / "output"
    output.mkdir()

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        relabeling.relabel_dataset(source, output)


def test_relabel_dataset_does_not_require_labeling_manifest(tmp_path: Path):
    source = _make_dataset(tmp_path / "arbitrarily_named_source")
    (source / "meta/mix_acbc_labeling.json").unlink()
    output = tmp_path / "another_arbitrary_name"

    summary = relabeling.relabel_dataset(source, output)

    assert summary["positive"] == 4
    assert not (output / "meta/mix_acbc_labeling.json").exists()


def test_relabel_raw_dataset_infers_prompt_and_adds_missing_task_index(tmp_path: Path):
    source = _make_raw_dataset(tmp_path / "raw_source", keep_task_index=False)
    (source / "meta/episodes_stats.jsonl").unlink()
    output = tmp_path / "output"

    summary = relabeling.relabel_dataset(source, output)

    assert summary["input_format"] == "raw_lerobot"
    assert summary["negative"] == 1
    assert summary["positive"] == 4
    assert "task_index" not in pq.read_schema(source / "data/chunk-000/episode_000000.parquet").names
    assert pq.read_table(output / "data/chunk-000/episode_000000.parquet")["task_index"].to_pylist() == [1, 1, 0]
    assert pq.read_table(output / "data/chunk-000/episode_000001.parquet")["task_index"].to_pylist() == [1, 1]

    expected_task = "pick up the stamp and press it into the rectangular area on the paper"
    assert _read_jsonl(output / "meta/tasks.jsonl") == [
        {"task_index": 0, "task": f"{expected_task}, Advantage: negative"},
        {"task_index": 1, "task": f"{expected_task}, Advantage: positive"},
    ]
    info = json.loads((output / "meta/info.json").read_text())
    assert info["total_tasks"] == 2
    assert info["features"]["task_index"] == {
        "dtype": "int64",
        "names": ["task_index"],
        "shape": [1],
    }
    assert not (output / "meta/episodes_stats.jsonl").exists()
    manifest = json.loads((output / "meta/mix_acbc_labeling.json").read_text())
    assert manifest["input_format"] == "raw_lerobot"
    assert manifest["group_summary"]["0"]["positive"] == 4


def test_relabel_raw_multitask_dataset_requires_explicit_prompt(tmp_path: Path):
    source = _make_raw_dataset(tmp_path / "raw_source", prompts=["task one", "task two"])
    failed_output = tmp_path / "failed_output"

    with pytest.raises(ValueError, match="pass --task-prompt explicitly"):
        relabeling.relabel_dataset(source, failed_output)
    assert not failed_output.exists()

    output = tmp_path / "output"
    summary = relabeling.relabel_dataset(source, output, task_prompt="combined task")

    assert summary["positive_prompt"] == "combined task, Advantage: positive"
    assert _read_jsonl(output / "meta/tasks.jsonl")[1] == {
        "task_index": 1,
        "task": "combined task, Advantage: positive",
    }
