import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts import organize_fold_towel_datasets as organization

VIDEO_KEYS = ("observation.images.cam_front", "observation.images.cam_left")


def _scalar_stats(values: np.ndarray) -> dict:
    return organization.compute_array_stats(np.asarray(values))


def _write_source_run(root: Path, lengths: list[int], marker: bytes) -> None:
    features = {
        "observation.state": {"dtype": "float32", "shape": [2], "names": ["x", "y"]},
        "action": {"dtype": "float32", "shape": [2], "names": ["x", "y"]},
        **{key: {"dtype": "video", "shape": [4, 4, 3], "names": ["height", "width", "channels"]} for key in VIDEO_KEYS},
        "timestamp": {"dtype": "float32", "shape": [1], "names": ["timestamp"]},
        "frame_index": {"dtype": "int64", "shape": [1], "names": ["frame_index"]},
        "episode_index": {"dtype": "int64", "shape": [1], "names": ["episode_index"]},
        "index": {"dtype": "int64", "shape": [1], "names": ["index"]},
        "task_index": {"dtype": "int64", "shape": [1], "names": ["task_index"]},
    }
    info = {
        "codebase_version": "v2.1",
        "robot_type": "test_robot",
        "total_episodes": len(lengths),
        "total_frames": sum(lengths),
        "total_tasks": 1,
        "total_videos": len(VIDEO_KEYS),
        "total_chunks": 1,
        "chunks_size": 1000,
        "data_files_size_in_mb": 100,
        "video_files_size_in_mb": 200,
        "fps": 30,
        "splits": {"train": f"0:{len(lengths)}"},
        "data_path": "data/chunk-{chunk_index:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{chunk_index:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }
    (root / "meta").mkdir(parents=True)
    (root / "meta/info.json").write_text(json.dumps(info))
    (root / "meta/tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": "teleoperation"}) + "\n")
    episodes = []
    episodes_stats = []
    global_index = 0
    for episode_index, length in enumerate(lengths):
        state = np.arange(length * 2, dtype=np.float32).reshape(length, 2)
        table = pa.table(
            {
                "observation.state": pa.array(state.tolist(), type=pa.list_(pa.float32(), 2)),
                "action": pa.array((state + 1).tolist(), type=pa.list_(pa.float32(), 2)),
                "timestamp": pa.array(np.arange(length, dtype=np.float32) / 30),
                "frame_index": pa.array(np.arange(length), type=pa.int64()),
                "episode_index": pa.array(np.full(length, episode_index), type=pa.int64()),
                "index": pa.array(np.arange(global_index, global_index + length), type=pa.int64()),
                "task_index": pa.array(np.zeros(length), type=pa.int64()),
            }
        ).replace_schema_metadata({b"huggingface": b'{"info":{"features":{"state":{"_type":"List"}}}}'})
        parquet_path = root / f"data/chunk-000/episode_{episode_index:06d}.parquet"
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, parquet_path)
        for video_key in VIDEO_KEYS:
            video_path = root / f"videos/chunk-000/{video_key}/episode_{episode_index:06d}.mp4"
            video_path.parent.mkdir(parents=True, exist_ok=True)
            video_path.write_bytes(marker + bytes([episode_index]))
        stats = {name: _scalar_stats(np.asarray(table[name].to_pylist())) for name in table.column_names}
        episodes.append({"episode_index": episode_index, "tasks": ["teleoperation"], "length": length})
        episodes_stats.append({"episode_index": episode_index, "stats": stats})
        global_index += length
    (root / "meta/episodes.jsonl").write_text("".join(json.dumps(record) + "\n" for record in episodes))
    (root / "meta/episodes_stats.jsonl").write_text("".join(json.dumps(record) + "\n" for record in episodes_stats))
    (root / "meta/stats.json").write_text("{}")


def test_format_dataset_path_accepts_both_chunk_placeholders():
    old = "data/chunk-{chunk_index:03d}/episode_{episode_index:06d}.parquet"
    canonical = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
    assert organization.format_dataset_path(old, chunk_index=2, episode_index=7) == Path(
        "data/chunk-002/episode_000007.parquet"
    )
    assert organization.format_dataset_path(canonical, chunk_index=2, episode_index=7) == Path(
        "data/chunk-002/episode_000007.parquet"
    )


def test_merge_stage1_reindexes_and_preserves_sources(tmp_path: Path):
    source_root = tmp_path / "sources"
    run_a = source_root / "run_20260101_000000Z"
    run_b = source_root / "run_20260101_010000Z"
    _write_source_run(run_a, [2, 3], b"a")
    _write_source_run(run_b, [4], b"b")
    before = {run: organization.inventory_tree(run) for run in (run_a, run_b)}

    preflight = organization.preflight_stage1_sources(source_root, verify_video_frames=False)
    target = tmp_path / "merged"
    organization.merge_stage1_sources(
        preflight,
        target,
        task_prompt="fold the towel",
        source_inventories=before,
    )

    info = json.loads((target / "meta/info.json").read_text())
    assert info["total_episodes"] == 3
    assert info["total_frames"] == 9
    assert info["total_videos"] == 6
    assert "{episode_chunk:03d}" in info["data_path"]
    assert pq.read_schema(target / "data/chunk-000/episode_000000.parquet").metadata is None
    episode_two = pq.read_table(target / "data/chunk-000/episode_000002.parquet")
    assert episode_two["episode_index"].to_pylist() == [2] * 4
    assert episode_two["index"].to_pylist() == [5, 6, 7, 8]
    assert episode_two["task_index"].to_pylist() == [0] * 4
    tasks = [json.loads(line) for line in (target / "meta/tasks.jsonl").read_text().splitlines()]
    assert tasks == [{"task_index": 0, "task": "fold the towel"}]
    manifest = json.loads((target / "meta/merge_manifest.json").read_text())
    assert [(record["source_run"], record["source_episode_index"]) for record in manifest["episodes"]] == [
        (run_a.name, 0),
        (run_a.name, 1),
        (run_b.name, 0),
    ]
    assert {run: organization.inventory_tree(run).entries for run in before} == {
        run: inventory.entries for run, inventory in before.items()
    }


def test_merge_refuses_existing_target(tmp_path: Path):
    source_root = tmp_path / "sources"
    run = source_root / "run_20260101_000000Z"
    _write_source_run(run, [2], b"a")
    preflight = organization.preflight_stage1_sources(source_root, verify_video_frames=False)
    target = tmp_path / "merged"
    target.mkdir()

    with pytest.raises(FileExistsError, match="existing target"):
        organization.merge_stage1_sources(
            preflight,
            target,
            task_prompt="fold the towel",
            source_inventories={run: organization.inventory_tree(run)},
        )
