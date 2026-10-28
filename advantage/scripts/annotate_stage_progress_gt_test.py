import json
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts import annotate_stage_progress_gt as annotate


def test_compute_linear_progress_gt():
    np.testing.assert_allclose(annotate.compute_linear_progress_gt(5), [0.0, 0.25, 0.5, 0.75, 1.0])
    np.testing.assert_array_equal(annotate.compute_linear_progress_gt(1), [0.0])
    with pytest.raises(ValueError, match="positive"):
        annotate.compute_linear_progress_gt(0)


def test_update_metadata(tmp_path):
    meta_dir = tmp_path / "meta"
    meta_dir.mkdir()
    (meta_dir / "info.json").write_text(
        json.dumps(
            {
                "total_videos": 3,
                "features": {
                    "image": {"dtype": "video"},
                    "extra_view_image": {"dtype": "video"},
                    "extra_view_image-0": {"dtype": "video"},
                },
            }
        )
    )
    (meta_dir / "episodes_stats.jsonl").write_text(
        json.dumps({"episode_index": 0, "stats": {"extra_view_image-0": {"count": [3]}}}) + "\n"
    )
    (meta_dir / "stats.json").write_text(json.dumps({"extra_view_image-0": {"count": [3]}}))

    annotate.update_metadata(
        tmp_path,
        labels_by_episode={0: np.asarray([0.0, 0.5, 1.0], dtype=np.float32)},
        excluded_features=["extra_view_image-0"],
    )

    info = json.loads((meta_dir / "info.json").read_text())
    assert info["total_videos"] == 2
    assert "extra_view_image-0" not in info["features"]
    assert info["features"]["stage_progress_gt"]["dtype"] == "float32"

    episode_stats = json.loads((meta_dir / "episodes_stats.jsonl").read_text())
    assert "extra_view_image-0" not in episode_stats["stats"]
    assert episode_stats["stats"]["stage_progress_gt"]["min"] == [0.0]
    assert episode_stats["stats"]["stage_progress_gt"]["max"] == [1.0]

    stats = json.loads((meta_dir / "stats.json").read_text())
    assert "extra_view_image-0" not in stats
    assert stats["stage_progress_gt"]["count"] == [3]


@pytest.fixture
def labeled_dataset(tmp_path):
    meta_dir = tmp_path / "meta"
    meta_dir.mkdir()
    (meta_dir / "info.json").write_text(
        json.dumps({"features": {"stage_progress_gt": {"dtype": "float32", "shape": [1]}}})
    )
    (meta_dir / "episodes_stats.jsonl").write_text(
        "".join(json.dumps({"episode_index": index, "stats": {}}) + "\n" for index in range(2))
    )
    data_dir = tmp_path / "data/chunk-000"
    data_dir.mkdir(parents=True)
    for index, values in enumerate(([0.0, 0.8, 1.0], [0.0, 0.1, 0.2, 0.8, 1.0])):
        pq.write_table(
            pa.table(
                {
                    "frame_index": list(range(len(values))),
                    "episode_index": [index] * len(values),
                    "state": [[float(index), float(frame)] for frame in range(len(values))],
                    "stage_progress_gt": pa.array(values, type=pa.float32()),
                }
            ),
            data_dir / f"episode_{index:06d}.parquet",
        )
    # Existing annotations are unrelated to whole-episode GT construction.
    (tmp_path / "augmentation_metadata.json").write_text(
        json.dumps({"0": {"subtask_completion_indices": [1]}, "1": {"subtask_completion_indices": [3]}})
    )
    return tmp_path


def dataset_snapshot(dataset):
    return {str(path.relative_to(dataset)): path.read_bytes() for path in dataset.rglob("*") if path.is_file()}


def test_linear_cli_overwrites_each_episode_and_updates_metadata(labeled_dataset, monkeypatch):
    annotation_before = (labeled_dataset / "augmentation_metadata.json").read_bytes()
    original_tables = {
        path: pq.read_table(path).drop_columns(["stage_progress_gt"])
        for path in (labeled_dataset / "data").rglob("*.parquet")
    }
    monkeypatch.setattr(sys, "argv", ["annotate", "--dataset-dir", str(labeled_dataset), "--linear"])

    annotate.main()

    for path, original in original_tables.items():
        table = pq.read_table(path)
        values = table.column("stage_progress_gt").to_numpy()
        expected = [0.0, 0.5, 1.0] if table.num_rows == 3 else [0.0, 0.25, 0.5, 0.75, 1.0]
        np.testing.assert_array_equal(values, expected)
        assert table.schema.field("stage_progress_gt").type == pa.float32()
        assert table.drop_columns(["stage_progress_gt"]).equals(original)
    assert (labeled_dataset / "augmentation_metadata.json").read_bytes() == annotation_before
    info = json.loads((labeled_dataset / "meta/info.json").read_text())
    assert info["features"]["stage_progress_gt"] == {"dtype": "float32", "shape": [1], "names": ["stage_progress_gt"]}
    stats = json.loads((labeled_dataset / "meta/stats.json").read_text())["stage_progress_gt"]
    assert stats["count"] == [8]
    assert stats["mean"] == [0.5]
    episode_stats = [
        json.loads(line)["stats"]["stage_progress_gt"]
        for line in (labeled_dataset / "meta/episodes_stats.jsonl").read_text().splitlines()
    ]
    assert [record["count"] for record in episode_stats] == [[3], [5]]
    assert all(record["min"] == [0.0] and record["max"] == [1.0] for record in episode_stats)


def test_linear_cli_preview_leaves_dataset_unchanged(labeled_dataset, monkeypatch, capsys):
    before = dataset_snapshot(labeled_dataset)
    monkeypatch.setattr(sys, "argv", ["annotate", "--dataset-dir", str(labeled_dataset), "--linear", "--dry-run"])

    annotate.main()

    assert dataset_snapshot(labeled_dataset) == before
    output = capsys.readouterr().out
    assert "Episode 0: n_frames=3, gt_range=[0.0000, 1.0000]" in output
    assert "Episode 1: n_frames=5, gt_range=[0.0000, 1.0000]" in output


@pytest.mark.parametrize("removed_argument", ["--aug-meta", "--k-stages"])
def test_removed_gt_modes_are_rejected_before_writing(labeled_dataset, monkeypatch, capsys, removed_argument):
    before = dataset_snapshot(labeled_dataset)
    monkeypatch.setattr(
        sys,
        "argv",
        ["annotate", "--dataset-dir", str(labeled_dataset), "--linear", removed_argument, "unused"],
    )

    with pytest.raises(SystemExit) as error:
        annotate.main()

    assert error.value.code == 2
    assert f"unrecognized arguments: {removed_argument}" in capsys.readouterr().err
    assert dataset_snapshot(labeled_dataset) == before
