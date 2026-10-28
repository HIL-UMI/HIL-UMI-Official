import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from scripts import annotate_subtask_boundaries_web as annotation_web


def _make_dataset(dataset_dir: Path, episode_lengths: list[int]) -> None:
    meta_dir = dataset_dir / "meta"
    meta_dir.mkdir(parents=True)
    info = {
        "total_episodes": len(episode_lengths),
        "chunks_size": 1000,
        "fps": 30,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    }
    (meta_dir / "info.json").write_text(json.dumps(info))
    (meta_dir / "episodes.jsonl").write_text(
        "".join(
            json.dumps({"episode_index": index, "length": length, "tasks": ["fold towel"]}) + "\n"
            for index, length in enumerate(episode_lengths)
        )
    )
    for episode_index, length in enumerate(episode_lengths):
        parquet_path = dataset_dir / f"data/chunk-000/episode_{episode_index:06d}.parquet"
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table({"frame_index": pa.array(range(length), type=pa.int64())}), parquet_path)
        for video_key in ("extra_view_image", "image"):
            video_path = dataset_dir / f"videos/chunk-000/{video_key}/episode_{episode_index:06d}.mp4"
            video_path.parent.mkdir(parents=True, exist_ok=True)
            video_path.touch()


def test_validate_boundaries():
    assert annotation_web.validate_boundaries([2, 4, 7], 10) == [2, 4, 7]

    for invalid in ([0, 4, 7], [2, 2, 7], [2, 8, 7], [2, 4, 10], [2, 4], [2, 4, 7.0]):
        try:
            annotation_web.validate_boundaries(invalid, 10)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected invalid boundaries to fail: {invalid}")


def test_api_saves_resumes_and_deletes_annotations(tmp_path: Path):
    dataset_dir = tmp_path / "dataset"
    _make_dataset(dataset_dir, [10, 12])
    annotation_path = dataset_dir / "augmentation_metadata.json"
    app = annotation_web.create_app(dataset_dir, annotation_path)
    client = app.test_client()

    episodes = client.get("/api/episodes").get_json()
    assert [episode["num_frames"] for episode in episodes] == [10, 12]
    assert not any(episode["completed"] for episode in episodes)
    assert episodes[0]["front_video_url"] == "media/0/front"

    invalid_response = client.put("/api/episodes/0/boundaries", json={"boundaries": [0, 4, 7]})
    assert invalid_response.status_code == 400

    response = client.put("/api/episodes/0/boundaries", json={"boundaries": [2, 4, 7]})
    assert response.status_code == 200
    assert response.get_json()["completed"]
    assert json.loads(annotation_path.read_text()) == {"0": {"segments": [], "subtask_completion_indices": [2, 4, 7]}}

    resumed = annotation_web.create_app(dataset_dir, annotation_path).test_client()
    assert resumed.get("/api/episodes/0").get_json()["boundaries"] == [2, 4, 7]
    assert resumed.delete("/api/episodes/0/boundaries").status_code == 200
    assert json.loads(annotation_path.read_text()) == {}


def test_media_rejects_unknown_camera(tmp_path: Path):
    dataset_dir = tmp_path / "dataset"
    _make_dataset(dataset_dir, [10])
    client = annotation_web.create_app(dataset_dir).test_client()

    assert client.get("/media/0/unknown").status_code == 404
