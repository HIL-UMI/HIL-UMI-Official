from types import SimpleNamespace

import scripts.train_pytorch as train_pytorch


def test_prune_old_checkpoints_keeps_latest_and_periodic(tmp_path):
    for name in ["5000", "10000", "15000", "20000", "25000", "tmp_30000", "notes"]:
        (tmp_path / name).mkdir()

    train_pytorch.prune_old_checkpoints(tmp_path, latest_step=25000, keep_period=10000)

    assert {path.name for path in tmp_path.iterdir()} == {"10000", "20000", "25000", "tmp_30000", "notes"}


def test_prune_old_checkpoints_without_period_keeps_every_checkpoint(tmp_path):
    for step in [1, 2, 3]:
        (tmp_path / str(step)).mkdir()

    train_pytorch.prune_old_checkpoints(tmp_path, latest_step=3, keep_period=None)

    assert {path.name for path in tmp_path.iterdir()} == {"1", "2", "3"}


def test_should_save_checkpoint_uses_interval_and_exact_final_step():
    saved_steps = [step for step in range(1, 10_001) if train_pytorch.should_save_checkpoint(step, 2_500, 10_000)]

    assert saved_steps == [2_500, 5_000, 7_500, 10_000]
    assert not train_pytorch.should_save_checkpoint(9_999, 2_500, 10_000)
    assert train_pytorch.should_save_checkpoint(10_001, 2_500, 10_001)


def test_save_checkpoint_does_not_overwrite_existing_step(tmp_path):
    checkpoint_dir = tmp_path / "2500"
    checkpoint_dir.mkdir()
    sentinel = checkpoint_dir / "sentinel.txt"
    sentinel.write_text("original")
    config = SimpleNamespace(checkpoint_dir=tmp_path, save_interval=2_500, num_train_steps=10_000)

    train_pytorch.save_checkpoint(
        model=None,
        optimizer=None,
        global_step=2_500,
        config=config,
        is_main=True,
        data_config=None,
    )

    assert sentinel.read_text() == "original"
