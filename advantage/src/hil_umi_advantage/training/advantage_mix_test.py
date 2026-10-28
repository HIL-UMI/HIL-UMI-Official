# Modified for HIL-UMI.
import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest

from hil_umi_advantage.training import config as training_config
from hil_umi_advantage.training import data_loader
from hil_umi_advantage.training.advantage_dataset import BASE_UMI_SCHEMA
from hil_umi_advantage.training.advantage_dataset import STAGE_UMI_SCHEMA
from hil_umi_advantage.training.advantage_dataset import CanonicalAdvantageDataset
from hil_umi_advantage.training.advantage_dataset import canonicalize_masked_state
from hil_umi_advantage.training.advantage_dataset import detect_advantage_dataset_schema


class _ListDataset:
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def _raw_sample(schema, marker: float) -> dict:
    return {
        schema.front_key: np.full((3, 4, 5), marker, dtype=np.float32),
        schema.wrist_key: np.full((3, 4, 5), marker + 1, dtype=np.float32),
        f"his_-100_{schema.front_key}": np.full((3, 4, 5), marker + 2, dtype=np.float32),
        f"his_-100_{schema.wrist_key}": np.full((3, 4, 5), marker + 3, dtype=np.float32),
        schema.state_key: np.arange(47, dtype=np.float32),
        schema.action_key: np.full((50, 30), marker, dtype=np.float32),
        "episode_length": np.int64(100),
        "frame_index": np.int64(4),
        "episode_index": np.int64(2),
        "progress": np.float32(0.25),
        "task_index": np.int64(0),
    }


@pytest.mark.parametrize("raw_state_dim", [10, 47])
def test_canonicalize_masked_state_ignores_raw_schema_width(raw_state_dim):
    raw_state = np.arange(raw_state_dim, dtype=np.float64)

    state = canonicalize_masked_state(raw_state, model_state_dim=32)

    np.testing.assert_array_equal(state, np.zeros(32, dtype=np.float32))
    assert state.dtype == np.float32


def test_canonicalize_masked_state_requires_positive_model_width():
    with pytest.raises(ValueError, match="must be positive"):
        canonicalize_masked_state(np.zeros(10, dtype=np.float32), model_state_dim=0)


@pytest.mark.parametrize("schema", [BASE_UMI_SCHEMA, STAGE_UMI_SCHEMA])
def test_detect_and_canonicalize_supported_schemas(schema):
    features = {
        schema.front_key: {},
        schema.wrist_key: {},
        schema.state_key: {},
        schema.action_key: {},
        "stage_progress_gt": {},
    }
    assert detect_advantage_dataset_schema(features) is schema

    dataset = CanonicalAdvantageDataset(_ListDataset([_raw_sample(schema, 3.0)]), schema, state_dim=32)
    sample = dataset[0]
    np.testing.assert_array_equal(sample["extra_view_image"], np.full((3, 4, 5), 3.0, dtype=np.float32))
    np.testing.assert_array_equal(sample["image"], np.full((3, 4, 5), 4.0, dtype=np.float32))
    np.testing.assert_array_equal(sample["state"], np.zeros(32, dtype=np.float32))
    assert sample["actions"].shape == (50, 30)
    assert sample["progress"] == np.float32(0.25)


def test_detect_schema_rejects_unknown_features():
    with pytest.raises(ValueError, match="Unsupported advantage dataset schema"):
        detect_advantage_dataset_schema({"stage_progress_gt": {}})


def test_mixed_dataset_records_group_lengths():
    mixed = data_loader.MixedAdvantageDataset(
        [_ListDataset([0, 1]), _ListDataset([2, 3, 4])],
        [_ListDataset([5])],
    )
    assert mixed.group_lengths == (5, 1)
    assert len(mixed) == 6
    assert [mixed[index] for index in range(6)] == list(range(6))


def test_two_group_sampler_has_equal_group_weight_and_is_reproducible():
    group_lengths = (1000, 100)
    sampler = data_loader.DistributedTwoGroupSampler(group_lengths, seed=17)
    assert sampler.weights[:1000].sum().item() == pytest.approx(0.5)
    assert sampler.weights[1000:].sum().item() == pytest.approx(0.5)
    assert sampler.weights[0].item() == pytest.approx(0.5 / 1000)
    assert sampler.weights[-1].item() == pytest.approx(0.5 / 100)

    sampled_indices = [index for _ in range(20) for index in sampler]
    fraction_from_group_a = sum(index < group_lengths[0] for index in sampled_indices) / len(sampled_indices)
    assert fraction_from_group_a == pytest.approx(0.5, abs=0.02)

    first = data_loader.DistributedTwoGroupSampler(group_lengths, seed=23)
    second = data_loader.DistributedTwoGroupSampler(group_lengths, seed=23)
    assert list(first) == list(second)
    assert list(first) == list(second)


def test_two_group_sampler_ddp_ranks_partition_one_global_draw():
    group_lengths = (4, 4)
    global_sampler = data_loader.DistributedTwoGroupSampler(group_lengths, seed=5)
    rank_zero = data_loader.DistributedTwoGroupSampler(group_lengths, num_replicas=2, rank=0, seed=5)
    rank_one = data_loader.DistributedTwoGroupSampler(group_lengths, num_replicas=2, rank=1, seed=5)

    rank_zero_indices = list(rank_zero)
    rank_one_indices = list(rank_one)
    interleaved = [index for pair in zip(rank_zero_indices, rank_one_indices, strict=True) for index in pair]
    assert interleaved == list(global_sampler)


def test_mix_requires_both_groups():
    config = SimpleNamespace(split="all", advantage_dataset_group_a=("/dataset/a",), advantage_dataset_group_b=())
    data_config = SimpleNamespace(repo_id="fake")
    with pytest.raises(ValueError, match="requires both"):
        data_loader.create_advantage_torch_dataset(data_config, action_horizon=1, model_config=None, config=config)


def test_train_config_validates_mix_groups_before_training():
    advantage_config = training_config.get_config("ADVANTAGE_TORCH_TOWEL_FOLD")
    with pytest.raises(ValueError, match="requires both"):
        dataclasses.replace(advantage_config, advantage_dataset_group_a=("/dataset/a",))

    policy_config = training_config.get_config("pi0_aloha")
    with pytest.raises(ValueError, match="only be used"):
        dataclasses.replace(
            policy_config,
            advantage_dataset_group_a=("/dataset/a",),
            advantage_dataset_group_b=("/dataset/b",),
        )
