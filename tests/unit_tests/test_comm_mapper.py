# Copyright 2026 The RLinf Authors.
# Modified for HIL-UMI.
# SPDX-License-Identifier: Apache-2.0

import torch

from training.data.embodied_io_struct import EnvOutput
from training.scheduler import (
    build_recv_plan,
    build_route_channel_key,
    build_send_plan,
    merge_batches,
    split_batch,
)


def _make_obs(start: int, batch_size: int) -> dict:
    return {
        "states": torch.arange(start, start + batch_size * 2, dtype=torch.float32).view(
            batch_size, 2
        ),
        "main_images": None,
        "wrist_images": None,
        "extra_view_images": None,
        "task_descriptions": [
            f"task-{idx}" for idx in range(start, start + batch_size)
        ],
    }


def test_build_send_plan_load_balance_env_to_rollout():
    plan = build_send_plan(
        src_group_name="env",
        dst_group_name="rollout",
        src_rank=0,
        src_world_size=2,
        dst_world_size=3,
        tag="train_obs",
        batch_size=12,
    )
    assert [(entry.peer_rank, entry.batch_size) for entry in plan.entries] == [
        (0, 4),
        (1, 2),
    ]

    plan = build_send_plan(
        src_group_name="env",
        dst_group_name="rollout",
        src_rank=1,
        src_world_size=2,
        dst_world_size=3,
        tag="train_obs",
        batch_size=12,
    )
    assert [(entry.peer_rank, entry.batch_size) for entry in plan.entries] == [
        (1, 2),
        (2, 4),
    ]


def test_build_send_plan_load_balance_rollout_to_env():
    plan = build_send_plan(
        src_group_name="rollout",
        dst_group_name="env",
        src_rank=0,
        src_world_size=3,
        dst_world_size=2,
        tag="train_actions",
        batch_size=12,
    )
    assert [(entry.peer_rank, entry.batch_size) for entry in plan.entries] == [(0, 4)]

    plan = build_send_plan(
        src_group_name="rollout",
        dst_group_name="env",
        src_rank=1,
        src_world_size=3,
        dst_world_size=2,
        tag="train_actions",
        batch_size=12,
    )
    assert [(entry.peer_rank, entry.batch_size) for entry in plan.entries] == [
        (0, 2),
        (1, 2),
    ]

    plan = build_send_plan(
        src_group_name="rollout",
        dst_group_name="env",
        src_rank=2,
        src_world_size=3,
        dst_world_size=2,
        tag="train_actions",
        batch_size=12,
    )
    assert [(entry.peer_rank, entry.batch_size) for entry in plan.entries] == [(1, 4)]


def test_build_recv_plan_matches_expected_receive_sizes():
    assert [
        (entry.peer_rank, entry.batch_size)
        for entry in build_recv_plan(
            src_group_name="env",
            dst_group_name="rollout",
            dst_rank=0,
            src_world_size=2,
            dst_world_size=3,
            tag="train_obs",
            batch_size=12,
        ).entries
    ] == [(0, 4)]
    assert [
        (entry.peer_rank, entry.batch_size)
        for entry in build_recv_plan(
            src_group_name="env",
            dst_group_name="rollout",
            dst_rank=1,
            src_world_size=2,
            dst_world_size=3,
            tag="train_obs",
            batch_size=12,
        ).entries
    ] == [(0, 2), (1, 2)]
    assert [
        (entry.peer_rank, entry.batch_size)
        for entry in build_recv_plan(
            src_group_name="env",
            dst_group_name="rollout",
            dst_rank=2,
            src_world_size=2,
            dst_world_size=3,
            tag="train_obs",
            batch_size=12,
        ).entries
    ] == [(1, 4)]


def test_build_route_channel_key_is_stable():
    assert build_route_channel_key("env", "rollout", 2, 1, "train") == (
        "scheduler_route",
        "env",
        "rollout",
        "train",
        "",
        2,
        1,
    )
    assert build_route_channel_key("rollout", "env", 0, 3, "eval", "k") == (
        "scheduler_route",
        "rollout",
        "env",
        "eval",
        "k",
        0,
        3,
    )


def test_split_and_merge_nested_batches():
    batch = {
        "obs": _make_obs(0, 6),
        "final_obs": None,
        "rewards": torch.arange(6, dtype=torch.float32).unsqueeze(-1),
    }
    shards = split_batch(batch, [4, 2])
    assert shards[0]["obs"]["states"].shape[0] == 4
    assert len(shards[1]["obs"]["task_descriptions"]) == 2

    merged = merge_batches(shards)
    assert torch.equal(merged["obs"]["states"], batch["obs"]["states"])
    assert merged["obs"]["task_descriptions"] == batch["obs"]["task_descriptions"]
    assert torch.equal(merged["rewards"], batch["rewards"])


def test_merge_env_outputs_with_partial_optional_fields():
    env_output_0 = EnvOutput(
        obs=_make_obs(0, 2),
        final_obs=None,
        dones=torch.zeros((2, 1), dtype=torch.bool),
        terminations=torch.zeros((2, 1), dtype=torch.bool),
        truncations=torch.zeros((2, 1), dtype=torch.bool),
        rewards=torch.ones((2, 1), dtype=torch.float32),
    ).to_dict()
    env_output_1 = EnvOutput(
        obs=_make_obs(100, 3),
        final_obs=_make_obs(200, 3),
        dones=torch.zeros((3, 1), dtype=torch.bool),
        terminations=torch.zeros((3, 1), dtype=torch.bool),
        truncations=torch.zeros((3, 1), dtype=torch.bool),
        rewards=torch.ones((3, 1), dtype=torch.float32) * 2,
    ).to_dict()

    merged = EnvOutput.merge_env_outputs([env_output_0, env_output_1])

    assert merged["obs"]["states"].shape[0] == 5
    assert len(merged["obs"]["task_descriptions"]) == 5
    assert merged["rewards"].shape[0] == 5
    assert merged["final_obs"] is not None
    assert torch.equal(merged["final_obs"]["states"][:2], env_output_0["obs"]["states"])
    assert torch.equal(
        merged["final_obs"]["states"][2:], env_output_1["final_obs"]["states"]
    )
