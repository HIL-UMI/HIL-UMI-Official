# Copyright 2025 The RLinf Authors.
# Modified for HIL-UMI.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Any, Optional

import torch

from training.utils.nested_dict_process import (
    put_tensor_device,
)


@dataclass(kw_only=True)
class EnvOutput:
    """Environment output for a single chunk step."""

    obs: dict[str, Any]
    final_obs: Optional[dict[str, Any]] = None
    dones: Optional[torch.Tensor] = None  # [B]
    terminations: Optional[torch.Tensor] = None  # [B]
    truncations: Optional[torch.Tensor] = None  # [B]
    rewards: Optional[torch.Tensor] = None  # [B]
    env_infos: Optional[dict[str, Any]] = None

    def __post_init__(self):
        self.obs = put_tensor_device(self.obs, "cpu")
        self.final_obs = (
            put_tensor_device(self.final_obs, "cpu")
            if self.final_obs is not None
            else None
        )
        self.dones = self.dones.cpu().contiguous() if self.dones is not None else None
        self.terminations = (
            self.terminations.cpu().contiguous()
            if self.terminations is not None
            else None
        )
        self.truncations = (
            self.truncations.cpu().contiguous()
            if self.truncations is not None
            else None
        )
        self.rewards = (
            self.rewards.cpu().contiguous() if self.rewards is not None else None
        )
        self.env_infos = (
            put_tensor_device(self.env_infos, "cpu")
            if self.env_infos is not None
            else None
        )

    def prepare_observations(self, obs: dict[str, Any]) -> dict[str, Any]:
        image_tensor = obs["main_images"] if "main_images" in obs else None
        wrist_image_tensor = obs["wrist_images"] if "wrist_images" in obs else None
        extra_view_image_tensor = (
            obs["extra_view_images"] if "extra_view_images" in obs else None
        )
        states = obs["states"] if "states" in obs else None
        task_descriptions = (
            list(obs["task_descriptions"])
            if "task_descriptions" in obs and obs["task_descriptions"] is not None
            else None
        )

        return {
            "main_images": image_tensor,  # [N_ENV, H, W, C]
            "wrist_images": wrist_image_tensor,  # [N_ENV, H, W, C] or [N_ENV, N_IMG, H, W, C]
            "extra_view_images": extra_view_image_tensor,  # [N_ENV, N_IMG, H, W, C]
            "states": states,
            "task_descriptions": task_descriptions,
        }

    @staticmethod
    def merge_env_outputs(env_outputs: list[dict]) -> dict[str, Any]:
        """Merge multiple env output dicts into one batch-aligned env output.

        Merge strategy:

        - Tensor fields: concatenate on batch dimension.
        - List fields: flatten in source order.
        - ``None`` fields: keep ``None``.
        - ``final_obs`` supports partial ``None`` across shards. For shards
            without ``final_obs``, use the corresponding ``obs`` as fallback to
            keep batch alignment.

        Args:
            env_outputs: Per-source env output dicts that share the same schema.

        Returns:
            A merged env output dict produced via ``EnvOutput(...).to_dict()``.
        """

        def _get_batch_size(env_output: dict[str, Any]) -> int:
            dones = env_output.get("dones")
            if isinstance(dones, torch.Tensor):
                return dones.shape[0]

            obs = env_output["obs"]
            for key in ("states", "main_images", "task_descriptions"):
                value = obs.get(key)
                if isinstance(value, torch.Tensor):
                    return value.shape[0]
                if isinstance(value, list):
                    return len(value)
            raise ValueError("Cannot infer batch size from env output.")

        def _merge_obs_dicts(obs_dicts: list[dict[str, Any]]) -> dict[str, Any]:
            merged_obs = {}
            for key in obs_dicts[0].keys():
                obs_elements = [obs_dict[key] for obs_dict in obs_dicts]
                first_non_none = next(
                    (element for element in obs_elements if element is not None), None
                )
                if first_non_none is None:
                    merged_obs[key] = None
                elif isinstance(first_non_none, torch.Tensor):
                    merged_obs[key] = torch.cat(obs_elements, dim=0)
                elif isinstance(first_non_none, list):
                    merged_obs[key] = [
                        item for sublist in obs_elements for item in sublist
                    ]
                else:
                    merged_obs[key] = obs_elements
            return merged_obs

        def _merge_optional_tensor_field(
            field_name: str,
            *,
            allow_partial_none: bool = False,
            fill_value: float | bool = 0,
        ) -> torch.Tensor | None:
            values = [env_output[field_name] for env_output in env_outputs]
            if all(value is None for value in values):
                return None

            if any(value is None for value in values):
                if not allow_partial_none:
                    raise ValueError(
                        f"Inconsistent field '{field_name}': some shards are None while others are tensors."
                    )

                ref_tensor = next(value for value in values if value is not None)
                filled_values = []
                for env_output, value in zip(env_outputs, values):
                    if value is None:
                        batch_size = _get_batch_size(env_output)
                        fill_shape = (batch_size, *ref_tensor.shape[1:])
                        filled_values.append(
                            torch.full(
                                fill_shape,
                                fill_value=fill_value,
                                dtype=ref_tensor.dtype,
                            )
                        )
                    else:
                        filled_values.append(value)
                values = filled_values

            return torch.cat(values, dim=0)

        merged_obs = _merge_obs_dicts([env_output["obs"] for env_output in env_outputs])

        merged_final_obs = None
        final_obs_list = [env_output["final_obs"] for env_output in env_outputs]
        if any(final_obs is not None for final_obs in final_obs_list):
            # Some shards may not have done episodes in this step, so their final_obs
            # is None. Use obs as fallback to keep merged batch shape aligned.
            final_obs_or_obs = [
                final_obs if final_obs is not None else env_output["obs"]
                for env_output, final_obs in zip(env_outputs, final_obs_list)
            ]
            merged_final_obs = _merge_obs_dicts(final_obs_or_obs)

        merged_dones = _merge_optional_tensor_field("dones")
        merged_terminations = _merge_optional_tensor_field("terminations")
        merged_truncations = _merge_optional_tensor_field("truncations")
        merged_rewards = _merge_optional_tensor_field("rewards")
        # turn to EnvOutput and turn to dict to call post init for tensor processing
        return EnvOutput(
            obs=merged_obs,
            final_obs=merged_final_obs,
            dones=merged_dones,
            terminations=merged_terminations,
            truncations=merged_truncations,
            rewards=merged_rewards,
        ).to_dict()

    def to_dict(self) -> dict[str, Any]:
        env_output_dict = {}

        env_output_dict["obs"] = self.prepare_observations(self.obs)
        env_output_dict["final_obs"] = (
            self.prepare_observations(self.final_obs)
            if self.final_obs is not None
            else None
        )
        env_output_dict["dones"] = self.dones
        env_output_dict["terminations"] = self.terminations
        env_output_dict["truncations"] = self.truncations
        env_output_dict["rewards"] = self.rewards
        env_output_dict["env_infos"] = self.env_infos

        return env_output_dict
