# Copyright 2025 The RLinf Authors.
# Modified for HIL-UMI.
# SPDX-License-Identifier: Apache-2.0

import numbers
import os
import warnings
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Optional

import gymnasium as gym
import imageio
import numpy as np

try:
    import torch
except ImportError:
    torch = None

from training.envs.utils import put_info_on_image, tile_images


class RecordVideo(gym.Wrapper):
    """
    A general video recording wrapper that owns the recording logic.

    ``RecordVideo`` centralizes frame collection and MP4 writing for both regular
    stepping and chunked stepping APIs. Frames are buffered in memory and flushed
    asynchronously to avoid blocking environment interaction.

    The wrapper supports multiple observation image layouts (single frame, batched
    frames, and temporal batches). For ``chunk_step()``, it correctly handles the
    terminal-to-reset transition by recording terminal observations (for the last
    step in the chunk) and then appending the corresponding reset observations.

    When ``video_cfg.info_on_video`` is enabled, per-frame text metadata is drawn
    through ``put_info_on_image()``. The overlay always includes reward and
    termination when available, and can include extra fields from environment
    ``info`` via ``video_cfg.extra_info_on_video``. Nested keys are supported with
    dot notation, for example
    ``["env_id", "episode.success_once", "episode.episode_len"]``.

    Args:
        env: Wrapped environment. It must expose a ``seed`` attribute and may
            optionally provide ``num_envs`` and metadata for FPS inference.
        video_cfg: Video configuration object/dict. Common fields:
            ``video_base_dir`` (output directory root),
            ``fps`` (optional FPS override),
            ``info_on_video`` (whether to render overlay text),
            ``extra_info_on_video`` (list of ``info`` keys to render), and
            ``additional_image_fields`` (mapping from output-view labels to
            observation fields). The primary view keeps the legacy ``N.mp4``
            name; configured additional views are written as
            ``N_<view_label>.mp4`` in the same directory.
        fps: Explicit FPS override. If ``None``, FPS is resolved from
            ``video_cfg.fps``, environment config/metadata, then fallback ``30``.
    """

    def __init__(self, env: gym.Env, video_cfg, fps: Optional[int] = None):
        """Initialize the wrapper and set FPS/config."""
        if isinstance(env, gym.Env):
            super().__init__(env)
        else:
            self.env = env

        if not hasattr(env, "seed"):
            raise AttributeError("Environment must have 'seed' attribute")

        self.video_cfg = video_cfg
        self.render_images: list[np.ndarray] = []
        self._additional_image_fields = self._get_additional_image_fields()
        self._additional_render_images: dict[str, list[np.ndarray]] = {
            label: [] for label in self._additional_image_fields
        }
        self.video_cnt = 0
        self._num_envs = getattr(env, "num_envs", 1)
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._save_futures: list[Future] = []

        if fps is not None:
            self._fps = fps
        else:
            self._fps = self._get_fps_from_env(env)

    def _get_additional_image_fields(self) -> dict[str, str]:
        """Return validated labels and observation fields for extra videos."""
        configured = self.video_cfg.get("additional_image_fields", {})
        if configured is None:
            return {}

        fields = dict(configured)
        validated: dict[str, str] = {}
        for raw_label, raw_field in fields.items():
            label = str(raw_label)
            field = str(raw_field)
            if not label or not field:
                raise ValueError(
                    "video_cfg.additional_image_fields labels and fields "
                    "must be non-empty"
                )
            if any(
                char
                not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
                for char in label
            ):
                raise ValueError(
                    "Video view labels may contain only letters, digits, '-' and '_'; "
                    f"got {label!r}"
                )
            if label in validated:
                raise ValueError(f"Duplicate additional video view label: {label!r}")
            validated[label] = field
        return validated

    @property
    def is_start(self):
        return getattr(self.env, "is_start")

    @is_start.setter
    def is_start(self, value):
        setattr(self.env, "is_start", value)

    def _get_fps_from_env(self, env: gym.Env) -> int:
        """Resolve FPS from config/env metadata with fallback."""
        if hasattr(self.video_cfg, "fps") and self.video_cfg.fps is not None:
            return int(self.video_cfg.fps)
        if hasattr(env, "cfg") and hasattr(env.cfg, "init_params"):
            if hasattr(env.cfg.init_params, "sim_config"):
                if hasattr(env.cfg.init_params.sim_config, "control_freq"):
                    return int(env.cfg.init_params.sim_config.control_freq)
        metadata = getattr(env, "metadata", None)
        if isinstance(metadata, dict) and "render_fps" in metadata:
            return int(metadata["render_fps"])
        return 30

    def _to_numpy(self, value: Any) -> np.ndarray:
        """Convert tensors/arrays to numpy."""
        if torch is not None and isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        if isinstance(value, np.ndarray):
            return value
        return np.array(value)

    def _get_image_from_dict(self, obs: dict) -> Optional[Any]:
        """Pick the best image field from an observation dict."""
        if hasattr(self.env, "capture_image"):
            return self.env.capture_image()
        for key in ("main_images", "images", "rgb", "full_image", "main_image"):
            if key in obs and obs[key] is not None:
                return obs[key]
        return None

    def _extract_frame_batches(
        self, obs: Any, image_field: Optional[str] = None
    ) -> list[list[np.ndarray]]:
        """Extract per-step image batches from the primary or a named field."""
        if obs is None:
            return []

        if isinstance(obs, dict):
            image_src = (
                self._get_image_from_dict(obs)
                if image_field is None
                else obs.get(image_field)
            )
            if image_src is None:
                return []
            return self._split_image_source(image_src)

        if isinstance(obs, (list, tuple)):
            if len(obs) == 0:
                return []
            if isinstance(obs[0], dict):
                frames = []
                for item in obs:
                    image_src = (
                        self._get_image_from_dict(item)
                        if image_field is None
                        else item.get(image_field)
                    )
                    if image_src is None:
                        continue
                    batches = self._split_image_source(image_src)
                    if batches:
                        frames.append(batches[0])
                return frames
            if image_field is not None:
                return []
            images = []
            for item in obs:
                img = self._to_numpy(item)
                if img.dtype != np.uint8:
                    img = img.astype(np.uint8)
                images.append(img)
            return [images] if images else []

        if image_field is not None:
            return []
        if torch is not None and isinstance(obs, torch.Tensor):
            return self._split_image_source(obs)
        if isinstance(obs, np.ndarray):
            return self._split_image_source(obs)
        return []

    def _split_image_source(self, image_src: Any) -> list[list[np.ndarray]]:
        """Normalize common image tensor layouts into frame batches."""
        img = self._to_numpy(image_src)

        if img.ndim == 3:
            if img.shape[0] in (1, 3, 4) and img.shape[-1] not in (1, 3, 4):
                img = np.transpose(img, (1, 2, 0))
            if img.dtype != np.uint8:
                img = img.astype(np.uint8)
            return [[img]]

        if img.ndim == 4:
            if img.shape[1] in (1, 3, 4) and img.shape[-1] not in (1, 3, 4):
                img = np.transpose(img, (0, 2, 3, 1))
            images = []
            for i in range(img.shape[0]):
                single = img[i]
                if single.dtype != np.uint8:
                    single = single.astype(np.uint8)
                images.append(single)
            return [images]

        if img.ndim == 5:
            if img.shape[2] in (1, 3, 4) and img.shape[-1] not in (1, 3, 4):
                img = np.transpose(img, (0, 1, 3, 4, 2))
            frames = []
            for t in range(img.shape[1]):
                images = []
                for i in range(img.shape[0]):
                    single = img[i, t]
                    if single.dtype != np.uint8:
                        single = single.astype(np.uint8)
                    images.append(single)
                frames.append(images)
            return frames

        return []

    def _value_for_env(self, value: Any, env_id: int):
        """Select a scalar/value for a specific env from batched inputs."""
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        if isinstance(value, np.ndarray):
            if value.shape == ():
                return value.item()
            if value.size == 1:
                return value.reshape(-1)[0].item()
            if value.shape[0] > env_id:
                return value[env_id]
            return value.reshape(-1)[0]
        if isinstance(value, (list, tuple)):
            if len(value) > env_id:
                return value[env_id]
            if len(value) > 0:
                return value[0]
        return value

    def _get_task_description(self, obs: Any, env_id: int):
        """Get task description from obs or env attribute."""
        if isinstance(obs, dict) and "task_descriptions" in obs:
            task_desc = obs["task_descriptions"]
            if isinstance(task_desc, (list, tuple)) and len(task_desc) > env_id:
                return task_desc[env_id]
            return task_desc[0] if isinstance(task_desc, (list, tuple)) else task_desc
        if hasattr(self.env, "task_descriptions"):
            task_desc = self.env.task_descriptions
            if isinstance(task_desc, (list, tuple)) and len(task_desc) > env_id:
                return task_desc[env_id]
            return task_desc[0] if isinstance(task_desc, (list, tuple)) else task_desc
        return None

    def _get_video_info_keys(self) -> list[str]:
        """Get configured info keys to overlay on video frames."""
        if hasattr(self.video_cfg, "extra_info_on_video"):
            keys = getattr(self.video_cfg, "extra_info_on_video")
        else:
            keys = None

        if keys:
            if isinstance(keys, str):
                return [keys]
            return list(keys)
        return []

    def _lookup_info_value(self, info: Any, key: str) -> Any:
        """Read a key from info, supporting dotted access for nested dicts."""
        if not isinstance(info, dict):
            return None
        if key in info:
            return info[key]

        value = info
        for part in key.split("."):
            if not isinstance(value, dict) or part not in value:
                return None
            value = value[part]
        return value

    def _build_info_item(
        self,
        infos: Optional[Any],
        rewards: Optional[Any],
        terminations: Optional[Any],
        env_id: int,
        time_idx: Optional[int] = None,
    ) -> dict:
        """Build a per-env info dict for overlay."""
        info_item: dict[str, Any] = {}

        if rewards is not None:
            value = self._value_for_env(rewards, env_id)
            if time_idx is not None and isinstance(value, (np.ndarray, list, tuple)):
                if len(value) > time_idx:
                    value = value[time_idx]
            info_item["reward"] = float(value) if value is not None else value

        if terminations is not None:
            value = self._value_for_env(terminations, env_id)
            if time_idx is not None and isinstance(value, (np.ndarray, list, tuple)):
                if len(value) > time_idx:
                    value = value[time_idx]
            info_item["termination"] = bool(value) if value is not None else value

        if infos is not None:
            for key in self._get_video_info_keys():
                value = self._lookup_info_value(infos, key)
                if value is None:
                    continue
                value = self._value_for_env(value, env_id)
                if isinstance(value, np.ndarray):
                    if value.shape == ():
                        value = value.item()
                    elif value.size == 1:
                        value = value.reshape(-1)[0].item()
                elif isinstance(value, numbers.Number):
                    pass
                else:
                    warnings.warn(f"Unsupported value type {type(value)} for key {key}")
                    continue
                info_item[key] = value

        return info_item

    def _append_frame(
        self,
        images: list[np.ndarray],
        infos: Optional[Any],
        rewards: Optional[Any],
        terminations: Optional[Any],
        time_idx: Optional[int] = None,
        frame_buffer: Optional[list[np.ndarray]] = None,
    ) -> None:
        """Overlay info (optional) and append a tiled frame."""
        if not images:
            return
        if frame_buffer is None:
            frame_buffer = self.render_images
        if self.video_cfg.get("info_on_video", True):
            images = [
                put_info_on_image(
                    img,
                    self._build_info_item(
                        infos, rewards, terminations, env_id, time_idx
                    ),
                )
                for env_id, img in enumerate(images)
            ]
        if len(images) > 1:
            nrows = int(np.sqrt(len(images)))
            full_image = tile_images(images, nrows=nrows)
            frame_buffer.append(full_image)
        else:
            frame_buffer.append(images[0])

    def _append_frame_batches(
        self,
        frames: list[list[np.ndarray]],
        infos: Optional[Any],
        rewards: Optional[Any],
        terminations: Optional[Any],
        frame_buffer: list[np.ndarray],
    ) -> None:
        """Append extracted temporal frames to one view-specific buffer."""
        if isinstance(infos, (list, tuple)):
            for time_idx, images in enumerate(frames):
                step_info = infos[time_idx] if len(infos) > time_idx else None
                self._append_frame(
                    images,
                    step_info,
                    rewards,
                    terminations,
                    time_idx,
                    frame_buffer,
                )
            return

        for time_idx, images in enumerate(frames):
            self._append_frame(
                images,
                infos,
                rewards,
                terminations,
                time_idx,
                frame_buffer,
            )

    def add_new_frames(
        self,
        obs: Any,
        infos: Optional[Any] = None,
        rewards: Optional[Any] = None,
        terminations: Optional[Any] = None,
    ):
        """Extract frames from obs and append to the buffer."""
        frames = self._extract_frame_batches(obs)
        if not frames:
            warnings.warn(
                f"Failed to extract images from obs, obs type: {type(obs)}, obs keys: "
                f"{list(obs.keys()) if isinstance(obs, dict) else 'N/A'}"
            )
        else:
            self._append_frame_batches(
                frames, infos, rewards, terminations, self.render_images
            )

        for label, image_field in self._additional_image_fields.items():
            additional_frames = self._extract_frame_batches(obs, image_field)
            if not additional_frames:
                continue
            self._append_frame_batches(
                additional_frames,
                infos,
                rewards,
                terminations,
                self._additional_render_images[label],
            )

    def reset(self, *args, **kwargs):
        """Reset env and record the initial frame."""
        obs, info = self.env.reset(*args, **kwargs)
        self.add_new_frames(obs, info)
        return obs, info

    def step(self, action):
        """Step env and record the resulting frame."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        terminations = (
            info.get("terminations", terminated)
            if isinstance(info, dict)
            else terminated
        )
        self.add_new_frames(obs, info, reward, terminations)
        return obs, reward, terminated, truncated, info

    def record_video_in_result(self, result) -> None:
        """Record video frames from a chunk_step / async_chunk_step result tuple."""
        if isinstance(result, tuple) and len(result) >= 5:
            obs_list, rewards, terminations, _truncations, infos_list = result[:5]

            # Some envs may skip intermediate observations for performance and return
            # None entries. Filter them out for video collection.
            if isinstance(obs_list, (list, tuple)):
                valid_indices = [i for i, obs in enumerate(obs_list) if obs is not None]
                if len(valid_indices) == 0:
                    return
                if len(valid_indices) != len(obs_list):
                    obs_list = [obs_list[i] for i in valid_indices]
                    if isinstance(infos_list, (list, tuple)):
                        infos_list = [infos_list[i] for i in valid_indices]
                    if (
                        torch is not None
                        and isinstance(rewards, torch.Tensor)
                        and rewards.ndim == 2
                    ):
                        rewards = rewards[:, valid_indices]
                    if (
                        torch is not None
                        and isinstance(terminations, torch.Tensor)
                        and terminations.ndim == 2
                    ):
                        terminations = terminations[:, valid_indices]

            final_obs = None
            last_info = None
            if isinstance(infos_list, (list, tuple)) and len(infos_list) > 0:
                last_info = infos_list[-1]
                if isinstance(last_info, dict):
                    if last_info.get("final_obs") is not None:
                        final_obs = last_info["final_obs"]
                    elif last_info.get("final_observation") is not None:
                        final_obs = last_info["final_observation"]

            if (
                final_obs is not None
                and isinstance(obs_list, (list, tuple))
                and len(obs_list) > 0
            ):
                reset_obs = obs_list[-1]
                obs_main = list(obs_list)
                obs_main[-1] = final_obs
                infos_main = (
                    list(infos_list)
                    if isinstance(infos_list, (list, tuple))
                    else infos_list
                )
                self.add_new_frames(obs_main, infos_main, rewards, terminations)
                self.add_new_frames(reset_obs, None)
            else:
                self.add_new_frames(obs_list, infos_list, rewards, terminations)

    def chunk_step(self, *args, **kwargs):
        """Step a chunk and record all frames from the chunk."""
        result = self.env.chunk_step(*args, **kwargs)
        self.record_video_in_result(result)
        return result

    def flush_video(self, video_sub_dir: Optional[str] = None):
        """Write buffered frames to an MP4 file.

        The encode happens on the background thread pool, but we wait for the
        just-submitted write to complete before returning. The wait is required
        so the MP4 has a finalized ``moov`` atom on disk: ``imageio`` only writes
        it during ``writer.close()``, and the pool's worker threads are daemon
        threads that get killed mid-task at interpreter exit (no ``atexit``
        handler is run under Ray actor shutdown either). Without this wait,
        eval videos end at ``mdat`` and no player can open them.
        """
        populated_additional_views = {
            label: frames
            for label, frames in self._additional_render_images.items()
            if frames
        }
        if not self.render_images and not populated_additional_views:
            return

        output_dir = os.path.join(
            self.video_cfg.get("video_base_dir"), f"seed_{self.env.seed}"
        )
        if video_sub_dir is not None:
            output_dir = os.path.join(output_dir, f"{video_sub_dir}")

        os.makedirs(output_dir, exist_ok=True)
        video_index = self.video_cnt
        pending_writes: list[Future] = []
        if self.render_images:
            mp4_path = os.path.join(output_dir, f"{video_index}.mp4")
            frames = list(self.render_images)
            pending_writes.append(self._submit_save(frames, mp4_path))
        self.render_images = []
        for label, view_frames in populated_additional_views.items():
            mp4_path = os.path.join(output_dir, f"{video_index}_{label}.mp4")
            pending_writes.append(self._submit_save(list(view_frames), mp4_path))
        for frames in self._additional_render_images.values():
            frames.clear()
        self.video_cnt += 1
        # Block until the encode + writer.close() returns so the MP4 is valid
        # on disk before the rollout loop continues (or the process exits).
        for future in pending_writes:
            future.result()

    def _submit_save(self, frames: list[np.ndarray], mp4_path: str) -> Future:
        """Submit a background job to save the video, return its Future."""
        self._prune_futures()
        future = self._executor.submit(self._save_video, frames, mp4_path)
        self._save_futures.append(future)
        return future

    def _save_video(self, frames: list[np.ndarray], mp4_path: str) -> None:
        """Save frames to disk (runs in background)."""
        video_writer = None
        try:
            video_writer = imageio.get_writer(mp4_path, fps=self._fps)
            for img in frames:
                video_writer.append_data(img)
        except Exception as exc:
            warnings.warn(f"Failed to save video {mp4_path}: {exc}")
        finally:
            if video_writer is not None:
                video_writer.close()

    def _prune_futures(self) -> None:
        """Remove finished futures to avoid unbounded growth."""
        self._save_futures = [f for f in self._save_futures if not f.done()]

    def close(self):
        """Wait for pending video writes before closing."""
        self._executor.shutdown(wait=True)
        self._save_futures = []
        return super().close()

    def update_reset_state_ids(self):
        if hasattr(self.env, "update_reset_state_ids"):
            self.env.update_reset_state_ids()
