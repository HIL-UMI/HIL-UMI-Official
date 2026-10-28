# Copyright 2026 The RLinf Authors.
# Modified for HIL-UMI.
# SPDX-License-Identifier: Apache-2.0

"""Browser UI for aligning a live camera view with a selected video frame."""

from __future__ import annotations

import json
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import cv2
import numpy as np

from training.envs.realworld.common.scene_reference_set import SceneReferenceSet


class ReferenceSceneAlignmentWebUI:
    """Serve a live/reference image overlay without requiring OpenCV HighGUI.

    One recorded video is selected per evaluation epoch. The operator can
    preview any frame and persist one reference image per epoch. Later runs load
    that cached image directly. The live front-camera image is refreshed by the
    environment's existing camera pipeline, and the browser performs the alpha
    blend. This works with the headless OpenCV build used by the Franka
    environment and never opens another camera pipeline.
    """

    JPEG_QUALITY = 90

    def __init__(
        self,
        reference_video_dir: str | None = None,
        *,
        reference_set_dir: str | None = None,
        video_suffix: str = "wrist_2",
        epoch_offset: int = 0,
        cache_dir: str | None = None,
        video_paths: dict[int, str | Path] | None = None,
        source_epoch_indices: dict[int, int] | None = None,
        calibration_epochs: list[int] | tuple[int, ...] | None = None,
        host: str = "127.0.0.1",
        port: int = 8765,
        default_reference_opacity: float = 0.35,
    ) -> None:
        self.reference_set = (
            SceneReferenceSet.load(reference_set_dir)
            if reference_set_dir is not None
            else None
        )
        if self.reference_set is not None:
            self.reference_set.validate_saved_references()
            self.reference_video_dir: Path | None = None
        else:
            if reference_video_dir is None:
                raise ValueError(
                    "reference_video_dir is required without reference_set_dir."
                )
            self.reference_video_dir = Path(reference_video_dir).expanduser().resolve()
            if not self.reference_video_dir.is_dir():
                raise ValueError(
                    "scene_alignment_reference_dir must be an existing directory, "
                    f"got {str(self.reference_video_dir)!r}."
                )
        self.video_suffix = str(video_suffix).strip()
        if not self.video_suffix:
            raise ValueError("scene_alignment_video_suffix must not be empty.")
        self.epoch_offset = int(epoch_offset)
        if self.epoch_offset < 0:
            raise ValueError("scene_alignment_epoch_offset must be non-negative.")
        self.cache_dir = (
            Path(cache_dir).expanduser().resolve()
            if cache_dir is not None and str(cache_dir).strip()
            else (
                self.reference_set.references_dir
                if self.reference_set is not None
                else self.reference_video_dir / "scene_alignment_references"
            )
        )
        self._video_paths = {
            int(epoch): Path(path).expanduser().resolve()
            for epoch, path in (video_paths or {}).items()
        }
        self._source_epoch_indices = {
            int(epoch): int(index)
            for epoch, index in (source_epoch_indices or {}).items()
        }
        self._calibration_epochs = tuple(
            sorted({int(epoch) for epoch in (calibration_epochs or [])})
        )
        if self._calibration_epochs and self.reference_set is not None:
            raise ValueError(
                "calibration_epochs cannot be used with an inference reference set."
            )
        if self._calibration_epochs and set(self._calibration_epochs) != set(
            self._video_paths
        ):
            raise ValueError(
                "calibration_epochs must exactly match the explicit video_paths."
            )
        self.host = str(host).strip()
        if not self.host:
            raise ValueError("scene_alignment_host must not be empty.")
        self.port = int(port)
        if not 1 <= self.port <= 65535:
            raise ValueError("scene_alignment_port must be in [1, 65535].")
        self.default_reference_opacity = float(default_reference_opacity)
        if not 0.0 <= self.default_reference_opacity <= 1.0:
            raise ValueError("scene_alignment_reference_opacity must be in [0, 1].")

        self._lock = threading.Lock()
        self._decode_lock = threading.Lock()
        self._reference_jpeg: bytes | None = None
        self._live_jpeg: bytes | None = None
        self._reference_shape: tuple[int, int] | None = None
        self._epoch_index: int | None = None
        self._video_path: Path | None = None
        self._reference_path: Path | None = None
        self._frame_count = 0
        self._fps = 0.0
        self._selected_frame_index = 0
        self._reference_saved = False
        self._reference_version = 0
        self._alignment_active = False
        self._frame_selection_enabled = False
        self._status = "Waiting for the first evaluation epoch."
        self._server: ThreadingHTTPServer | None = None
        self._server_thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        """Return a locally usable URL for the alignment page."""
        display_host = "127.0.0.1" if self.host in {"0.0.0.0", "::"} else self.host
        return f"http://{display_host}:{self.port}/"

    def start_epoch(self, epoch_index: int) -> Path:
        """Load the saved reference or first frame for ``epoch_index``."""
        epoch_index = int(epoch_index)
        if epoch_index < 0:
            raise ValueError("scene-alignment epoch index must be non-negative.")
        if self.reference_set is not None:
            return self._start_reference_set_epoch(epoch_index)

        video_path, source_epoch_index = self._source_video(epoch_index)
        frame_count, fps = self._inspect_video(video_path)
        cached_image_path, metadata_path = self._cache_paths(epoch_index)
        if cached_image_path.is_file():
            reference_frame = cv2.imread(str(cached_image_path), cv2.IMREAD_COLOR)
            if reference_frame is None:
                raise RuntimeError(
                    "Could not decode the saved scene-alignment reference image: "
                    f"{cached_image_path}"
                )
            selected_frame_index = self._read_cached_frame_index(
                metadata_path,
                frame_count=frame_count,
            )
            reference_path = cached_image_path
            reference_saved = True
        else:
            reference_frame = self._read_video_frame(video_path, 0)
            selected_frame_index = 0
            reference_path = video_path
            reference_saved = False
        reference_jpeg = self._encode_jpeg(reference_frame)

        with self._lock:
            self._epoch_index = epoch_index
            self._video_path = video_path
            self._reference_path = reference_path
            self._reference_shape = reference_frame.shape[:2]
            self._reference_jpeg = reference_jpeg
            self._live_jpeg = None
            self._frame_count = frame_count
            self._fps = fps
            self._selected_frame_index = selected_frame_index
            self._reference_saved = reference_saved
            self._reference_version += 1
            self._alignment_active = True
            self._frame_selection_enabled = True
            if reference_saved:
                self._status = (
                    f"Epoch {epoch_index}: using saved frame "
                    f"{selected_frame_index}; align the scene, then press pedal 'a'."
                )
            else:
                self._status = (
                    f"Epoch {epoch_index}: frame 0 is temporary. Choose a clean "
                    "frame and save it, then align the scene."
                )
        self._ensure_server()
        return video_path

    def _start_reference_set_epoch(self, epoch_index: int) -> Path:
        reference_image_path = self.reference_set.reference_image_path(epoch_index)
        reference_frame = cv2.imread(
            str(reference_image_path),
            cv2.IMREAD_COLOR,
        )
        if reference_frame is None:
            raise RuntimeError(
                f"Could not decode scene-reference image: {reference_image_path}"
            )
        expected_shape = (self.reference_set.height, self.reference_set.width)
        if reference_frame.shape[:2] != expected_shape:
            raise ValueError(
                f"Scene-reference image {reference_image_path} has resolution "
                f"{reference_frame.shape[1]}x{reference_frame.shape[0]}, expected "
                f"{self.reference_set.width}x{self.reference_set.height}."
            )
        metadata_path = self.reference_set.reference_metadata_path(epoch_index)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        selected_frame_index = int(metadata.get("frame_index", 0))
        reference_jpeg = self._encode_jpeg(reference_frame)
        with self._lock:
            self._epoch_index = epoch_index
            self._video_path = None
            self._reference_path = reference_image_path
            self._reference_shape = reference_frame.shape[:2]
            self._reference_jpeg = reference_jpeg
            self._live_jpeg = None
            self._frame_count = 0
            self._fps = float(metadata.get("fps", 0.0))
            self._selected_frame_index = selected_frame_index
            self._reference_saved = True
            self._reference_version += 1
            self._alignment_active = True
            self._frame_selection_enabled = False
            self._status = (
                f"Epoch {epoch_index}: using {self.reference_set.name!r} reference; "
                "align the scene, then press pedal 'a'."
            )
        self._ensure_server()
        return reference_image_path

    def select_calibration_epoch(self, epoch_index: int) -> dict[str, object]:
        """Switch an offline calibration page to another configured epoch."""
        epoch_index = int(epoch_index)
        if epoch_index not in self._calibration_epochs:
            raise ValueError(
                f"Calibration epoch must be one of {self._calibration_epochs}, "
                f"got {epoch_index}."
            )
        video_path = self.start_epoch(epoch_index)
        return {
            "epoch": epoch_index,
            "video_path": str(video_path),
            "reference_saved": self._reference_saved,
        }

    def preview_video_frame(self, epoch_index: int, frame_index: int) -> bytes:
        """Decode one video frame for an explicit slider preview request."""
        video_path, frame_count, _fps = self._active_video(epoch_index)
        frame_index = self._validate_frame_index(frame_index, frame_count)
        frame = self._read_video_frame(video_path, frame_index)
        return self._encode_jpeg(frame)

    def save_reference_frame(self, epoch_index: int, frame_index: int) -> dict:
        """Persist and activate one reference frame for an evaluation epoch."""
        video_path, frame_count, fps = self._active_video(epoch_index)
        frame_index = self._validate_frame_index(frame_index, frame_count)
        frame = self._read_video_frame(video_path, frame_index)
        reference_jpeg = self._encode_jpeg(frame)

        # Reject a late browser request if the rollout advanced while decoding.
        self._active_video(epoch_index)
        cached_image_path, metadata_path = self._cache_paths(epoch_index)
        cached_image_path.parent.mkdir(parents=True, exist_ok=True)
        self._atomic_write(cached_image_path, reference_jpeg)
        metadata = {
            "epoch": epoch_index,
            "source_epoch_index": self._source_epoch_index(epoch_index),
            "frame_index": frame_index,
            "fps": fps,
            "timestamp_seconds": frame_index / fps if fps > 0 else None,
            "source_video": str(video_path),
        }
        self._atomic_write(
            metadata_path,
            (json.dumps(metadata, indent=2) + "\n").encode("utf-8"),
        )

        with self._lock:
            self._reference_path = cached_image_path
            self._reference_shape = frame.shape[:2]
            self._reference_jpeg = reference_jpeg
            self._selected_frame_index = frame_index
            self._reference_saved = True
            self._reference_version += 1
            reference_version = self._reference_version
            self._status = (
                f"Epoch {epoch_index}: saved frame {frame_index}; align the scene, "
                "then press pedal 'a'."
            )
        return {
            "epoch": epoch_index,
            "frame_index": frame_index,
            "reference_path": str(cached_image_path),
            "reference_version": reference_version,
        }

    def update_live_frame(self, frame_bgr: np.ndarray) -> None:
        """Publish one current front-camera frame to connected browsers."""
        frame = np.asarray(frame_bgr)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(
                "scene-alignment live frame must have shape (H, W, 3), "
                f"got {frame.shape}."
            )
        with self._lock:
            reference_shape = self._reference_shape
        if reference_shape is None:
            raise RuntimeError("start_epoch() must be called before publishing frames.")
        if frame.shape[:2] != reference_shape:
            raise ValueError(
                "Live and reference frames must have identical resolution for "
                f"pixel alignment: live={frame.shape[:2]}, "
                f"reference={reference_shape}."
            )
        live_jpeg = self._encode_jpeg(frame)
        with self._lock:
            self._live_jpeg = live_jpeg

    def mark_rollout_started(self) -> None:
        """Update the page status after the operator accepts the alignment."""
        with self._lock:
            epoch_index = self._epoch_index
            self._status = f"Epoch {epoch_index}: rollout started."
            self._alignment_active = False

    def _source_video(self, epoch_index: int) -> tuple[Path, int]:
        if self._video_paths:
            try:
                return self._video_paths[epoch_index], self._source_epoch_index(
                    epoch_index
                )
            except KeyError as exc:
                raise FileNotFoundError(
                    f"No calibration source video is configured for epoch "
                    f"{epoch_index}."
                ) from exc
        source_epoch_index = epoch_index - self.epoch_offset
        if source_epoch_index < 0:
            raise ValueError(
                f"Epoch {epoch_index} precedes configured scene-alignment offset "
                f"{self.epoch_offset}."
            )
        return (
            self.reference_video_dir / f"{source_epoch_index}_{self.video_suffix}.mp4",
            source_epoch_index,
        )

    def _source_epoch_index(self, epoch_index: int) -> int:
        return self._source_epoch_indices.get(
            epoch_index,
            epoch_index - self.epoch_offset,
        )

    def close(self) -> None:
        """Stop the HTTP server and release its listening socket."""
        server = self._server
        thread = self._server_thread
        self._server = None
        self._server_thread = None
        if server is not None:
            server.shutdown()
            server.server_close()
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)

    @staticmethod
    def _inspect_video(video_path: Path) -> tuple[int, float]:
        if not video_path.is_file():
            raise FileNotFoundError(
                f"Scene-alignment reference video does not exist: {video_path}"
            )
        capture = cv2.VideoCapture(str(video_path))
        try:
            if not capture.isOpened():
                raise RuntimeError(
                    f"Could not open scene-alignment reference video: {video_path}"
                )
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(capture.get(cv2.CAP_PROP_FPS))
        finally:
            capture.release()
        if frame_count <= 0:
            raise RuntimeError(
                f"Reference video reports no decodable frames: {video_path}"
            )
        if not np.isfinite(fps) or fps <= 0:
            fps = 30.0
        return frame_count, fps

    def _read_video_frame(self, video_path: Path, frame_index: int) -> np.ndarray:
        with self._decode_lock:
            capture = cv2.VideoCapture(str(video_path))
            try:
                if not capture.isOpened():
                    raise RuntimeError(
                        f"Could not open scene-alignment reference video: {video_path}"
                    )
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
                ok, frame = capture.read()
            finally:
                capture.release()
        if not ok or frame is None:
            raise RuntimeError(
                f"Could not decode frame {frame_index} of reference video: {video_path}"
            )
        return frame

    def _active_video(self, epoch_index: int) -> tuple[Path, int, float]:
        epoch_index = int(epoch_index)
        with self._lock:
            active_epoch = self._epoch_index
            video_path = self._video_path
            frame_count = self._frame_count
            fps = self._fps
            alignment_active = self._alignment_active
        if active_epoch != epoch_index or video_path is None or not alignment_active:
            raise ValueError(
                f"Epoch {epoch_index} is not accepting frame selections; current "
                f"epoch is {active_epoch}."
            )
        return video_path, frame_count, fps

    @staticmethod
    def _validate_frame_index(frame_index: int, frame_count: int) -> int:
        frame_index = int(frame_index)
        if not 0 <= frame_index < frame_count:
            raise ValueError(
                f"frame_index must be in [0, {frame_count - 1}], got {frame_index}."
            )
        return frame_index

    def _cache_paths(self, epoch_index: int) -> tuple[Path, Path]:
        stem = f"{epoch_index}_{self.video_suffix}_reference"
        return self.cache_dir / f"{stem}.jpg", self.cache_dir / f"{stem}.json"

    @staticmethod
    def _read_cached_frame_index(metadata_path: Path, *, frame_count: int) -> int:
        if not metadata_path.is_file():
            return 0
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            return ReferenceSceneAlignmentWebUI._validate_frame_index(
                metadata["frame_index"], frame_count
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Invalid scene-alignment reference metadata: {metadata_path}"
            ) from exc

    @staticmethod
    def _atomic_write(path: Path, payload: bytes) -> None:
        temporary_path = path.with_name(f".{path.name}.tmp")
        temporary_path.write_bytes(payload)
        temporary_path.replace(path)

    @classmethod
    def _encode_jpeg(cls, frame_bgr: np.ndarray) -> bytes:
        ok, encoded = cv2.imencode(
            ".jpg",
            frame_bgr,
            [cv2.IMWRITE_JPEG_QUALITY, cls.JPEG_QUALITY],
        )
        if not ok:
            raise RuntimeError("OpenCV failed to encode a scene-alignment frame.")
        return encoded.tobytes()

    def _ensure_server(self) -> None:
        if self._server is not None:
            return

        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
                parsed_url = urlsplit(self.path)
                path = parsed_url.path
                if path in {"/", "/index.html"}:
                    self._send_bytes(
                        owner._render_html(),
                        content_type="text/html; charset=utf-8",
                    )
                    return
                if path == "/reference.jpg":
                    self._send_image(owner._snapshot("reference"))
                    return
                if path == "/live.jpg":
                    self._send_image(owner._snapshot("live"))
                    return
                if path == "/status.json":
                    self._send_bytes(
                        json.dumps(owner._status_snapshot()).encode("utf-8"),
                        content_type="application/json",
                    )
                    return
                if path == "/video-frame.jpg":
                    try:
                        query = parse_qs(parsed_url.query)
                        epoch_index = int(query["epoch"][0])
                        frame_index = int(query["index"][0])
                        payload = owner.preview_video_frame(epoch_index, frame_index)
                    except (KeyError, TypeError, ValueError, RuntimeError) as exc:
                        self.send_error(HTTPStatus.BAD_REQUEST, str(exc))
                        return
                    self._send_image(payload)
                    return
                self.send_error(HTTPStatus.NOT_FOUND)

            def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
                path = urlsplit(self.path).path
                if path not in {"/save-reference", "/select-epoch"}:
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                try:
                    content_length = int(self.headers.get("Content-Length", "0"))
                    if not 0 < content_length <= 4096:
                        raise ValueError("Invalid request body size.")
                    request = json.loads(self.rfile.read(content_length))
                    if path == "/save-reference":
                        result = owner.save_reference_frame(
                            int(request["epoch"]),
                            int(request["frame_index"]),
                        )
                    else:
                        result = owner.select_calibration_epoch(int(request["epoch"]))
                except (
                    KeyError,
                    TypeError,
                    ValueError,
                    RuntimeError,
                    OSError,
                    json.JSONDecodeError,
                ) as exc:
                    self._send_json(
                        {"error": str(exc)},
                        status=HTTPStatus.BAD_REQUEST,
                    )
                    return
                self._send_json(result)

            def _send_image(self, payload: bytes | None) -> None:
                if payload is None:
                    self.send_error(HTTPStatus.SERVICE_UNAVAILABLE)
                    return
                self._send_bytes(payload, content_type="image/jpeg")

            def _send_bytes(self, payload: bytes, *, content_type: str) -> None:
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", "no-store, max-age=0")
                self.end_headers()
                self.wfile.write(payload)

            def _send_json(
                self,
                payload: dict[str, object],
                *,
                status: HTTPStatus = HTTPStatus.OK,
            ) -> None:
                encoded = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.send_header("Cache-Control", "no-store, max-age=0")
                self.end_headers()
                self.wfile.write(encoded)

            def log_message(self, _format: str, *_args) -> None:
                return

        try:
            server = ThreadingHTTPServer((self.host, self.port), Handler)
        except OSError as exc:
            raise RuntimeError(
                f"Could not start scene-alignment UI on {self.host}:{self.port}: {exc}"
            ) from exc
        server.daemon_threads = True
        thread = threading.Thread(
            target=server.serve_forever,
            name="scene-alignment-web-ui",
            daemon=True,
        )
        self._server = server
        self._server_thread = thread
        thread.start()

    def _snapshot(self, kind: str) -> bytes | None:
        with self._lock:
            if kind == "reference":
                return self._reference_jpeg
            return self._live_jpeg

    def _status_snapshot(self) -> dict[str, object]:
        with self._lock:
            return {
                "epoch": self._epoch_index,
                "epoch_offset": self.epoch_offset,
                "video_path": str(self._video_path) if self._video_path else None,
                "reference_path": (
                    str(self._reference_path) if self._reference_path else None
                ),
                "status": self._status,
                "live_ready": self._live_jpeg is not None,
                "frame_count": self._frame_count,
                "fps": self._fps,
                "selected_frame_index": self._selected_frame_index,
                "reference_saved": self._reference_saved,
                "reference_version": self._reference_version,
                "alignment_active": self._alignment_active,
                "frame_selection_enabled": self._frame_selection_enabled,
                "calibration_mode": bool(self._calibration_epochs),
                "calibration_epochs": self._calibration_epochs,
                "preset_name": (
                    self.reference_set.name if self.reference_set is not None else None
                ),
            }

    def _render_html(self) -> bytes:
        opacity = self.default_reference_opacity
        html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RLinf 场景对齐</title>
  <style>
    body {{ margin: 0; background: #111; color: #eee; font-family: sans-serif; }}
    main {{ max-width: 1040px; margin: 0 auto; padding: 18px; }}
    h1 {{ font-size: 22px; margin: 0 0 8px; }}
    #status {{ color: #9fe6ad; margin: 8px 0; }}
    #path {{ color: #aaa; font-size: 13px; overflow-wrap: anywhere; }}
    .stage {{ position: relative; width: min(96vw, 960px); margin-top: 14px; }}
    .stage img {{ display: block; width: 100%; height: auto; }}
    #reference {{ position: absolute; inset: 0; opacity: {opacity:.3f}; }}
    .controls {{ display: flex; align-items: center; gap: 12px; margin-top: 14px;
                 flex-wrap: wrap; }}
    input[type=range] {{ width: min(480px, 70vw); }}
    button {{ background: #2e7d4f; color: white; border: 0; border-radius: 5px;
              padding: 8px 14px; cursor: pointer; }}
    button:disabled {{ background: #555; cursor: wait; }}
    #saveState {{ color: #f0cf75; }}
    #calibrationNav {{ display: none; }}
    .hint {{ color: #ddd; margin-top: 12px; }}
  </style>
</head>
<body>
<main>
  <h1 id="pageTitle">Front 相机场景对齐</h1>
  <div id="status">等待实时画面……</div>
  <div id="path"></div>
  <div id="calibrationNav" class="controls">
    <button id="previousEpoch" type="button">上一个位置</button>
    <label for="epochSelect">标定位置</label>
    <select id="epochSelect"></select>
    <button id="nextEpoch" type="button">下一个位置</button>
    <span id="calibrationProgress"></span>
  </div>
  <div class="stage">
    <img id="live" alt="实时 front 画面">
    <img id="reference" src="/reference.jpg" alt="历史参考帧">
  </div>
  <div class="controls">
    <label for="opacity">参考帧透明度</label>
    <input id="opacity" type="range" min="0" max="1" step="0.01" value="{opacity:.3f}">
    <output id="opacityValue">{opacity:.2f}</output>
  </div>
  <div class="controls">
    <label for="frameSlider">本 epoch 参考帧</label>
    <input id="frameSlider" type="range" min="0" max="0" step="1" value="0">
    <output id="frameValue">0 / 0</output>
    <button id="saveFrame" type="button">保存为本 epoch 参考帧</button>
    <span id="saveState">正在读取视频信息……</span>
  </div>
  <div id="hint" class="hint">拖动滑块时只更新数字，松开后才解码预览。选好后点击保存，移动物块直至两层画面重合，再按物理键盘/踏板 a 开始。</div>
</main>
<script>
  const live = document.getElementById('live');
  const reference = document.getElementById('reference');
  const opacity = document.getElementById('opacity');
  const opacityValue = document.getElementById('opacityValue');
  const frameSlider = document.getElementById('frameSlider');
  const frameValue = document.getElementById('frameValue');
  const saveFrame = document.getElementById('saveFrame');
  const saveState = document.getElementById('saveState');
  const frameControls = frameSlider.closest('.controls');
  const opacityControls = opacity.closest('.controls');
  const calibrationNav = document.getElementById('calibrationNav');
  const epochSelect = document.getElementById('epochSelect');
  const previousEpoch = document.getElementById('previousEpoch');
  const nextEpoch = document.getElementById('nextEpoch');
  const calibrationProgress = document.getElementById('calibrationProgress');
  const pageTitle = document.getElementById('pageTitle');
  const hint = document.getElementById('hint');
  let activeEpoch = null;
  let activeFps = 30;
  let referenceVersion = null;
  let previewRequestId = 0;
  let dirtyFrame = false;
  let alignmentActive = false;
  let savingReference = false;
  let calibrationMode = false;
  let calibrationEpochs = [];

  opacity.addEventListener('input', () => {{
    reference.style.opacity = opacity.value;
    opacityValue.value = Number(opacity.value).toFixed(2);
  }});

  function updateFrameLabel() {{
    const frameIndex = Number(frameSlider.value);
    const frameCount = Number(frameSlider.max) + 1;
    const seconds = activeFps > 0 ? frameIndex / activeFps : 0;
    frameValue.value = `${{frameIndex}} / ${{Math.max(0, frameCount - 1)}} (${{seconds.toFixed(2)}} s)`;
  }}

  frameSlider.addEventListener('input', updateFrameLabel);
  frameSlider.addEventListener('change', () => {{
    if (activeEpoch === null || !alignmentActive) return;
    const frameIndex = Number(frameSlider.value);
    const requestId = ++previewRequestId;
    saveState.textContent = `正在读取第 ${{frameIndex}} 帧……`;
    const next = new Image();
    next.onload = () => {{
      if (requestId !== previewRequestId) return;
      reference.src = next.src;
      dirtyFrame = true;
      saveState.textContent = '当前预览尚未保存';
    }};
    next.onerror = () => {{
      if (requestId === previewRequestId) saveState.textContent = '该帧读取失败';
    }};
    next.src = `/video-frame.jpg?epoch=${{activeEpoch}}&index=${{frameIndex}}&t=${{Date.now()}}`;
  }});

  saveFrame.addEventListener('click', async () => {{
    if (activeEpoch === null || !alignmentActive) return;
    ++previewRequestId;
    savingReference = true;
    saveFrame.disabled = true;
    saveState.textContent = '正在保存……';
    try {{
      const response = await fetch('/save-reference', {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify({{
          epoch: activeEpoch,
          frame_index: Number(frameSlider.value),
        }}),
      }});
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || '保存失败');
      dirtyFrame = false;
      referenceVersion = result.reference_version;
      reference.src = `/reference.jpg?v=${{referenceVersion}}`;
      saveState.textContent = `已永久保存第 ${{result.frame_index}} 帧`;
    }} catch (error) {{
      saveState.textContent = error.message || '保存失败';
    }} finally {{
      savingReference = false;
      saveFrame.disabled = !alignmentActive;
    }}
  }});

  async function selectEpoch(epoch) {{
    if (!calibrationMode) return;
    previousEpoch.disabled = true;
    nextEpoch.disabled = true;
    saveState.textContent = `正在加载位置 ${{epoch}}……`;
    try {{
      const response = await fetch('/select-epoch', {{
        method: 'POST',
        headers: {{'Content-Type': 'application/json'}},
        body: JSON.stringify({{epoch}}),
      }});
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || '位置切换失败');
      activeEpoch = null;
      await refreshStatus();
    }} catch (error) {{
      saveState.textContent = error.message || '位置切换失败';
    }}
  }}
  epochSelect.addEventListener('change', () => selectEpoch(Number(epochSelect.value)));
  previousEpoch.addEventListener('click', () => {{
    const index = calibrationEpochs.indexOf(activeEpoch);
    if (index > 0) selectEpoch(calibrationEpochs[index - 1]);
  }});
  nextEpoch.addEventListener('click', () => {{
    const index = calibrationEpochs.indexOf(activeEpoch);
    if (index >= 0 && index + 1 < calibrationEpochs.length) {{
      selectEpoch(calibrationEpochs[index + 1]);
    }}
  }});

  function refreshLive() {{
    if (!alignmentActive || calibrationMode) return;
    const next = new Image();
    next.onload = () => {{ live.src = next.src; }};
    next.src = '/live.jpg?t=' + Date.now();
  }}
  async function refreshStatus() {{
    try {{
      const response = await fetch('/status.json?t=' + Date.now());
      const data = await response.json();
      document.getElementById('status').textContent = data.status;
      document.getElementById('path').textContent = data.video_path || data.reference_path || '';
      alignmentActive = Boolean(data.alignment_active);
      calibrationMode = Boolean(data.calibration_mode);
      calibrationEpochs = Array.from(data.calibration_epochs || [], Number);
      const frameSelectionEnabled = Boolean(data.frame_selection_enabled);
      frameControls.style.display = frameSelectionEnabled ? 'flex' : 'none';
      frameSlider.disabled = !alignmentActive || !frameSelectionEnabled;
      saveFrame.disabled = !alignmentActive || !frameSelectionEnabled || savingReference;
      calibrationNav.style.display = calibrationMode ? 'flex' : 'none';
      if (calibrationMode) {{
        pageTitle.textContent = '离线参考帧标定';
        hint.textContent = '逐个位置选择干净的视频帧并保存。此页面不会启动机械臂、相机、ROS 或 Ray。';
        live.style.display = 'none';
        reference.style.position = 'static';
        reference.style.opacity = '1';
        opacityControls.style.display = 'none';
        if (epochSelect.options.length !== calibrationEpochs.length) {{
          epochSelect.replaceChildren(...calibrationEpochs.map(epoch => {{
            const option = document.createElement('option');
            option.value = epoch;
            option.textContent = `位置 ${{epoch}}`;
            return option;
          }}));
        }}
        epochSelect.value = data.epoch;
        const epochPosition = calibrationEpochs.indexOf(data.epoch);
        previousEpoch.disabled = epochPosition <= 0;
        nextEpoch.disabled = epochPosition < 0 || epochPosition + 1 >= calibrationEpochs.length;
        calibrationProgress.textContent = `${{epochPosition + 1}} / ${{calibrationEpochs.length}}`;
      }} else if (!frameSelectionEnabled) {{
        pageTitle.textContent = 'Front 相机场景对齐';
        hint.textContent = '移动物块直至实时画面与半透明参考帧重合，再按物理键盘/踏板 a 开始本轮推理。';
      }}
      if (data.epoch !== activeEpoch) {{
        activeEpoch = data.epoch;
        activeFps = Number(data.fps) || 30;
        frameSlider.max = Math.max(0, Number(data.frame_count) - 1);
        frameSlider.value = Number(data.selected_frame_index) || 0;
        updateFrameLabel();
        dirtyFrame = false;
        referenceVersion = data.reference_version;
        reference.src = `/reference.jpg?v=${{referenceVersion}}`;
        saveState.textContent = data.reference_saved
          ? `已保存第 ${{data.selected_frame_index}} 帧`
          : '当前为临时第 0 帧，请选择并保存';
      }} else if (!dirtyFrame && data.reference_version !== referenceVersion) {{
        referenceVersion = data.reference_version;
        frameSlider.value = Number(data.selected_frame_index) || 0;
        updateFrameLabel();
        reference.src = `/reference.jpg?v=${{referenceVersion}}`;
        saveState.textContent = data.reference_saved
          ? `已保存第 ${{data.selected_frame_index}} 帧`
          : '参考帧尚未保存';
      }}
    }} catch (_error) {{}}
  }}
  setInterval(refreshLive, 80);
  setInterval(refreshStatus, 500);
  refreshStatus();
  refreshLive();
</script>
</body>
</html>"""
        return html.encode("utf-8")
