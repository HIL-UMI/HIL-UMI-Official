#!/usr/bin/env python3
# Modified for HIL-UMI.
# Copyright 2026 The HIL-UMI Authors.
# SPDX-License-Identifier: Apache-2.0

"""Apply the OOD integration patch to data_recorder's runtime modules.

Compatibility is limited to user-supplied data_recorder 0.1.21 wheels.
The source wheel is preserved and no external package is downloaded or installed.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path
from zipfile import ZipFile


def patch_recorder_py(path: str) -> None:
    file_path = Path(path)
    src = file_path.read_text(encoding="utf-8")

    # 1. Add imports near the top.
    import_block = "import os\n"
    if "import socket\n" not in src:
        src = src.replace(
            import_block,
            import_block + "import json\nimport socket\n",
            1,
        )

    # 2. Initialize the OOD broadcast socket early in the main() setup.
    marker_setup = "    shm_info: dict[str, dict[str, tuple[str, int, int]]] = {}\n"
    if marker_setup not in src:
        raise RuntimeError("Cannot find recorder setup marker; patch aborted.")

    init_block = """    shm_info: dict[str, dict[str, tuple[str, int, int]]] = {}
    writer_proc = None
    monitor = NullUI()
    kb = None
    _ood_socket_path = os.environ.get("COLLECTION_OOD_SOCKET_PATH", "")
    _ood_socket = None
    if _ood_socket_path:
        try:
            _ood_socket = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
            _ood_socket.setblocking(False)
        except Exception:
            _ood_socket = None
"""
    if "_ood_socket_path" not in src:
        src = src.replace(marker_setup, init_block, 1)

    # 3. Broadcast state next to the writer queue push. Later normalization
    # decouples the two sinks so OOD also receives idle samples.
    marker_push = "                _put_drop_oldest(state_queue, state_data)\n"
    already_guarded_push = """                if recording:
                    _put_drop_oldest(state_queue, state_data)
"""
    if marker_push not in src and already_guarded_push not in src:
        raise RuntimeError("Cannot find state_queue push marker; patch aborted.")

    broadcast_block = """                _put_drop_oldest(state_queue, state_data)
                if _ood_socket is not None and _ood_socket_path:
                    try:
                        _ood_payload = {
                            "sys_ts": state_data.get("sys_ts"),
                            "oculus_sys_ts": state_data.get("oculus_sys_ts"),
                            "oculus_recv_sys_ts": state_data.get("oculus_recv_sys_ts"),
                            "oculus_poll_sys_ts": state_data.get("oculus_poll_sys_ts"),
                            "head_pose": (
                                state_data["head_pose"].tolist()
                                if state_data.get("head_pose") is not None
                                else None
                            ),
                            "right_pose": (
                                state_data["right_pose"].tolist()
                                if state_data.get("right_pose") is not None
                                else None
                            ),
                            "left_pose": (
                                state_data["left_pose"].tolist()
                                if state_data.get("left_pose") is not None
                                else None
                            ),
                            "buttons": state_data.get("buttons"),
                            "gripper_cmd_right": state_data.get("gripper_cmd_right"),
                            "gripper_cmd_left": state_data.get("gripper_cmd_left"),
                            "gripper_cmd_right_pos_pct": state_data.get("gripper_cmd_right_pos_pct"),
                            "gripper_cmd_left_pos_pct": state_data.get("gripper_cmd_left_pos_pct"),
                            "gripper_cmd_right_vel_pct": state_data.get("gripper_cmd_right_vel_pct"),
                            "gripper_cmd_left_vel_pct": state_data.get("gripper_cmd_left_vel_pct"),
                            "gripper_cmd_right_force_pct": state_data.get("gripper_cmd_right_force_pct"),
                            "gripper_cmd_left_force_pct": state_data.get("gripper_cmd_left_force_pct"),
                            "gripper_cmd_right_acc_pct": state_data.get("gripper_cmd_right_acc_pct"),
                            "gripper_cmd_left_acc_pct": state_data.get("gripper_cmd_left_acc_pct"),
                            "gripper_cmd_right_dec_pct": state_data.get("gripper_cmd_right_dec_pct"),
                            "gripper_cmd_left_dec_pct": state_data.get("gripper_cmd_left_dec_pct"),
                            "gripper_state_right": state_data.get("gripper_state_right"),
                            "gripper_state_left": state_data.get("gripper_state_left"),
                        }
                        _ood_socket.sendto(
                            json.dumps(_ood_payload).encode("utf-8"),
                            _ood_socket_path,
                        )
                    except Exception:
                        # Broadcasting to OOD monitor must never stall recording.
                        pass
"""
    if "_ood_payload" not in src:
        src = src.replace(marker_push, broadcast_block, 1)

    # 3a. Build state for every new Oculus sample. Only the dataset writer queue
    # remains recording-gated; the OOD datagram is emitted in both modes.
    recording_state_gate = "            if recording and is_new_oculus_sample:\n"
    continuous_state_gate = "            if is_new_oculus_sample:\n"
    if recording_state_gate in src:
        src = src.replace(recording_state_gate, continuous_state_gate, 1)
    elif continuous_state_gate not in src:
        raise RuntimeError("Cannot find Oculus state emission gate; patch aborted.")

    unguarded_writer_push = (
        "                _put_drop_oldest(state_queue, state_data)\n"
    )
    guarded_writer_push = already_guarded_push
    if guarded_writer_push in src:
        pass
    elif unguarded_writer_push in src:
        src = src.replace(unguarded_writer_push, guarded_writer_push, 1)
    else:
        raise RuntimeError("Cannot guard dataset state queue push; patch aborted.")

    unguarded_state_counter = """                monitor.inc_state_pkt()
                state_packet_emitted = True
"""
    guarded_state_counter = """                if recording:
                    monitor.inc_state_pkt()
                    state_packet_emitted = True
"""
    if guarded_state_counter in src:
        pass
    elif unguarded_state_counter in src:
        src = src.replace(unguarded_state_counter, guarded_state_counter, 1)
    else:
        raise RuntimeError("Cannot guard dataset state metrics; patch aborted.")

    # 3b. Tag state packets with episode identity. The run root is authoritative
    # once the writer has created it; early packets may legitimately carry null.
    state_payload_marker = """                        _ood_payload = {
                            "sys_ts": state_data.get("sys_ts"),
"""
    state_payload_replacement = """                        _ood_payload = {
                            "message_type": "state",
                            "episode": episode_idx,
                            "recording": recording,
                            "writer_state": writer_state,
                            "run_root": (
                                str(active_run_root_ref[0])
                                if active_run_root_ref[0] is not None
                                else None
                            ),
                            "sys_ts": state_data.get("sys_ts"),
"""
    if '"message_type": "state"' not in src:
        if state_payload_marker not in src:
            raise RuntimeError("Cannot find OOD state payload marker; patch aborted.")
        src = src.replace(state_payload_marker, state_payload_replacement, 1)

    # 3c. Evaluate the exact same configured gate used by the A-button start
    # path and publish its decision with every OOD state packet.
    payload_try_marker = """                    try:
                        _ood_payload = {
"""
    payload_gate_block = """                    try:
                        _ood_gate = evaluate_start_gate(
                            oculus_connected=oculus_mgr.is_connected(),
                            oculus_detail=oculus_mgr.detail,
                            start_require_all_valid=start_require_all_valid,
                            validity=raw_validity,
                            start_require_all_devices=start_require_all_devices,
                            device_summary=device_summary,
                            device_detail=health.format_gate_detail(device_summary),
                        )
                        _ood_payload = {
"""
    if "_ood_gate = evaluate_start_gate(" not in src:
        if payload_try_marker not in src:
            raise RuntimeError("Cannot find OOD payload try block; patch aborted.")
        src = src.replace(payload_try_marker, payload_gate_block, 1)

    gate_result_marker = """                            device_detail=health.format_gate_detail(device_summary),
                        )
                        _ood_payload = {
"""
    complete_gate_result = """                            device_detail=health.format_gate_detail(device_summary),
                        )
                        _ood_adb_serial = str(
                            oculus_identity.get("adb_serial") or ""
                        ).strip()
                        _ood_ready = bool(_ood_gate.allowed and _ood_adb_serial)
                        _ood_payload = {
"""
    if "_ood_ready = bool(_ood_gate.allowed and _ood_adb_serial)" not in src:
        if gate_result_marker not in src:
            raise RuntimeError("Cannot complete OOD hardware gate; patch aborted.")
        src = src.replace(gate_result_marker, complete_gate_result, 1)

    readiness_marker = """                            "recording": recording,
                            "writer_state": writer_state,
"""
    legacy_readiness = """                            "recording": recording,
                            "writer_state": writer_state,
                            "ood_ready": bool(_ood_gate.allowed),
                            "ood_block_reason": (
                                "" if _ood_gate.allowed else str(_ood_gate.reason or "not_ready")
                            ),
                            "ood_block_detail": (
                                "" if _ood_gate.allowed else str(_ood_gate.detail or "")
                            ),
"""
    readiness_replacement = """                            "recording": recording,
                            "writer_state": writer_state,
                            "ood_ready": bool(_ood_ready),
                            "ood_block_reason": (
                                ""
                                if _ood_ready
                                else (
                                    str(_ood_gate.reason or "not_ready")
                                    if not _ood_gate.allowed
                                    else "oculus_identity_unavailable"
                                )
                            ),
                            "ood_block_detail": (
                                ""
                                if _ood_ready
                                else (
                                    str(_ood_gate.detail or "")
                                    if not _ood_gate.allowed
                                    else "Oculus adb_serial is unavailable"
                                )
                            ),
"""
    if legacy_readiness in src:
        src = src.replace(legacy_readiness, readiness_replacement, 1)
    elif '"ood_ready": bool(_ood_ready)' not in src:
        if readiness_marker not in src:
            raise RuntimeError(
                "Cannot find OOD readiness payload marker; patch aborted."
            )
        src = src.replace(readiness_marker, readiness_replacement, 1)

    # 3d. Forward writer lifecycle events. These provide authoritative
    # recording/saved/discarded boundaries and the dataset run root.
    lifecycle_marker = """            if not isinstance(event, dict):
                continue

            event_episode = None
"""
    lifecycle_block = """            if not isinstance(event, dict):
                continue

            if _ood_socket is not None and _ood_socket_path:
                try:
                    _ood_lifecycle_payload = dict(event)
                    _ood_lifecycle_payload["message_type"] = "lifecycle"
                    _ood_lifecycle_payload["timestamp"] = time.time()
                    _ood_socket.sendto(
                        json.dumps(_ood_lifecycle_payload).encode("utf-8"),
                        _ood_socket_path,
                    )
                except Exception:
                    pass

            event_episode = None
"""
    if "_ood_lifecycle_payload" not in src:
        if lifecycle_marker not in src:
            raise RuntimeError("Cannot find writer lifecycle marker; patch aborted.")
        src = src.replace(lifecycle_marker, lifecycle_block, 1)

    # 4. Close the socket during cleanup.
    marker_cleanup = '        logger.info("Parent cleanup complete")\n'
    if marker_cleanup not in src:
        raise RuntimeError("Cannot find cleanup marker; patch aborted.")

    cleanup_block = """        if _ood_socket is not None:
            try:
                _ood_socket.close()
            except Exception:
                pass
        logger.info("Parent cleanup complete")
"""
    if cleanup_block not in src:
        src = src.replace(marker_cleanup, cleanup_block, 1)

    file_path.write_text(src, encoding="utf-8")
    print(f"Patched {file_path}")


def patch_camera_worker_py(path: str) -> None:
    """Keep RGB shared memory live while dataset publication stays gated."""
    file_path = Path(path)
    src = file_path.read_text(encoding="utf-8")

    idle_gate = """            if not capture_event.is_set():
                try:
                    metrics_queue.put_nowait({"source": f"cam_{cam_name}", "count": 1})
                except Exception:
                    pass
                continue

"""
    old_idle_comment = (
        "            # Keep camera streaming alive, but gate recording writes, "
        "\n            # i.e. not write to ring and not publish to image_queue. \n"
    )
    src = src.replace(old_idle_comment, "", 1)
    continuous_rgb_gate = """            # RGB shared memory is a live sensor surface shared with the OOD
            # monitor. Dataset metadata/depth publication remains capture-gated.
            capture_active = bool(capture_event.is_set())

"""
    if idle_gate in src:
        src = src.replace(idle_gate, continuous_rgb_gate, 1)
    elif continuous_rgb_gate not in src:
        raise RuntimeError("Cannot find camera capture gate; patch aborted.")

    depth_gate = "            if depth_frame is not None and depth_ring is not None:\n"
    gated_depth = "            if capture_active and depth_frame is not None and depth_ring is not None:\n"
    if depth_gate in src:
        src = src.replace(depth_gate, gated_depth, 1)
    elif gated_depth not in src:
        raise RuntimeError("Cannot gate idle depth ring writes; patch aborted.")

    publication_block = """            _put_drop_oldest(
                image_queue,
                {
                    "cam": cam_name,
                    "frame_id": int(frame_id),
                    "hw_ts": hw_ts,
                    "sys_ts": sys_ts,
                    "rgb": rgb_meta,
                    "depth": depth_meta,
                },
            )

            try:
                metrics_queue.put_nowait({"source": f"cam_{cam_name}", "count": 1})
            except Exception:
                pass

            try:
                metrics_queue.put_nowait({"source": f"run_cam_{cam_name}", "count": 1})
            except Exception:
                pass

            slot_idx += 1
"""
    gated_publication_block = """            try:
                metrics_queue.put_nowait({"source": f"cam_{cam_name}", "count": 1})
            except Exception:
                pass

            if capture_active:
                _put_drop_oldest(
                    image_queue,
                    {
                        "cam": cam_name,
                        "frame_id": int(frame_id),
                        "hw_ts": hw_ts,
                        "sys_ts": sys_ts,
                        "rgb": rgb_meta,
                        "depth": depth_meta,
                    },
                )
                try:
                    metrics_queue.put_nowait(
                        {"source": f"run_cam_{cam_name}", "count": 1}
                    )
                except Exception:
                    pass

            slot_idx += 1
"""
    if publication_block in src:
        src = src.replace(publication_block, gated_publication_block, 1)
    elif gated_publication_block not in src:
        raise RuntimeError("Cannot gate camera writer publication; patch aborted.")

    file_path.write_text(src, encoding="utf-8")
    print(f"Patched {file_path}")


def patch_wheel(source: Path, output_dir: Path) -> Path:
    """Create a patched wheel without replacing the source or existing output."""
    source = source.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    repository = Path(__file__).resolve().parents[2]
    ignored_output = repository / "collection" / ".local"
    if output_dir.is_relative_to(repository) and not output_dir.is_relative_to(
        ignored_output
    ):
        raise ValueError(
            "In-repository patch output must be under ignored collection/.local"
        )
    expected = "data_recorder-0.1.21-py3-none-any.whl"
    if source.name != expected or not source.is_file():
        raise ValueError(f"Supply your external {expected} wheel")
    output = output_dir / expected
    if output == source:
        raise ValueError("Output must differ from the input wheel directory")
    if output.exists():
        raise FileExistsError(f"Refusing to replace existing output: {output}")
    with tempfile.TemporaryDirectory(prefix="hil-umi-recorder-") as directory:
        unpacked = Path(directory) / "unpacked"
        unpacked.mkdir()
        with ZipFile(source) as wheel:
            for member in wheel.namelist():
                if not (unpacked / member).resolve().is_relative_to(unpacked):
                    raise ValueError("Wheel contains an unsafe archive path")
            metadata = wheel.read("data_recorder-0.1.21.dist-info/METADATA").decode(
                "utf-8"
            )
            if "Version: 0.1.21" not in metadata.splitlines():
                raise ValueError(
                    "Wheel metadata does not identify data_recorder 0.1.21"
                )
            wheel.extractall(unpacked)
        patch_recorder_py(str(unpacked / "data_recorder" / "recorder.py"))
        patch_camera_worker_py(str(unpacked / "data_recorder" / "camera_worker.py"))
        # Regenerate RECORD hashes using the standard wheel tool. No install.
        packed = Path(directory) / "packed"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "wheel",
                "pack",
                str(unpacked),
                "--dest-dir",
                str(packed),
            ],
            check=True,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        # Exclusive creation protects an existing result even if another patch
        # process produces the same target between the earlier check and now.
        with output.open("xb") as target:
            target.write((packed / expected).read_bytes())
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    print(f"Patched wheel: {output}\nSHA256: {digest}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    patch_wheel(args.wheel, args.output_dir)


if __name__ == "__main__":
    main()
