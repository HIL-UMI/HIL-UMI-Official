# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
# Modified for HIL-UMI.

import os
import queue
import threading
import warnings

import cv2
import numpy as np


class VideoPlayer:
    def __init__(self, enable: bool = True):
        self.queue = queue.Queue()
        self.is_running = False
        if not enable:
            return
        self._run_thread = threading.Thread(target=self._play, daemon=True)
        self._run_thread.start()

    def put_frame(self, frame):
        # Bound the queue so a slow or dead consumer never grows it unboundedly.
        if self.is_running and self.queue.qsize() < 5:
            self.queue.put(frame)

    def _play(self):
        if os.environ.get("DISPLAY") is None:
            warnings.warn(
                "No display found. VideoPlayer will not run. Set DISPLAY environment variable to enable."
            )
            return

        self.is_running = True
        try:
            while True:
                img_array = self.queue.get()  # retrieve an image from the queue
                if img_array is None:  # None is our signal to exit
                    break

                frame = np.concatenate(
                    [v for k, v in img_array.items() if "full" not in k], axis=0
                )

                cv2.imshow("Cameras", frame)
                cv2.waitKey(1)
        except Exception as exc:
            warnings.warn(
                f"VideoPlayer display failed ({exc}); disabling camera preview."
            )
        finally:
            # is_running must be reset so put_frame stops enqueueing after the
            # display loop dies (e.g. headless OpenCV without GUI support).
            self.is_running = False
