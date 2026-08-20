"""
capture/video_recorder.py

Continuous screen video recording service (optional continuous recording mode).

Features:
---------
- Records full-screen desktop video using OpenCV VideoWriter (MP4/mp4v codec) and MSS screen grabber.
- Time-segmented recording (default: 15-minute segments) to prevent file corruption on sudden crash/power loss.
- Sensible naming and organization: saved to data/videos/YYYY-MM-DD/<timestamp>_<app>_seg<index>.mp4.
- Unobtrusive background operation with start/pause/resume/stop controls.
- Safe resource management with automatic context flushing on exit.
"""
from __future__ import annotations

import logging
import platform
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import mss
import numpy as np

from capture.window_info import WindowInfoProvider
from config.settings import settings
from storage.file_manager import FileManager

logger = logging.getLogger(__name__)


class ScreenVideoRecorder:
    """
    Continuous screen video recorder with segmenting and background thread management.

    Args:
        file_manager:     FileManager for resolving storage paths.
        monitor_index:    MSS monitor index to record (1 = primary).
        fps:              Target recording framerate (e.g. 2.0 to 15.0 FPS).
        segment_minutes:  Length of each video segment in minutes before rolling over to a new file.
        codec:            FourCC codec string (default 'mp4v').
    """

    def __init__(
        self,
        file_manager: FileManager,
        monitor_index: int = 1,
        fps: float = 2.0,
        segment_minutes: int = 15,
        codec: str = "mp4v",
    ) -> None:
        self._file_manager = file_manager
        self._monitor_index = monitor_index
        self._fps = max(0.5, float(fps))
        self._segment_duration_seconds = max(30.0, float(segment_minutes * 60))
        self._codec = codec
        self._window_provider = WindowInfoProvider()

        # Threading & lifecycle events
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._lock = threading.Lock()

        # State tracking
        self.segments_recorded: int = 0
        self.total_frames_written: int = 0
        self.current_video_file: Optional[Path] = None
        self._is_active: bool = False

    def start(self) -> None:
        """Start the continuous screen video recording thread."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                logger.warning("ScreenVideoRecorder is already running.")
                return

            self._stop_event.clear()
            self._pause_event.clear()
            self._is_active = True
            self._thread = threading.Thread(
                target=self._recording_loop,
                name="ScreenVideoRecorderThread",
                daemon=True,
            )
            self._thread.start()
            logger.info(
                "ScreenVideoRecorder started — monitor=%d  fps=%.1f  segment_mins=%d  codec=%s",
                self._monitor_index,
                self._fps,
                int(self._segment_duration_seconds / 60),
                self._codec,
            )

    def stop(self) -> None:
        """Stop the video recording thread cleanly and release video resources."""
        with self._lock:
            if not self._thread:
                return

            logger.info("ScreenVideoRecorder stopping...")
            self._stop_event.set()

        if self._thread.is_alive():
            self._thread.join(timeout=10)

        with self._lock:
            self._thread = None
            self._is_active = False
            self.current_video_file = None
            logger.info(
                "ScreenVideoRecorder stopped cleanly. Segments: %d, Frames: %d",
                self.segments_recorded,
                self.total_frames_written,
            )

    def pause(self) -> None:
        """Pause video recording without tearing down the worker thread."""
        self._pause_event.set()
        logger.info("ScreenVideoRecorder paused.")

    def resume(self) -> None:
        """Resume video recording."""
        self._pause_event.clear()
        logger.info("ScreenVideoRecorder resumed.")

    @property
    def is_recording(self) -> bool:
        return self._is_active and not self._pause_event.is_set()

    @property
    def is_paused(self) -> bool:
        return self._pause_event.is_set()

    def _grab_frame(self, sct: mss.mss, monitor: dict) -> Optional[np.ndarray]:
        """Captures a single BGR frame with fallback protection."""
        try:
            raw = sct.grab(monitor)
            return np.array(raw)[:, :, :3]
        except Exception as exc:  # noqa: BLE001
            logger.debug("Video recorder MSS grab failed: %s", exc)
            return None

    def _recording_loop(self) -> None:
        """Main recording loop running on dedicated background thread."""
        try:
            if platform.system() == "Windows":
                import ctypes
                ctypes.windll.user32.SetProcessDPIAware()  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass

        frame_interval = 1.0 / self._fps
        fourcc = cv2.VideoWriter_fourcc(*self._codec)  # type: ignore[attr-defined]

        try:
            with mss.mss() as sct:
                monitors = sct.monitors
                if self._monitor_index >= len(monitors):
                    self._monitor_index = 1
                monitor = monitors[self._monitor_index]
                width = int(monitor["width"])
                height = int(monitor["height"])

                writer: Optional[cv2.VideoWriter] = None
                segment_start_time = 0.0
                segment_index = 0

                while not self._stop_event.is_set():
                    tick_start = time.monotonic()

                    # Handle pause state
                    if self._pause_event.is_set():
                        self._stop_event.wait(timeout=0.2)
                        continue

                    # Check if we need to open a new segment file
                    now_mono = time.monotonic()
                    if writer is None or (now_mono - segment_start_time >= self._segment_duration_seconds):
                        if writer is not None:
                            writer.release()
                            self.segments_recorded += 1

                        segment_start_time = now_mono
                        segment_index += 1
                        ts_now = datetime.now(timezone.utc)
                        win = self._window_provider.get()
                        app_name = win.application if win else None

                        video_path = self._file_manager.get_video_path(
                            timestamp=ts_now,
                            segment_index=segment_index,
                            application=app_name,
                        )
                        self.current_video_file = video_path
                        writer = cv2.VideoWriter(
                            str(video_path),
                            fourcc,
                            self._fps,
                            (width, height),
                        )
                        logger.info("Opened new video segment: %s", video_path.name)

                    # Grab and write screen frame
                    frame = self._grab_frame(sct, monitor)
                    if frame is not None and writer is not None and writer.isOpened():
                        # Resize if necessary to match initial dimensions
                        if frame.shape[1] != width or frame.shape[0] != height:
                            frame = cv2.resize(frame, (width, height))
                        writer.write(frame)
                        self.total_frames_written += 1

                    # Sleep to maintain target FPS
                    elapsed = time.monotonic() - tick_start
                    sleep_time = max(0.001, frame_interval - elapsed)
                    self._stop_event.wait(timeout=sleep_time)

                # Clean up writer at end of loop
                if writer is not None:
                    writer.release()
                    self.segments_recorded += 1
                    logger.info("Final video segment written and closed.")

        except Exception as exc:  # noqa: BLE001
            logger.exception("Fatal error in ScreenVideoRecorder: %s", exc)
        finally:
            self._is_active = False
