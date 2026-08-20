"""
capture/screen_recorder.py

Continuous screen-capture service — Stage 1 of the Ambient Screen
Understanding pipeline.

Responsibilities
----------------
- Capture the configured monitor at a fixed interval.
- Call ChangeDetector.analyse() to determine whether to save the frame
  and record WHY it was saved (visual_change, application_change,
  periodic_capture, manual_capture).
- Read the active window application + title.
- Save accepted frames as compressed WebP via FileManager.
- Insert a CaptureRecord into the SQLite database.
- Emit the completed CaptureRecord to an optional output queue for
  downstream consumers (OCR stage, UI).
- Handle all errors gracefully; never crash the background thread.
- Support start / pause / resume / stop / force_capture controls from
  any thread.
- Log structured progress at configurable verbosity.

Threading model
---------------
A single daemon thread runs the capture loop.  All public methods are
safe to call from any thread.  Internal state is protected by
threading.Event objects and ChangeDetector's internal lock.

Error handling
--------------
Individual capture errors are caught, logged at WARNING level, and the
loop continues.  Three consecutive failures trigger a 30 s back-off
sleep, then the loop resumes automatically.

Long-running stability
----------------------
- MSS context is opened once and reused (avoids per-frame OS handles).
- The loop uses threading.Event.wait() so stop() wakes immediately.
- Memory: only one numpy array is held in memory at a time.
- Stats (captured, skipped, errored, by reason) maintained for UI/log.
"""
from __future__ import annotations

import logging
import queue
import threading
from collections import Counter
from datetime import datetime, timezone
from typing import Callable, Optional

import mss
import numpy as np

from capture.change_detector import ChangeDetector, CaptureReason, FrameAnalysis
from capture.models import CaptureRecord
from capture.window_info import WindowInfoProvider
from config.settings import settings
from storage.db import Database
from storage.file_manager import FileManager

logger = logging.getLogger(__name__)


class ScreenRecorder:
    """
    Continuous screen-capture service.

    Args:
        db:                       Database instance for persistence.
        file_manager:             FileManager for saving WebP images.
        output_queue:             Optional queue receiving each CaptureRecord.
        monitor_index:            MSS monitor index (1 = primary).
        interval_seconds:         Seconds between capture attempts.
        change_threshold:         Visual pixel-diff threshold [0–1].
        max_capture_interval:     Seconds before a periodic forced capture.
                                  0 = disabled.
        idle_threshold_seconds:   Seconds of idle before logging idle state.
                                  0 = disabled.
        on_capture:               Optional callback(CaptureRecord) called after
                                  each successful save (on the capture thread).
    """

    def __init__(
        self,
        db: Database,
        file_manager: FileManager,
        output_queue: Optional[queue.Queue] = None,
        monitor_index: int = 1,
        interval_seconds: float = 5.0,
        change_threshold: float = 0.02,
        max_capture_interval: float = 300.0,
        idle_threshold_seconds: float = 0.0,
        on_capture: Optional[Callable[[CaptureRecord], None]] = None,
    ) -> None:
        self._db = db
        self._file_manager = file_manager
        self._queue = output_queue
        self._monitor_index = monitor_index
        self._interval = interval_seconds
        self._on_capture = on_capture

        self._detector = ChangeDetector(
            visual_threshold=change_threshold,
            resize_to=(settings.capture_resize_width, settings.capture_resize_height),
            max_capture_interval=max_capture_interval,
            idle_threshold_seconds=idle_threshold_seconds,
        )
        self._window_provider = WindowInfoProvider()

        # Thread control
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._force_capture_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Statistics — writes only from capture thread; safe to read from any thread
        self.frames_captured: int = 0
        self.frames_skipped: int = 0
        self.frames_errored: int = 0
        self.reason_counts: Counter = Counter()  # {CaptureReason.value: int}

    # ------------------------------------------------------------------
    # Public control interface (thread-safe)
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the capture thread.  No-op if already running."""
        if self._thread and self._thread.is_alive():
            logger.warning("ScreenRecorder.start() called but thread is already alive.")
            return
        self._stop_event.clear()
        self._pause_event.clear()
        self._detector.reset()
        self._detector.force_next()
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="ScreenRecorder",
        )
        self._thread.start()
        logger.info(
            "ScreenRecorder started — monitor=%d  interval=%.1fs  threshold=%.3f",
            self._monitor_index,
            self._interval,
            self._detector._visual_threshold,
        )

    def stop(self) -> None:
        """Signal the capture thread to stop and block until it exits."""
        if not self._thread:
            return
        logger.info("ScreenRecorder stopping…")
        self._stop_event.set()
        self._thread.join(timeout=15)
        if self._thread.is_alive():
            logger.warning("ScreenRecorder thread did not exit cleanly within timeout.")
        else:
            logger.info(
                "ScreenRecorder stopped. captured=%d  skipped=%d  errored=%d  by_reason=%s",
                self.frames_captured,
                self.frames_skipped,
                self.frames_errored,
                dict(self.reason_counts),
            )

    def pause(self) -> None:
        """Pause captures without stopping the thread."""
        self._pause_event.set()
        logger.info("ScreenRecorder paused.")

    def resume(self) -> None:
        """Resume captures after a pause."""
        self._pause_event.clear()
        # Reset the detector's keyframe so the next frame is always saved
        self._detector.reset()
        logger.info("ScreenRecorder resumed.")

    def force_capture(self) -> None:
        """
        Request a manual capture on the next tick, regardless of visual diff.
        Thread-safe.  The saved record will have reason=MANUAL_CAPTURE.
        """
        self._detector.force_next()
        self._force_capture_event.set()
        logger.debug("Manual capture requested.")

    @property
    def is_running(self) -> bool:
        """True if the capture thread is alive and not paused."""
        return (
            self._thread is not None
            and self._thread.is_alive()
            and not self._pause_event.is_set()
        )

    @property
    def is_paused(self) -> bool:
        return self._pause_event.is_set()

    # ------------------------------------------------------------------
    # Internal capture loop
    # ------------------------------------------------------------------

    def _run(self) -> None:
        """Main loop — runs on the dedicated capture thread."""
        consecutive_errors = 0

        # Enable DPI awareness on Windows to prevent BitBlt failures.
        # Must be called before creating the MSS context.
        try:
            import ctypes
            import platform
            if platform.system() == "Windows":
                ctypes.windll.user32.SetProcessDPIAware()  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass

        try:
            with mss.mss() as sct:
                monitors = sct.monitors
                if self._monitor_index >= len(monitors):
                    logger.error(
                        "Monitor index %d is out of range (%d monitors available). "
                        "Falling back to primary (index 1).",
                        self._monitor_index,
                        len(monitors) - 1,
                    )
                    self._monitor_index = 1

                monitor = monitors[self._monitor_index]
                logger.debug(
                    "Capturing monitor %d: %dx%d",
                    self._monitor_index,
                    monitor["width"],
                    monitor["height"],
                )

                while not self._stop_event.is_set():
                    # ── Pause handling ──────────────────────────────────
                    if self._pause_event.is_set():
                        self._stop_event.wait(timeout=0.5)
                        continue

                    # ── Back-off after repeated errors ──────────────────
                    if consecutive_errors >= 3:
                        logger.warning(
                            "3 consecutive capture errors. Sleeping 30s before retry."
                        )
                        self._stop_event.wait(timeout=30)
                        consecutive_errors = 0
                        continue

                    # ── Capture attempt ─────────────────────────────────
                    try:
                        record = self._capture_one(sct, monitor)
                        if record is not None:
                            consecutive_errors = 0
                            self.frames_captured += 1
                            self.reason_counts[record.capture_reason.value] += 1
                            logger.info(
                                "Frame saved: id=%s  reason=%s  diff=%.4f  app=%r  title=%r",
                                record.id,
                                record.capture_reason.value,
                                record.diff_score,
                                record.application,
                                record.window_title,
                            )
                            if self._on_capture:
                                try:
                                    self._on_capture(record)
                                except Exception as cb_exc:  # noqa: BLE001
                                    logger.warning("on_capture callback raised: %s", cb_exc)
                            if self._queue is not None:
                                try:
                                    self._queue.put_nowait(record)
                                except queue.Full:
                                    logger.debug("Output queue full — dropping record.")
                        else:
                            self.frames_skipped += 1

                    except Exception as exc:  # noqa: BLE001
                        consecutive_errors += 1
                        self.frames_errored += 1
                        logger.warning(
                            "Capture error (%d/3): %s",
                            consecutive_errors,
                            exc,
                            exc_info=True,
                        )

                    # ── Clear force-capture flag ────────────────────────
                    self._force_capture_event.clear()

                    # ── Sleep until next interval or stop/force signal ───
                    # We wake early if force_capture() is called.
                    self._stop_event.wait(timeout=self._interval)

        except Exception as fatal:  # noqa: BLE001
            logger.exception("ScreenRecorder fatal error — thread exiting: %s", fatal)

    def _grab_frame_pixels(self, sct: Optional["mss.mss"], monitor: dict) -> np.ndarray:
        """
        Grabs screen pixels with multi-backend resilience:
        1. MSS screen grab (fastest, thread-safe, native multi-monitor)
        2. PIL.ImageGrab.grab fallback (thread-safe native Windows GDI/Desktop grab)
        3. PySide6 QGuiApplication grabWindow (main thread fallback)
        """
        # Strategy 1: MSS Grab (fastest and safe on background threads)
        if sct is not None:
            try:
                raw = sct.grab(monitor)
                return np.array(raw)[:, :, :3]
            except Exception as exc:  # noqa: BLE001
                logger.debug("MSS grab failed, falling back to PIL: %s", exc)

        # Strategy 2: PIL ImageGrab
        try:
            from PIL import ImageGrab
            import cv2
            img = ImageGrab.grab()
            return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        except Exception as exc:  # noqa: BLE001
            logger.debug("PIL ImageGrab failed, falling back to Qt: %s", exc)

        # Strategy 3: Qt Application Screen Grab
        try:
            from PySide6.QtGui import QGuiApplication, QImage
            from PySide6.QtWidgets import QApplication
            app = QApplication.instance()
            if app is not None:
                screen = QGuiApplication.primaryScreen()
                if screen is not None:
                    pix = screen.grabWindow(0)
                    if not pix.isNull():
                        qimg = pix.toImage().convertToFormat(QImage.Format.Format_RGBA8888)
                        ptr = qimg.bits()
                        arr = np.frombuffer(ptr, dtype=np.uint8).reshape((qimg.height(), qimg.width(), 4))
                        import cv2
                        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Qt grab fallback: %s", exc)

        raise RuntimeError("All screen capture backends failed")

    def _capture_one(
        self,
        sct: "mss.mss",
        monitor: dict,
    ) -> Optional[CaptureRecord]:
        """
        Capture one frame, run smart analysis, save if warranted.

        Returns CaptureRecord on save, None if the frame was dropped.
        """
        ts = datetime.now(timezone.utc)

        # ── Grab pixels with multi-backend resilience ───────────────────
        frame_bgr: np.ndarray = self._grab_frame_pixels(sct, monitor)
        h, w = frame_bgr.shape[:2]

        # ── Window metadata (read before analysis for app-change check) ──
        win = self._window_provider.get()
        app_name: Optional[str] = win.application if win else None
        win_title: Optional[str] = win.window_title if win else None

        # ── Ignore self-window (RIOM Dashboard) to prevent recursive self-inspection ──
        if win_title and ("RIOM — AI Work Memory Dashboard" in win_title or (app_name and "python" in app_name.lower() and "RIOM" in win_title)):
            logger.debug("Skipping capture of RIOM self-window: %s", win_title)
            return None

        # ── Smart analysis ───────────────────────────────────────────────
        # Handle manual-capture override: force_next() was already called
        # on the detector, so analyse() will return VISUAL_CHANGE with
        # diff=1.0.  We override the reason to MANUAL_CAPTURE here.
        is_manual = self._force_capture_event.is_set()

        analysis: FrameAnalysis = self._detector.analyse(
            frame=frame_bgr,
            application=app_name,
        )

        if is_manual and analysis.should_save:
            # Relabel the reason so the record is correctly tagged
            from dataclasses import replace as dc_replace
            analysis = dc_replace(analysis, reason=CaptureReason.MANUAL_CAPTURE)

        # ── Log idle state if detected ───────────────────────────────────
        if analysis.reason == CaptureReason.IDLE:
            logger.debug("Screen idle (diff=%.4f).", analysis.diff_score)

        # ── Drop frame if not saving ─────────────────────────────────────
        if not analysis.should_save:
            logger.debug(
                "Frame dropped: reason=%s  diff=%.4f",
                analysis.reason.value,
                analysis.diff_score,
            )
            return None

        # ── Save image to disk ──────────────────────────────────────────
        rel_path = self._file_manager.save_frame(
            frame=frame_bgr,
            timestamp=ts,
            frame_id=None,   # Temporary name; renamed after DB insert
            application=app_name,
            window_title=win_title,
        )

        # ── Insert into database ────────────────────────────────────────
        frame_id = self._db.insert_frame(
            captured_at=ts,
            image_path=rel_path,
            width=w,
            height=h,
            application=app_name,
            window_title=win_title,
            monitor=self._monitor_index,
            capture_reason=analysis.reason.value,
            diff_score=round(analysis.diff_score, 6),
        )

        # ── Rename to stable ID-based filename ──────────────────────────
        final_path = self._file_manager.rename_to_id(
            rel_path, frame_id, ts, application=app_name, window_title=win_title
        )
        if final_path != rel_path:
            self._db.update_frame_image_path(frame_id, final_path)

        # ── Build CaptureRecord ─────────────────────────────────────────
        abs_path = self._file_manager.absolute_path(final_path)
        file_size = abs_path.stat().st_size if abs_path.exists() else None

        return CaptureRecord(
            id=frame_id,
            timestamp=ts,
            image_path=final_path,
            application=app_name,
            window_title=win_title,
            monitor=self._monitor_index,
            width=w,
            height=h,
            file_size_bytes=file_size,
            capture_reason=analysis.reason,
            diff_score=round(analysis.diff_score, 6),
        )
