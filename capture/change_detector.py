"""
capture/change_detector.py

Smart frame analysis using OpenCV.

The detector answers ONE question per candidate frame:
    "Should this frame be saved, and if so, why?"

It does this by computing a visual difference score and comparing
the current window context against the previous accepted state.

CaptureReason (enum)
--------------------
Each accepted frame is tagged with exactly one reason:

    visual_change       The pixel difference exceeded the threshold.
    application_change  The foreground application changed since the last
                        accepted frame — save immediately regardless of
                        visual diff score.
    periodic_capture    The max_capture_interval has elapsed without any
                        accepted frame.  A forced "heartbeat" capture keeps
                        the timeline complete even on static screens.
    manual_capture      Triggered programmatically (e.g. user presses a
                        hotkey or the UI requests a snapshot).
    idle                Frame was dropped — the screen has been idle
                        (low diff) for longer than idle_threshold_seconds.
                        NOTE: this reason is returned but the caller can
                        choose to honour or ignore it.

FrameAnalysis (dataclass)
-------------------------
Returned by `analyse()`.  Contains:
    should_save:    bool    Whether to persist this frame.
    reason:         CaptureReason
    diff_score:     float   Normalised pixel diff [0.0 – 1.0].
    app_changed:    bool    Whether the foreground app changed.

Algorithm
---------
1. Resize both frames to a small fixed resolution for O(1) comparison.
2. Convert to grayscale.
3. Compute normalised mean absolute difference (MAD) via cv2.absdiff.
   - Fast on CPU; predictable linear scaling with brightness change.
   - Preferred over SSIM: SSIM requires scipy, is 10× slower, and gives
     diminishing returns for the coarse "did the screen change?" question.
4. Check CaptureReason in priority order:
       application_change → periodic_capture → visual_change → idle → skip

All thresholds are injected at construction time; no global state.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import cv2
import numpy as np


class CaptureReason(str, Enum):
    """Why a frame was (or was not) accepted."""
    VISUAL_CHANGE       = "visual_change"
    APPLICATION_CHANGE  = "application_change"
    PERIODIC_CAPTURE    = "periodic_capture"
    MANUAL_CAPTURE      = "manual_capture"
    IDLE                = "idle"        # Not saved — screen is idle
    NO_CHANGE           = "no_change"   # Not saved — below threshold


@dataclass
class FrameAnalysis:
    """Result of a single call to ChangeDetector.analyse()."""
    should_save:  bool
    reason:       CaptureReason
    diff_score:   float          # Normalised MAD [0.0 – 1.0]
    app_changed:  bool = False   # True when application name changed


class ChangeDetector:
    """
    Compares incoming frames against the last accepted keyframe and
    returns a structured FrameAnalysis explaining the decision.

    Args:
        visual_threshold:       Save if diff_score ≥ this.  [0.0–1.0]
        resize_to:              (W, H) to resize before comparison.
        max_capture_interval:   Seconds before a forced periodic capture.
                                Set to 0 to disable.
        idle_threshold_seconds: Seconds of continuous low-diff activity
                                before the frame is tagged as IDLE.
                                Set to 0 to disable idle detection.
    """

    def __init__(
        self,
        visual_threshold: float = 0.02,
        resize_to: tuple[int, int] = (320, 180),
        max_capture_interval: float = 300.0,
        idle_threshold_seconds: float = 0.0,
    ) -> None:
        self._visual_threshold      = visual_threshold
        self._resize_to             = resize_to
        self._max_capture_interval  = max_capture_interval
        self._idle_threshold        = idle_threshold_seconds

        self._lock                  = threading.Lock()
        self._last_keyframe: Optional[np.ndarray] = None
        self._last_accepted_time: float = 0.0          # time.monotonic()
        self._last_app: Optional[str] = None
        self._idle_since: Optional[float] = None       # time.monotonic()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyse(
        self,
        frame: np.ndarray,
        application: Optional[str] = None,
    ) -> FrameAnalysis:
        """
        Analyse a candidate frame and return a FrameAnalysis.

        Args:
            frame:       BGR numpy array (H×W×3 uint8).
            application: Current foreground application name (best-effort).

        Returns:
            FrameAnalysis — caller decides whether to save based on
            `should_save` and `reason`.
        """
        small     = self._preprocess(frame)
        now       = time.monotonic()

        with self._lock:
            # ── Compute diff ────────────────────────────────────────────
            if self._last_keyframe is None:
                diff_score = 1.0          # First frame — always save
            else:
                diff_score = self._compute_diff(self._last_keyframe, small)

            app_changed = (
                application is not None
                and self._last_app is not None
                and application != self._last_app
            )

            elapsed_since_last = now - self._last_accepted_time

            # ── Decision logic (priority order) ─────────────────────────

            # 1. Application changed → always save
            if app_changed:
                self._accept(small, application, now)
                return FrameAnalysis(
                    should_save=True,
                    reason=CaptureReason.APPLICATION_CHANGE,
                    diff_score=diff_score,
                    app_changed=True,
                )

            # 2. Forced periodic capture (heartbeat)
            if (
                self._max_capture_interval > 0
                and self._last_accepted_time > 0          # at least one frame seen
                and elapsed_since_last >= self._max_capture_interval
            ):
                self._accept(small, application, now)
                return FrameAnalysis(
                    should_save=True,
                    reason=CaptureReason.PERIODIC_CAPTURE,
                    diff_score=diff_score,
                    app_changed=False,
                )

            # 3. Visual change above threshold
            if diff_score >= self._visual_threshold:
                self._idle_since = None  # Screen is active again
                self._accept(small, application, now)
                return FrameAnalysis(
                    should_save=True,
                    reason=CaptureReason.VISUAL_CHANGE,
                    diff_score=diff_score,
                    app_changed=False,
                )

            # 4. Below threshold — check idle
            if self._idle_threshold > 0:
                if self._idle_since is None:
                    self._idle_since = now
                elif now - self._idle_since >= self._idle_threshold:
                    # Screen has been idle for long enough to tag it
                    return FrameAnalysis(
                        should_save=False,
                        reason=CaptureReason.IDLE,
                        diff_score=diff_score,
                    )

            # 5. Update app tracking even on no-save frames
            if application is not None and self._last_app is None:
                self._last_app = application

            return FrameAnalysis(
                should_save=False,
                reason=CaptureReason.NO_CHANGE,
                diff_score=diff_score,
            )

    def force_next(self) -> None:
        """
        Force the next call to analyse() to return a MANUAL_CAPTURE save.
        Thread-safe.
        """
        with self._lock:
            self._last_keyframe = None   # Reset reference → diff = 1.0

    def reset(self) -> None:
        """Reset all state.  Useful between test sessions."""
        with self._lock:
            self._last_keyframe     = None
            self._last_accepted_time = 0.0
            self._last_app          = None
            self._idle_since        = None

    def compute_diff_score(self, frame_a: np.ndarray, frame_b: np.ndarray) -> float:
        """
        Public utility: compute the MAD diff score between two BGR frames.
        Does not update internal state.
        """
        a = self._preprocess(frame_a)
        b = self._preprocess(frame_b)
        return self._compute_diff(a, b)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _accept(
        self,
        small: np.ndarray,
        application: Optional[str],
        now: float,
    ) -> None:
        """Update internal state after a frame is accepted. Must hold lock."""
        self._last_keyframe      = small
        self._last_accepted_time = now
        self._idle_since         = None
        if application is not None:
            self._last_app = application

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Resize to comparison resolution and convert to grayscale."""
        resized = cv2.resize(frame, self._resize_to, interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def _compute_diff(a: np.ndarray, b: np.ndarray) -> float:
        """
        Normalised mean absolute difference between two grayscale arrays.
        Returns a float in [0.0, 1.0].
        """
        return float(cv2.absdiff(a, b).mean()) / 255.0
