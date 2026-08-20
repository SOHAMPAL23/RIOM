"""
Capture module — handles continuous screen capture, video recording, and change detection.
"""
from capture.change_detector import ChangeDetector, CaptureReason, FrameAnalysis
from capture.models import CaptureRecord
from capture.screen_recorder import ScreenRecorder
from capture.video_recorder import ScreenVideoRecorder
from capture.window_info import WindowInfoProvider

__all__ = [
    "ChangeDetector",
    "CaptureReason",
    "FrameAnalysis",
    "CaptureRecord",
    "ScreenRecorder",
    "ScreenVideoRecorder",
    "WindowInfoProvider",
]

