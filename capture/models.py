"""
capture/models.py

Defines CaptureRecord: the structured output produced for every
accepted keyframe.  This is the canonical data contract between the
capture stage and everything downstream (storage, OCR, UI).

CaptureRecord is a Pydantic model so it can be:
- Validated at construction time.
- Serialised to JSON for logging / debugging.
- Extended with new optional fields without breaking existing code.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from capture.change_detector import CaptureReason


class CaptureRecord(BaseModel):
    """
    Represents one saved keyframe produced by the capture pipeline.

    Fields
    ------
    id              Database primary key (set after DB insert; None before).
    timestamp       UTC datetime when the frame was captured.
    image_path      Relative path to the saved WebP file (relative to data_dir).
    application     Name of the foreground process at capture time (best-effort).
    window_title    Title of the foreground window at capture time (best-effort).
    monitor         Monitor index that was captured (1 = primary).
    width           Frame width in pixels.
    height          Frame height in pixels.
    file_size_bytes Size of the saved WebP file on disk (set after save).
    capture_reason  Why this frame was captured (visual_change, application_change,
                    periodic_capture, manual_capture).
    diff_score      Normalised pixel-diff score against the previous keyframe [0–1].
    """

    id: Optional[int] = Field(default=None, description="Database row ID; None before insert.")
    timestamp: datetime = Field(description="UTC capture time.")
    image_path: str = Field(description="Relative path to the WebP image file.")
    application: Optional[str] = Field(default=None, description="Foreground process name.")
    window_title: Optional[str] = Field(default=None, description="Foreground window title.")
    monitor: int = Field(default=1, description="MSS monitor index (1 = primary).")
    width: int = Field(description="Frame width in pixels.")
    height: int = Field(description="Frame height in pixels.")
    file_size_bytes: Optional[int] = Field(default=None, description="Compressed file size.")
    capture_reason: CaptureReason = Field(
        default=CaptureReason.VISUAL_CHANGE,
        description="Why this frame was saved.",
    )
    diff_score: float = Field(
        default=0.0,
        description="Normalised pixel-diff score vs previous keyframe [0.0–1.0].",
    )

    model_config = {"frozen": False}  # Mutable so id/file_size_bytes can be set post-init
