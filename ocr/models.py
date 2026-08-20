"""
ocr/models.py

RawTextRecord: the canonical data contract produced by Stage 2.

A RawTextRecord is created for every frame that passes OCR. It
preserves full provenance back to the source CaptureRecord and is the
input to Stage 3 (LLM metadata extraction).
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ProcessingState(str, Enum):
    """Lifecycle processing states for captured frames and records."""
    CAPTURED = "CAPTURED"
    OCR_PROCESSING = "OCR_PROCESSING"
    OCR_COMPLETE = "OCR_COMPLETE"
    CLEANED = "CLEANED"
    METADATA_PROCESSING = "METADATA_PROCESSING"
    METADATA_COMPLETE = "METADATA_COMPLETE"
    FAILED = "FAILED"
    RETRY = "RETRY"


class RawTextRecord(BaseModel):
    """
    OCR output for one captured frame with complete provenance.

    Fields
    ------
    id              Database primary key (set after DB insert; None before).
    frame_id        Foreign key to the frames table.
    timestamp       Capture timestamp (copied from CaptureRecord).
    image_path      Relative path to the source WebP (for provenance).
    application     Foreground application at capture time.
    window_title    Foreground window title at capture time.
    raw_text        Full extracted text, normalised, newline-separated.
    confidence      Mean OCR confidence [0.0–1.0]; None if unavailable.
    ocr_engine      Name of the OCR backend that produced this result.
    blocks_json     JSON string of raw TextBlock list (bboxes + per-block text).
    char_count      Number of characters in raw_text.
    is_empty        True when no text was detected on the screen.
    ocr_error       Non-empty if the OCR engine reported an error.
    processing_status State in the processing pipeline lifecycle.
    """

    id:                Optional[int]      = Field(default=None)
    frame_id:          int                = Field(description="Source frame database ID.")
    timestamp:         datetime           = Field(description="UTC capture time.")
    image_path:        str                = Field(description="Relative path to source WebP.")
    application:       Optional[str]      = Field(default=None)
    window_title:      Optional[str]      = Field(default=None)
    raw_text:          str                = Field(default="", description="Normalised OCR text.")
    confidence:        Optional[float]    = Field(default=None, description="Mean block confidence [0–1].")
    ocr_engine:        str                = Field(default="unknown")
    blocks_json:       str                = Field(default="[]", description="JSON TextBlock list.")
    char_count:        int                = Field(default=0)
    is_empty:          bool               = Field(default=True)
    ocr_error:         Optional[str]      = Field(default=None)
    processing_status: ProcessingState    = Field(default=ProcessingState.OCR_COMPLETE)

    model_config = {"frozen": False}

    # ── Convenience properties matching interface specs ─────────────────
    @property
    def record_id(self) -> Optional[int]:
        return self.id

    @property
    def text(self) -> str:
        return self.raw_text

    @property
    def timestamp_start(self) -> datetime:
        return self.timestamp

    @property
    def timestamp_end(self) -> datetime:
        return self.timestamp

    @property
    def source_frame_ids(self) -> list[int]:
        return [self.frame_id]

    @property
    def source_image_paths(self) -> list[str]:
        return [self.image_path] if self.image_path else []
