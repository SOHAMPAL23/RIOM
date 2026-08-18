"""
ocr/models.py

RawTextRecord: the canonical data contract produced by Stage 2.

A RawTextRecord is created for every frame that passes OCR.  It
preserves full provenance back to the source CaptureRecord and is the
input to Stage 3 (LLM metadata extraction).

Design
------
- Pydantic model for validation and JSON serialisation.
- All provenance fields (frame_id, image_path, application, window_title,
  timestamp) are copied from the upstream CaptureRecord — Stage 3 never
  needs to join back to the frames table for basic context.
- `blocks_json` stores the raw OCR blocks as a JSON string so the DB
  schema stays flat while still preserving bounding-box detail.
- `confidence` is the mean confidence across all blocks [0.0–1.0].
  None when the engine does not provide per-block confidence.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class RawTextRecord(BaseModel):
    """
    OCR output for one captured frame.

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
    """

    id:           Optional[int]   = Field(default=None)
    frame_id:     int             = Field(description="Source frame database ID.")
    timestamp:    datetime        = Field(description="UTC capture time.")
    image_path:   str             = Field(description="Relative path to source WebP.")
    application:  Optional[str]   = Field(default=None)
    window_title: Optional[str]   = Field(default=None)
    raw_text:     str             = Field(default="", description="Normalised OCR text.")
    confidence:   Optional[float] = Field(default=None, description="Mean block confidence [0–1].")
    ocr_engine:   str             = Field(default="unknown")
    blocks_json:  str             = Field(default="[]", description="JSON TextBlock list.")
    char_count:   int             = Field(default=0)
    is_empty:     bool            = Field(default=True)
    ocr_error:    Optional[str]   = Field(default=None)

    model_config = {"frozen": False}
