
from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class MergedTextRecord(BaseModel):
    id:                       Optional[int]       = Field(default=None)
    contributing_frame_ids:   list[int]           = Field(default_factory=list)
    contributing_timestamps:  list[str]           = Field(default_factory=list)  # ISO strings
    contributing_image_paths: list[str]           = Field(default_factory=list)
    first_timestamp:          datetime            = Field(description="Earliest capture time.")
    last_timestamp:           datetime            = Field(description="Latest capture time.")
    application:              Optional[str]       = Field(default=None)
    window_title:             Optional[str]       = Field(default=None)
    merged_text:              str                 = Field(default="")
    char_count:               int                 = Field(default=0)
    is_empty:                 bool                = Field(default=True)
    is_deduplicated:          bool                = Field(default=False)
    frame_count:              int                 = Field(default=0)
    similarity_scores:        dict[str, float]    = Field(default_factory=dict)
    ocr_engines:              list[str]           = Field(default_factory=list)

    model_config = {"frozen": False}
