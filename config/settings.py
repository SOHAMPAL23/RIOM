
from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AMBIENT_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    data_dir: Path = Field(
        default=Path.home() / ".ambient_screen",
        description="Root directory for images and the SQLite database.",
    )

    @property
    def db_path(self) -> Path:
        return self.data_dir / "ambient.db"

    @property
    def log_file(self) -> Path:
        return self.data_dir / "ambient.log"

    @property
    def meeting_notes_dir(self) -> Path:
        p = self.data_dir / "meeting_notes"
        p.mkdir(parents=True, exist_ok=True)
        return p

    auto_generate_meeting_notes: bool = Field(
        default=True,
        description="Automatically generate and save Markdown meeting notes when meetings are detected.",
    )

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    log_level: str = Field(
        default="INFO",
        description="Logging level: DEBUG, INFO, WARNING, ERROR.",
    )

    # ------------------------------------------------------------------
    # Capture
    # ------------------------------------------------------------------
    monitor_index: int = Field(
        default=1,
        description="MSS monitor index to capture. 1 = primary, 0 = all monitors combined.",
    )
    capture_interval_seconds: float = Field(
        default=5.0,
        description="How often (in seconds) to attempt a screen capture.",
    )
    capture_save_on_change_only: bool = Field(
        default=True,
        description=(
            "If True, frames are only saved when the change detector fires. "
            "If False, every frame is saved regardless."
        ),
    )
    change_threshold: float = Field(
        default=0.02,
        description="Minimum mean pixel difference (0–1) to save a new frame.",
    )
    capture_resize_width: int = Field(
        default=320,
        description="Width to resize frames to for change detection.",
    )
    capture_resize_height: int = Field(
        default=180,
        description="Height to resize frames to for change detection.",
    )
    max_capture_interval_seconds: float = Field(
        default=300.0,
        description=(
            "Maximum seconds between accepted frames before a forced 'periodic_capture' "
            "heartbeat is saved, even if the screen has not changed. "
            "Set to 0 to disable periodic captures."
        ),
    )
    idle_threshold_seconds: float = Field(
        default=0.0,
        description=(
            "Seconds of continuous low-diff activity before the frame is tagged IDLE "
            "and the log reports the screen as idle. "
            "Set to 0 to disable idle detection."
        ),
    )



    # ------------------------------------------------------------------
    # OCR
    # ------------------------------------------------------------------
    ocr_lang: str = Field(default="en", description="PaddleOCR language code.")
    ocr_min_confidence: float = Field(
        default=0.6,
        description="Minimum confidence score to accept an OCR text block.",
    )
    ocr_use_gpu: bool = Field(default=False, description="Enable GPU for PaddleOCR.")

    # ------------------------------------------------------------------
    # TextProcessor (Stage 2.5)
    # ------------------------------------------------------------------
    text_similarity_threshold: float = Field(
        default=0.85,
        description=(
            "Jaccard similarity threshold [0–1] above which two frames are "
            "considered near-duplicates and merged. 0.85 works well for screen text."
        ),
    )
    text_ui_chrome_window: int = Field(
        default=10,
        description="Number of recent frames the UITextFilter inspects for repeated UI chrome.",
    )
    text_ui_chrome_min_repeats: int = Field(
        default=5,
        description="A line appearing in this many frames within the window is treated as UI chrome.",
    )
    text_min_chars_to_compare: int = Field(
        default=10,
        description="Minimum characters in a text block before similarity dedup is applied.",
    )


    # ------------------------------------------------------------------
    # LLM
    # ------------------------------------------------------------------
    llm_api_key: Optional[str] = Field(default=None, description="LLM provider API key.")
    llm_model: str = Field(default="gpt-4o-mini", description="LLM model identifier.")
    llm_base_url: Optional[str] = Field(
        default=None,
        description="Override base URL for non-OpenAI providers (e.g. Groq).",
    )
    llm_max_retries: int = Field(default=3, description="Retry attempts for LLM API calls.")
    llm_timeout_seconds: float = Field(default=30.0, description="LLM request timeout.")

    # ------------------------------------------------------------------
    # Privacy
    # ------------------------------------------------------------------
    privacy_redact_card_numbers: bool = Field(default=True)
    privacy_redact_ssn: bool = Field(default=True)
    privacy_redact_api_keys: bool = Field(default=True)
    privacy_redact_emails: bool = Field(default=False)

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------
    webp_quality: int = Field(
        default=85,
        description="WebP compression quality (0–100). 85 is near-lossless for screen content.",
    )
    retention_days: int = Field(
        default=30,
        description="Delete images older than this many days. 0 = keep forever.",
    )
    max_queue_size: int = Field(
        default=50,
        description="Maximum number of frames queued between stages.",
    )

    # ------------------------------------------------------------------
    # Video Recording (Continuous Recording Option)
    # ------------------------------------------------------------------
    enable_video_recording: bool = Field(
        default=False,
        description="Optionally record continuous screen video in addition to / alongside smart stills.",
    )
    video_fps: float = Field(
        default=2.0,
        description="Frames per second for continuous screen video recording.",
    )
    video_segment_minutes: int = Field(
        default=15,
        description="Duration in minutes per video segment file (prevents file corruption).",
    )
    video_codec: str = Field(
        default="mp4v",
        description="FourCC video codec identifier (e.g. mp4v, XVID, avc1).",
    )

    # ------------------------------------------------------------------
    # Application & Tray
    # ------------------------------------------------------------------
    minimize_to_tray: bool = Field(
        default=True,
        description="Minimize to system tray on window minimize or close.",
    )


# Singleton — import and use `settings` everywhere
settings = Settings()

