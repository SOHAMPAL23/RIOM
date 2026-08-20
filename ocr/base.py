from __future__ import annotations

import abc
import time
from pathlib import Path
from typing import TYPE_CHECKING, Union, Optional
from dataclasses import dataclass, field

import cv2
import numpy as np

if TYPE_CHECKING:
    from ocr.paddle_ocr_engine import OCRResult


class OCRProvider(abc.ABC):
    """
    Abstract OCR provider/engine interface.
    Supports extract(image: np.ndarray) and extract_text(image_path: Union[str, Path, np.ndarray]).
    """

    @abc.abstractmethod
    def extract(self, image: np.ndarray) -> "OCRResult":
        """
        Extract text from an image array.

        Args:
            image: uint8 numpy array, BGR (H×W×3) or grayscale (H×W).

        Returns:
            OCRResult — always returns, never raises.
            On failure sets OCRResult.error to the exception message.
        """

    def extract_text(self, image_input: Union[str, Path, np.ndarray]) -> "OCRResult":
        """
        Convenience method to extract text from a file path or numpy array.
        Measures processing time and populates OCRResult.processing_time.
        """
        from ocr.paddle_ocr_engine import OCRResult

        t0 = time.monotonic()
        if isinstance(image_input, (str, Path)):
            path = Path(image_input)
            if not path.exists():
                return OCRResult(
                    error=f"Image file not found: {path}",
                    engine=getattr(self, "_name", type(self).__name__),
                    processing_time=0.0,
                )
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                return OCRResult(
                    error=f"cv2.imread failed to load image: {path}",
                    engine=getattr(self, "_name", type(self).__name__),
                    processing_time=0.0,
                )
        elif isinstance(image_input, np.ndarray):
            img = image_input
        else:
            return OCRResult(
                error=f"Unsupported image input type: {type(image_input)}",
                engine=getattr(self, "_name", type(self).__name__),
                processing_time=0.0,
            )

        res = self.extract(img)
        res.processing_time = round(time.monotonic() - t0, 4)
        return res

    @classmethod
    def is_available(cls) -> bool:
        """Return True if this backend can be imported and used."""
        return False


# Alias for backward compatibility
OCREngine = OCRProvider


@dataclass
class TranscriptionSegment:
    """A segment of transcribed audio/video text with timestamps."""
    start_time: float
    end_time: float
    text: str
    confidence: float = 1.0
    speaker: Optional[str] = None


@dataclass
class TranscriptionResult:
    """Output from an audio or video transcription provider."""
    text: str = ""
    segments: list[TranscriptionSegment] = field(default_factory=list)
    language: str = "en"
    duration_seconds: float = 0.0
    processing_time: float = 0.0
    provider: str = "unknown"
    error: Optional[str] = None

    @property
    def is_empty(self) -> bool:
        return not bool(self.text.strip())


class TranscriptionProvider(abc.ABC):
    """
    Abstract interface for future audio/video transcription backends (e.g. Whisper,
    hosted STT, local STT).
    """

    @abc.abstractmethod
    def transcribe(self, media_path: Union[str, Path]) -> TranscriptionResult:
        """
        Transcribe audio/video file at media_path into timestamped text.

        Args:
            media_path: Path to .wav, .mp3, .mp4, or .mkv media file.

        Returns:
            TranscriptionResult containing full text and timestamped segments.
        """

    @classmethod
    def is_available(cls) -> bool:
        """Return True if this transcription backend is installed and usable."""
        return False
