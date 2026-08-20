"""
ocr/paddle_ocr_engine.py

PaddleOCR backend implementing the OCRProvider / OCREngine interface.

Characteristics
---------------
- Lazy-loads model weights on first extract() call (~3-5 s cold start).
- All PaddleOCR calls are serialised behind a threading.Lock because
  PaddleOCR's internal state is not thread-safe.
- Returns OCRResult.error on any exception — never propagates.
- Blocks are sorted in reading order (top-to-bottom, left-to-right).
- is_available() does a cheap import check without loading weights.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ocr.base import OCRProvider

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared data structures (used by all engine backends)
# ---------------------------------------------------------------------------

@dataclass
class BoundingBox:
    """Pixel bounding box for a detected text region."""
    x1: int
    y1: int
    x2: int
    y2: int

    def to_dict(self) -> dict[str, int]:
        return {"x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2}


@dataclass
class TextBlock:
    """A single detected text region with its bounding box and confidence."""
    text: str
    confidence: float
    bbox: BoundingBox

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "confidence": round(self.confidence, 4),
            "bbox": self.bbox.to_dict(),
        }


@dataclass
class OCRResult:
    """
    Structured output from any OCR engine for a single image.

    Fields
    ------
    blocks          Detected text blocks in reading order.
    error           Non-empty if the engine failed; blocks will be empty.
    engine          Name of the backend that produced this result.
    processing_time Duration of OCR execution in seconds.
    """
    blocks:          list[TextBlock] = field(default_factory=list)
    error:           Optional[str]   = None
    engine:          str             = "unknown"
    processing_time: float           = 0.0

    @property
    def full_text(self) -> str:
        """All text joined by newlines, in reading order."""
        return "\n".join(b.text for b in self.blocks if b.text.strip())

    @property
    def text(self) -> str:
        """Alias for full_text."""
        return self.full_text

    @property
    def bounding_boxes(self) -> list[TextBlock]:
        """Alias for blocks."""
        return self.blocks

    @property
    def provider(self) -> str:
        """Alias for engine."""
        return self.engine

    @property
    def is_empty(self) -> bool:
        """True if no text was detected (and no error)."""
        return not bool(self.blocks)

    @property
    def mean_confidence(self) -> float:
        """Average confidence across all blocks, or 0.0 if empty."""
        if not self.blocks:
            return 0.0
        return sum(b.confidence for b in self.blocks) / len(self.blocks)

    @property
    def confidence(self) -> float:
        """Alias for mean_confidence."""
        return self.mean_confidence


# ---------------------------------------------------------------------------
# PaddleOCR engine
# ---------------------------------------------------------------------------

class PaddleOCREngine(OCRProvider):
    """
    Thread-safe PaddleOCR wrapper.

    Args:
        lang:           Language code (default 'en').
        min_confidence: Drop blocks whose confidence is below this value.
        use_gpu:        Enable GPU acceleration.
    """

    def __init__(
        self,
        lang: str = "en",
        min_confidence: float = 0.6,
        use_gpu: bool = False,
    ) -> None:
        self._lang           = lang
        self._min_confidence = min_confidence
        self._use_gpu        = use_gpu
        self._ocr            = None   # Loaded lazily
        self._lock           = threading.Lock()
        self._init_lock      = threading.Lock()

    @classmethod
    def is_available(cls) -> bool:
        try:
            import paddleocr  # noqa: F401
            import paddle      # noqa: F401
            return True
        except ImportError:
            return False

    def _ensure_loaded(self) -> None:
        if self._ocr is not None:
            return
        with self._init_lock:
            if self._ocr is not None:
                return
            logger.info("Loading PaddleOCR model (lang=%s)…", self._lang)
            from paddleocr import PaddleOCR  # type: ignore
            try:
                self._ocr = PaddleOCR(
                    use_angle_cls=True,
                    lang=self._lang,
                    use_gpu=self._use_gpu,
                    show_log=False,
                )
            except Exception:
                try:
                    self._ocr = PaddleOCR(lang=self._lang)
                except Exception as exc:
                    logger.error("PaddleOCR constructor failed: %s", exc)
                    raise
            logger.info("PaddleOCR model loaded.")

    def extract(self, image: np.ndarray) -> OCRResult:
        """Run PaddleOCR on the image. Never raises."""
        try:
            self._ensure_loaded()
        except Exception as exc:
            logger.error("PaddleOCR failed to load: %s", exc)
            return OCRResult(error=str(exc), engine="paddleocr")

        try:
            with self._lock:
                try:
                    raw = self._ocr.ocr(image, cls=True)  # type: ignore
                except Exception:
                    raw = self._ocr.ocr(image)  # type: ignore
        except Exception as exc:
            logger.error("PaddleOCR.ocr() raised: %s", exc)
            return OCRResult(error=str(exc), engine="paddleocr")

        blocks: list[TextBlock] = []
        if raw and isinstance(raw, list) and len(raw) > 0 and raw[0]:
            for line in raw[0]:
                if not line:
                    continue
                try:
                    quad, text_conf = line[0], line[1]
                    if isinstance(text_conf, (list, tuple)):
                        text, conf = str(text_conf[0]), float(text_conf[1])
                    else:
                        text, conf = str(text_conf), 1.0

                    if conf < self._min_confidence:
                        continue

                    xs = [int(p[0]) for p in quad]
                    ys = [int(p[1]) for p in quad]
                    blocks.append(TextBlock(
                        text=text,
                        confidence=conf,
                        bbox=BoundingBox(min(xs), min(ys), max(xs), max(ys)),
                    ))
                except Exception:
                    continue

        # Reading order: top-to-bottom, left-to-right
        blocks.sort(key=lambda b: (b.bbox.y1, b.bbox.x1))
        return OCRResult(blocks=blocks, engine="paddleocr")
