"""
ocr/windows_ocr_engine.py

Windows 10/11 native Media OCR backend via winocr / WinRT.
Fast, offline, zero model download overhead, and native Windows OS accuracy.
"""
from __future__ import annotations

import asyncio
import logging
import platform
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from ocr.base import OCREngine
from ocr.paddle_ocr_engine import BoundingBox, OCRResult, TextBlock

logger = logging.getLogger(__name__)


class WindowsMediaOCREngine(OCREngine):
    """
    OCR Engine using Windows 10/11 native Media OCR.
    Fast (~20-50ms), reliable, and built directly into Windows OS.
    """

    def __init__(self, lang: str = "en-US") -> None:
        self._lang = lang if "-" in lang else f"{lang}-US"

    @classmethod
    def is_available(cls) -> bool:
        """True if running on Windows and winocr is importable."""
        if platform.system() != "Windows":
            return False
        try:
            import winocr  # noqa: F401
            return True
        except ImportError:
            return False

    def extract(self, image: np.ndarray) -> OCRResult:
        """Run Windows Media OCR on the image. Never raises."""
        try:
            import winocr

            # Convert BGR/Grayscale to PIL Image (RGB)
            if len(image.shape) == 2:
                rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:
                rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            else:
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            pil_img = Image.fromarray(rgb)

            # Run async winocr
            loop = asyncio.new_event_loop()
            try:
                result = loop.run_until_complete(winocr.recognize_pil(pil_img, self._lang))
            finally:
                loop.close()

            blocks: list[TextBlock] = []
            for line in result.lines:
                text = line.text.strip()
                if not text:
                    continue

                # Compute line bounding box from word rects or line bounds
                words = getattr(line, "words", [])
                if words:
                    min_x = min(w.bounding_rect.x for w in words)
                    min_y = min(w.bounding_rect.y for w in words)
                    max_x = max(w.bounding_rect.x + w.bounding_rect.width for w in words)
                    max_y = max(w.bounding_rect.y + w.bounding_rect.height for w in words)
                else:
                    min_x, min_y, max_x, max_y = 0, 0, image.shape[1], image.shape[0]

                blocks.append(
                    TextBlock(
                        text=text,
                        confidence=0.95,
                        bbox=BoundingBox(
                            int(min_x),
                            int(min_y),
                            int(max_x),
                            int(max_y),
                        ),
                    )
                )

            # Sort top-to-bottom, left-to-right
            blocks.sort(key=lambda b: (b.bbox.y1, b.bbox.x1))
            return OCRResult(blocks=blocks, engine="windows_media_ocr")

        except Exception as exc:  # noqa: BLE001
            logger.error("Windows Media OCR failed: %s", exc)
            return OCRResult(error=str(exc), engine="windows_media_ocr")
