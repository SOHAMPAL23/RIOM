
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from ocr.base import OCREngine
from ocr.paddle_ocr_engine import BoundingBox, OCRResult, TextBlock

logger = logging.getLogger(__name__)

# Minimum Tesseract confidence to accept a word (0–100 scale)
_MIN_CONF = 30


class TesseractEngine(OCREngine):

    def __init__(
        self,
        lang: str = "eng",
        min_confidence: int = _MIN_CONF,
        tesseract_cmd: Optional[str] = None,
    ) -> None:
        self._lang           = lang
        self._min_confidence = min_confidence
        self._tesseract_cmd  = tesseract_cmd

    @classmethod
    def is_available(cls) -> bool:
        try:
            import pytesseract  # noqa: F401
            pytesseract.get_tesseract_version()
            return True
        except Exception:
            return False

    def extract(self, image: np.ndarray) -> OCRResult:
        """Run Tesseract on the image. Never raises."""
        try:
            import pytesseract
            if self._tesseract_cmd:
                pytesseract.pytesseract.tesseract_cmd = self._tesseract_cmd
        except ImportError as exc:
            return OCRResult(error=f"pytesseract not installed: {exc}", engine="tesseract")

        try:
            data = pytesseract.image_to_data(
                image,
                lang=self._lang,
                output_type=pytesseract.Output.DICT,
            )
        except Exception as exc:
            logger.error("Tesseract extraction failed: %s", exc)
            return OCRResult(error=str(exc), engine="tesseract")

        blocks = self._parse_data(data)
        return OCRResult(blocks=blocks, engine="tesseract")

    def _parse_data(self, data: dict) -> list[TextBlock]:
        """
        Convert pytesseract image_to_data output to TextBlock list.

        Groups words by (block_num, par_num, line_num) to form one
        TextBlock per line, preserving bounding boxes.
        """
        from collections import defaultdict

        # Each entry: (line_key) → list of word dicts
        lines: dict[tuple, list[dict]] = defaultdict(list)

        n = len(data["text"])
        for i in range(n):
            word = str(data["text"][i]).strip()
            conf = int(data["conf"][i])
            if not word or conf < self._min_confidence:
                continue
            key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
            lines[key].append({
                "word": word,
                "conf": conf,
                "x": int(data["left"][i]),
                "y": int(data["top"][i]),
                "w": int(data["width"][i]),
                "h": int(data["height"][i]),
            })

        blocks: list[TextBlock] = []
        for words in lines.values():
            if not words:
                continue
            line_text = " ".join(w["word"] for w in words)
            avg_conf  = sum(w["conf"] for w in words) / len(words) / 100.0
            x1 = min(w["x"] for w in words)
            y1 = min(w["y"] for w in words)
            x2 = max(w["x"] + w["w"] for w in words)
            y2 = max(w["y"] + w["h"] for w in words)
            blocks.append(TextBlock(
                text=line_text,
                confidence=avg_conf,
                bbox=BoundingBox(x1, y1, x2, y2),
            ))

        blocks.sort(key=lambda b: (b.bbox.y1, b.bbox.x1))
        return blocks
