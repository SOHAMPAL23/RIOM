
from __future__ import annotations

import abc
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ocr.paddle_ocr_engine import OCRResult


class OCREngine(abc.ABC):
    """Abstract OCR engine interface."""

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

    @classmethod
    def is_available(cls) -> bool:
        """Return True if this backend can be imported and used."""
        return False
