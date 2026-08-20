"""
processing/image_processor.py

Lightweight, configurable image preprocessing module for OCR.
Ensures image quality is enhanced (contrast, noise reduction) without
destroying fine screen fonts or small text.
"""
from __future__ import annotations

import cv2
import numpy as np


class ImagePreprocessor:
    """
    Configurable image preprocessor for screen OCR.

    Args:
        upscale_factor:   Factor to scale image (>1.0 to enlarge small fonts).
        apply_grayscale:  Convert input to grayscale if BGR.
        apply_clahe:      Apply Contrast Limited Adaptive Histogram Equalization.
        apply_blur:       Apply subtle Gaussian blur to reduce high-frequency noise.
        apply_threshold:  Apply binarization thresholding (optional).
        threshold_type:   'otsu' or 'adaptive' (only used when apply_threshold=True).
        clip_limit:       CLAHE contrast limit.
        tile_grid_size:   CLAHE grid tile size.
    """

    def __init__(
        self,
        upscale_factor: float = 1.0,
        apply_grayscale: bool = True,
        apply_clahe: bool = True,
        apply_blur: bool = True,
        apply_threshold: bool = False,
        threshold_type: str = "none",
        clip_limit: float = 2.0,
        tile_grid_size: tuple[int, int] = (8, 8),
    ) -> None:
        self._upscale = upscale_factor
        self._apply_grayscale = apply_grayscale
        self._apply_clahe = apply_clahe
        self._apply_blur = apply_blur
        self._apply_threshold = apply_threshold
        self._threshold_type = threshold_type.lower()
        self._clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess input image for OCR.

        Args:
            frame: Input numpy array (BGR or grayscale).

        Returns:
            Preprocessed numpy array.
        """
        if frame is None or frame.size == 0:
            return frame

        out = frame.copy()

        # 1. Grayscale
        if self._apply_grayscale and len(out.shape) == 3 and out.shape[2] >= 3:
            out = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)

        # 2. Contrast Enhancement (CLAHE)
        if self._apply_clahe:
            if len(out.shape) == 2:
                out = self._clahe.apply(out)
            else:
                # Apply CLAHE to L channel in LAB color space if color preserved
                lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                l2 = self._clahe.apply(l)
                out = cv2.cvtColor(cv2.merge((l2, a, b)), cv2.COLOR_LAB2BGR)

        # 3. Noise Reduction (Subtle Gaussian Blur)
        if self._apply_blur:
            out = cv2.GaussianBlur(out, (3, 3), 0)

        # 4. Optional Thresholding
        if self._apply_threshold and len(out.shape) == 2:
            if self._threshold_type == "otsu":
                _, out = cv2.threshold(out, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            elif self._threshold_type == "adaptive":
                out = cv2.adaptiveThreshold(
                    out, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
                )

        # 5. Optional Resizing / Upscaling
        if self._upscale != 1.0:
            h, w = out.shape[:2]
            out = cv2.resize(
                out,
                (int(w * self._upscale), int(h * self._upscale)),
                interpolation=cv2.INTER_CUBIC,
            )

        return out

    # Alias for process
    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        return self.process(frame)


# Alias for backward compatibility
ImageProcessor = ImagePreprocessor
