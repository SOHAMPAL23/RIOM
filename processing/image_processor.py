
from __future__ import annotations

import cv2
import numpy as np


class ImageProcessor:
    def __init__(
        self,
        upscale_factor: float = 1.0,
        clip_limit: float = 2.0,
        tile_grid_size: tuple[int, int] = (8, 8),
    ) -> None:
        self._upscale = upscale_factor
        self._clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def process(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Args:
            frame_bgr: Raw captured frame in BGR colour space.

        Returns:
            Grayscale numpy array ready for PaddleOCR input.
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        enhanced = self._clahe.apply(gray)
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)

        if self._upscale != 1.0:
            h, w = blurred.shape
            blurred = cv2.resize(
                blurred,
                (int(w * self._upscale), int(h * self._upscale)),
                interpolation=cv2.INTER_CUBIC,
            )

        return blurred
