"""
ocr/pipeline.py

Stage 2 OCR pipeline orchestrator.

Flow
----
    CaptureRecord (image_path)
        ↓  load image from disk
        ↓  preprocess (ImageProcessor: CLAHE + blur)
        ↓  OCR engine.extract()
        ↓  TextNormalizer.normalize()
        ↓  build RawTextRecord
        ↓  persist to DB (raw_text_records + update frames.raw_text)
        ↓  persist OCR blocks (ocr_blocks table)
        → RawTextRecord

Responsibilities
----------------
- Load the image file (handles missing file gracefully).
- Apply preprocessing via ImageProcessor.
- Call the injected OCREngine.
- Normalise the raw text via TextNormalizer.
- Build a fully-populated RawTextRecord with provenance.
- Write to the database in one transaction.
- Emit the RawTextRecord via an optional output queue for Stage 3.
- Log timing and outcome at INFO level.
- Never raise — all errors produce a RawTextRecord with ocr_error set.

Engine selection
----------------
OCRPipeline.build() is a convenience factory that auto-selects the best
available engine:
    1. PaddleOCR — if paddleocr + paddle are importable
    2. Tesseract  — if pytesseract + tesseract binary are available
    3. NullEngine — returns empty OCRResult (graceful no-op)
"""
from __future__ import annotations

import json
import logging
import queue
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from capture.models import CaptureRecord
from ocr.base import OCREngine
from ocr.models import RawTextRecord
from ocr.paddle_ocr_engine import OCRResult
from ocr.text_normalizer import TextNormalizer
from processing.image_processor import ImageProcessor
from storage.db import Database

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Null engine — used when no OCR backend is installed
# ---------------------------------------------------------------------------

class NullEngine(OCREngine):
    """No-op OCR engine. Returns empty result with a clear error message."""

    @classmethod
    def is_available(cls) -> bool:
        return True   # Always available as last resort

    def extract(self, image: np.ndarray) -> OCRResult:
        return OCRResult(
            error="No OCR backend installed. Install paddleocr or pytesseract.",
            engine="null",
        )


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class OCRPipeline:
    """
    Runs the full Screenshot → OCR → RawTextRecord pipeline.

    Args:
        db:              Database for persisting results.
        data_dir:        Root data directory (images stored relative to this).
        engine:          OCREngine to use.
        normalizer:      TextNormalizer for post-processing.
        image_processor: ImageProcessor for pre-processing.
        output_queue:    Optional queue that receives each RawTextRecord.
    """

    def __init__(
        self,
        db: Database,
        data_dir: Path,
        engine: OCREngine,
        normalizer: Optional[TextNormalizer] = None,
        image_processor: Optional[ImageProcessor] = None,
        output_queue: Optional[queue.Queue] = None,
    ) -> None:
        self._db              = db
        self._data_dir        = data_dir
        self._engine          = engine
        self._normalizer      = normalizer or TextNormalizer()
        self._processor       = image_processor or ImageProcessor()
        self._queue           = output_queue

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        db: Database,
        data_dir: Path,
        output_queue: Optional[queue.Queue] = None,
    ) -> "OCRPipeline":
        """Auto-select the best available OCR engine and return a ready pipeline."""
        from ocr.windows_ocr_engine import WindowsMediaOCREngine
        from ocr.paddle_ocr_engine import PaddleOCREngine
        from ocr.tesseract_engine import TesseractEngine

        if WindowsMediaOCREngine.is_available():
            engine: OCREngine = WindowsMediaOCREngine()
            logger.info("OCRPipeline: using Windows Native Media OCR backend.")
        elif PaddleOCREngine.is_available():
            engine = PaddleOCREngine()
            logger.info("OCRPipeline: using PaddleOCR backend.")
        elif TesseractEngine.is_available():
            engine = TesseractEngine()
            logger.info("OCRPipeline: using Tesseract backend.")
        else:
            engine = NullEngine()
            logger.warning(
                "OCRPipeline: no OCR backend available. "
                "Install winocr, paddleocr, or pytesseract."
            )

        return cls(db=db, data_dir=data_dir, engine=engine, output_queue=output_queue)

    # ------------------------------------------------------------------
    # Main processing entry point
    # ------------------------------------------------------------------

    def process(self, record: CaptureRecord) -> RawTextRecord:
        """
        Run the full OCR pipeline for one CaptureRecord.

        Always returns a RawTextRecord.  On error, ocr_error is set and
        raw_text is empty.
        """
        t0 = time.monotonic()

        frame_id = record.id
        if frame_id is None:
            logger.error("CaptureRecord has no id — cannot persist OCR result.")
            return self._error_record(record, "CaptureRecord.id is None")

        # ── Load image ──────────────────────────────────────────────────
        image = self._load_image(record.image_path)
        if image is None:
            msg = f"Image file not found: {record.image_path}"
            logger.warning(msg)
            result = self._error_record(record, msg)
            self._persist(result)
            return result

        # ── Preprocess ──────────────────────────────────────────────────
        try:
            processed = self._processor.process(image)
        except Exception as exc:
            logger.warning("Image preprocessing failed for frame %d: %s", frame_id, exc)
            processed = image   # Fall back to raw image

        # ── OCR ─────────────────────────────────────────────────────────
        ocr_result: OCRResult = self._engine.extract(processed)

        # ── Normalise ───────────────────────────────────────────────────
        normalised = self._normalizer.normalize(ocr_result.full_text)

        # ── Build RawTextRecord ─────────────────────────────────────────
        blocks_json = json.dumps(
            [
                {
                    "text":       b.text,
                    "confidence": round(b.confidence, 4),
                    "bbox":       {"x1": b.bbox.x1, "y1": b.bbox.y1,
                                   "x2": b.bbox.x2, "y2": b.bbox.y2},
                }
                for b in ocr_result.blocks
            ],
            ensure_ascii=False,
        )

        confidence: Optional[float] = (
            round(ocr_result.mean_confidence, 4)
            if ocr_result.blocks
            else None
        )

        raw_record = RawTextRecord(
            frame_id=frame_id,
            timestamp=record.timestamp,
            image_path=record.image_path,
            application=record.application,
            window_title=record.window_title,
            raw_text=normalised,
            confidence=confidence,
            ocr_engine=ocr_result.engine,
            blocks_json=blocks_json,
            char_count=len(normalised),
            is_empty=not bool(normalised.strip()),
            ocr_error=ocr_result.error,
        )

        # ── Persist ─────────────────────────────────────────────────────
        self._persist(raw_record)

        elapsed = time.monotonic() - t0
        logger.info(
            "OCR done: frame_id=%d  engine=%s  chars=%d  confidence=%s  %.2fs",
            frame_id,
            ocr_result.engine,
            raw_record.char_count,
            f"{confidence:.2f}" if confidence is not None else "n/a",
            elapsed,
        )

        if self._queue is not None:
            try:
                self._queue.put_nowait(raw_record)
            except queue.Full:
                logger.debug("OCR output queue full — dropping record for frame %d.", frame_id)

        return raw_record

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_image(self, relative_path: str) -> Optional[np.ndarray]:
        # Load image from disk, returning None if missing or unreadable.
        abs_path = self._data_dir / relative_path
        if not abs_path.exists():
            return None
        img = cv2.imread(str(abs_path), cv2.IMREAD_COLOR)
        if img is None:
            logger.warning("cv2.imread returned None for: %s", abs_path)
        return img

    def _persist(self, raw_record: RawTextRecord) -> None:
        # Write OCR results to the database.
        frame_id = raw_record.frame_id
        try:
            # 1. Insert into raw_text_records
            rec_id = self._db.insert_raw_text_record(raw_record)
            raw_record.id = rec_id

            # 2. Update frames.raw_text + ocr_processed flag
            self._db.update_frame_ocr(frame_id, raw_record.raw_text)

            # 3. Persist individual OCR blocks
            if raw_record.blocks_json and raw_record.blocks_json != "[]":
                blocks = json.loads(raw_record.blocks_json)
                self._db.insert_ocr_blocks(frame_id, blocks)

        except Exception as exc:
            logger.error("Failed to persist OCR result for frame %d: %s", frame_id, exc)

    def _error_record(self, record: CaptureRecord, error: str) -> RawTextRecord:
        # Build a RawTextRecord that represents a failed OCR attempt.
        return RawTextRecord(
            frame_id=record.id or 0,
            timestamp=record.timestamp,
            image_path=record.image_path,
            application=record.application,
            window_title=record.window_title,
            raw_text="",
            ocr_engine=type(self._engine).__name__,
            is_empty=True,
            ocr_error=error,
        )
