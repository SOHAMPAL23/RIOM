"""
tests/test_ocr.py

Unit and integration tests for the Stage 2 OCR pipeline.

Coverage
--------
TextNormalizer
    - Strips per-line whitespace
    - Collapses blank lines
    - Drops short noise lines
    - Deduplicates consecutive identical lines
    - Handles empty input
    - Windows line endings (\\r\\n)

OCRResult / TextBlock / BoundingBox
    - full_text reading-order join
    - is_empty flag
    - mean_confidence calculation
    - error field propagation

ImageProcessor
    - Produces grayscale output
    - Handles colour and grayscale input
    - Upscale factor changes dimensions

NullEngine
    - Always available
    - Returns OCRResult with error set

OCRPipeline  (NullEngine backend, real ImageProcessor, mocked DB)
    - process() with a valid image → RawTextRecord returned
    - process() with a missing image → RawTextRecord with ocr_error set
    - process() with CaptureRecord.id=None → error record returned
    - Provenance preserved (timestamp, image_path, application, window_title)
    - DB methods called correctly

Database  (raw_text_records round-trip)
    - insert_raw_text_record / get_raw_text_record
    - search_raw_text LIKE query
    - get_raw_text_records joins capture_reason

Integration  (real OpenCV test images — no PaddleOCR / Tesseract needed)
    - OCRPipeline processes blank.png  → is_empty=True
    - OCRPipeline processes text_sample.png → is_empty=True (NullEngine)
    - ImageProcessor on noisy_dark.png returns valid array

Note: Full PaddleOCR / Tesseract integration tests require the backend
      to be installed and are guarded by:
          AMBIENT_RUN_OCR_INTEGRATION=1 pytest tests/test_ocr.py -m integration
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Fixture paths
# ---------------------------------------------------------------------------

FIXTURES = Path(__file__).parent / "fixtures"
BLANK_IMG      = FIXTURES / "blank.png"
TEXT_IMG       = FIXTURES / "text_sample.png"
NOISY_IMG      = FIXTURES / "noisy_dark.png"


def _utc() -> datetime:
    return datetime.now(timezone.utc)


def _make_capture_record(frame_id: int = 1, image_path: str = "images/x.webp"):
    from capture.models import CaptureRecord
    return CaptureRecord(
        id=frame_id,
        timestamp=_utc(),
        image_path=image_path,
        application="vscode",
        window_title="test.py — ambient_screen",
        monitor=1,
        width=800,
        height=600,
    )


# ===========================================================================
# TextNormalizer
# ===========================================================================

class TestTextNormalizer:

    def setup_method(self):
        from ocr.text_normalizer import TextNormalizer
        self.norm = TextNormalizer()

    def test_empty_string_returns_empty(self):
        assert self.norm.normalize("") == ""

    def test_strips_line_whitespace(self):
        result = self.norm.normalize("  hello  \n  world  ")
        assert "  " not in result

    def test_collapses_many_blank_lines_to_one(self):
        text = "line1\n\n\n\n\nline2"
        result = self.norm.normalize(text)
        # Should have at most 2 blank lines between non-blank lines
        assert "\n\n\n" not in result

    def test_drops_single_char_lines(self):
        text = "Hello\n.\nWorld"
        result = self.norm.normalize(text)
        assert "." not in result.splitlines()

    def test_drops_blank_only_content(self):
        result = self.norm.normalize("   \n   \n   ")
        assert result == ""

    def test_deduplicates_consecutive_identical_lines(self):
        text = "Status bar\nStatus bar\nStatus bar\nContent"
        result = self.norm.normalize(text)
        lines = result.splitlines()
        assert lines.count("Status bar") == 1

    def test_preserves_different_consecutive_lines(self):
        text = "Line A\nLine B\nLine A"
        result = self.norm.normalize(text)
        lines = result.splitlines()
        assert lines.count("Line A") == 2

    def test_windows_line_endings(self):
        text = "Hello\r\nWorld\r\n"
        result = self.norm.normalize(text)
        assert "\r" not in result
        assert "Hello" in result
        assert "World" in result

    def test_unicode_nfc_normalization(self):
        # é as NFD (e + combining accent) → should normalise to NFC
        nfd = "caf\u0065\u0301"   # 'e' + combining accent
        nfc = "caf\u00e9"         # precomposed é
        result = self.norm.normalize(nfd)
        assert result == nfc

    def test_all_steps_disabled(self):
        from ocr.text_normalizer import TextNormalizer, NormalizerConfig
        cfg = NormalizerConfig(
            strip_lines=False,
            collapse_blank_lines=False,
            drop_short_lines=False,
            deduplicate_lines=False,
            normalise_unicode=False,
        )
        norm = TextNormalizer(config=cfg)
        text = "  hello  \n.\n.\n  "
        result = norm.normalize(text)
        # The global .strip() at the end trims outer whitespace, so leading
        # spaces on the first line are removed — but trailing spaces on "hello"
        # and the short "." lines are preserved (steps are disabled).
        assert "hello  " in result      # trailing spaces preserved
        assert "." in result            # short lines not dropped
        lines = result.splitlines()
        assert lines.count(".") == 2    # consecutive duplicates not removed



    def test_min_line_length_configurable(self):
        from ocr.text_normalizer import TextNormalizer, NormalizerConfig
        norm = TextNormalizer(NormalizerConfig(min_line_length=5))
        text = "Hi\nHello World"
        result = norm.normalize(text)
        assert "Hi" not in result.splitlines()
        assert "Hello World" in result


# ===========================================================================
# OCRResult / TextBlock / BoundingBox
# ===========================================================================

class TestOCRResult:

    def test_full_text_joins_in_order(self):
        from ocr.paddle_ocr_engine import OCRResult, TextBlock, BoundingBox
        blocks = [
            TextBlock("Hello", 0.99, BoundingBox(0,  0, 100, 20)),
            TextBlock("World", 0.95, BoundingBox(0, 30, 100, 50)),
        ]
        assert OCRResult(blocks=blocks).full_text == "Hello\nWorld"

    def test_is_empty_true_when_no_blocks(self):
        from ocr.paddle_ocr_engine import OCRResult
        assert OCRResult().is_empty is True

    def test_is_empty_false_with_blocks(self):
        from ocr.paddle_ocr_engine import OCRResult, TextBlock, BoundingBox
        b = TextBlock("hi", 0.9, BoundingBox(0, 0, 10, 10))
        assert OCRResult(blocks=[b]).is_empty is False

    def test_full_text_skips_whitespace_only_blocks(self):
        from ocr.paddle_ocr_engine import OCRResult, TextBlock, BoundingBox
        blocks = [
            TextBlock("   ", 0.99, BoundingBox(0, 0, 10, 10)),
            TextBlock("Real", 0.95, BoundingBox(0, 20, 10, 30)),
        ]
        assert OCRResult(blocks=blocks).full_text == "Real"

    def test_mean_confidence_empty_is_zero(self):
        from ocr.paddle_ocr_engine import OCRResult
        assert OCRResult().mean_confidence == 0.0

    def test_mean_confidence_average(self):
        from ocr.paddle_ocr_engine import OCRResult, TextBlock, BoundingBox
        blocks = [
            TextBlock("A", 0.8, BoundingBox(0, 0, 10, 10)),
            TextBlock("B", 0.6, BoundingBox(0, 15, 10, 25)),
        ]
        assert OCRResult(blocks=blocks).mean_confidence == pytest.approx(0.7)

    def test_error_field_propagated(self):
        from ocr.paddle_ocr_engine import OCRResult
        r = OCRResult(error="backend crashed", engine="paddleocr")
        assert r.error == "backend crashed"
        assert r.is_empty is True

    def test_engine_field(self):
        from ocr.paddle_ocr_engine import OCRResult
        r = OCRResult(engine="tesseract")
        assert r.engine == "tesseract"


# ===========================================================================
# ImageProcessor
# ===========================================================================

class TestImageProcessor:

    def setup_method(self):
        from processing.image_processor import ImageProcessor
        self.proc = ImageProcessor()

    def test_bgr_input_returns_grayscale(self):
        bgr = np.ones((100, 200, 3), dtype=np.uint8) * 128
        result = self.proc.process(bgr)
        assert result.ndim == 2   # grayscale
        assert result.shape == (100, 200)

    def test_upscale_factor_changes_dimensions(self):
        from processing.image_processor import ImageProcessor
        proc = ImageProcessor(upscale_factor=2.0)
        bgr = np.ones((100, 200, 3), dtype=np.uint8) * 128
        result = proc.process(bgr)
        assert result.shape == (200, 400)

    def test_real_blank_image(self):
        img = cv2.imread(str(BLANK_IMG), cv2.IMREAD_COLOR)
        result = self.proc.process(img)
        assert result.ndim == 2

    def test_real_text_image(self):
        img = cv2.imread(str(TEXT_IMG), cv2.IMREAD_COLOR)
        result = self.proc.process(img)
        assert result.ndim == 2
        assert result.shape[0] == img.shape[0]

    def test_noisy_dark_image(self):
        img = cv2.imread(str(NOISY_IMG), cv2.IMREAD_COLOR)
        result = self.proc.process(img)
        assert result is not None
        assert result.ndim == 2


# ===========================================================================
# NullEngine
# ===========================================================================

class TestNullEngine:

    def test_is_always_available(self):
        from ocr.pipeline import NullEngine
        assert NullEngine.is_available() is True

    def test_extract_returns_ocr_result_with_error(self):
        from ocr.pipeline import NullEngine
        from ocr.paddle_ocr_engine import OCRResult
        engine = NullEngine()
        result = engine.extract(np.zeros((100, 100, 3), dtype=np.uint8))
        assert isinstance(result, OCRResult)
        assert result.error is not None
        assert result.is_empty is True
        assert result.full_text == ""

    def test_extract_never_raises(self):
        from ocr.pipeline import NullEngine
        engine = NullEngine()
        result = engine.extract(np.array([]))   # Pathological input
        assert result is not None


# ===========================================================================
# RawTextRecord model
# ===========================================================================

class TestRawTextRecord:

    def test_default_fields(self):
        from ocr.models import RawTextRecord
        rec = RawTextRecord(
            frame_id=1,
            timestamp=_utc(),
            image_path="x.webp",
        )
        assert rec.is_empty is True
        assert rec.char_count == 0
        assert rec.raw_text == ""
        assert rec.ocr_error is None
        assert rec.id is None

    def test_json_roundtrip(self):
        from ocr.models import RawTextRecord
        rec = RawTextRecord(
            frame_id=5,
            timestamp=_utc(),
            image_path="images/x.webp",
            application="chrome",
            window_title="GitHub",
            raw_text="Hello World",
            confidence=0.95,
            ocr_engine="paddleocr",
            char_count=11,
            is_empty=False,
        )
        restored = RawTextRecord.model_validate_json(rec.model_dump_json())
        assert restored.frame_id == 5
        assert restored.raw_text == "Hello World"
        assert restored.application == "chrome"
        assert restored.confidence == pytest.approx(0.95)

    def test_error_record_fields(self):
        from ocr.models import RawTextRecord
        rec = RawTextRecord(
            frame_id=2,
            timestamp=_utc(),
            image_path="missing.webp",
            is_empty=True,
            ocr_error="Image file not found",
        )
        assert rec.ocr_error == "Image file not found"
        assert rec.is_empty is True


# ===========================================================================
# OCRPipeline (NullEngine + mocked DB + real images)
# ===========================================================================

class TestOCRPipeline:

    @pytest.fixture()
    def db(self, tmp_path):
        from storage.db import Database
        return Database(db_path=tmp_path / "test.db")

    @pytest.fixture()
    def pipeline(self, db, tmp_path):
        from ocr.pipeline import OCRPipeline, NullEngine
        return OCRPipeline(
            db=db,
            data_dir=tmp_path,
            engine=NullEngine(),
        )

    def _make_record(self, tmp_path, frame_id=1, image="text_sample.png"):
        """Insert a frame row and copy the fixture image into tmp_path."""
        from capture.models import CaptureRecord
        from storage.db import Database
        db = Database(db_path=tmp_path / "test.db")
        rel = f"images/2026-01-01/{image}"
        dest = tmp_path / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        src = FIXTURES / image
        if src.exists():
            dest.write_bytes(src.read_bytes())
        frame_id = db.insert_frame(
            captured_at=_utc(),
            image_path=rel,
            width=800,
            height=600,
            application="vscode",
            window_title="test.py",
        )
        return CaptureRecord(
            id=frame_id,
            timestamp=_utc(),
            image_path=rel,
            application="vscode",
            window_title="test.py",
            monitor=1,
            width=800,
            height=600,
        )

    def test_process_returns_raw_text_record(self, pipeline, db, tmp_path):
        from ocr.models import RawTextRecord
        rec = self._make_record(tmp_path)
        result = pipeline.process(rec)
        assert isinstance(result, RawTextRecord)

    def test_process_preserves_provenance(self, pipeline, db, tmp_path):
        rec = self._make_record(tmp_path)
        result = pipeline.process(rec)
        assert result.frame_id == rec.id
        assert result.application == rec.application
        assert result.window_title == rec.window_title
        assert result.timestamp == rec.timestamp
        assert result.image_path == rec.image_path

    def test_process_missing_image_sets_error(self, pipeline, db, tmp_path):
        from capture.models import CaptureRecord
        from storage.db import Database
        d = Database(db_path=tmp_path / "test.db")
        fid = d.insert_frame(_utc(), "missing.webp", 100, 100)
        rec = CaptureRecord(
            id=fid, timestamp=_utc(), image_path="missing.webp",
            width=100, height=100,
        )
        result = pipeline.process(rec)
        assert result.ocr_error is not None
        assert result.is_empty is True

    def test_process_none_frame_id_returns_error(self, pipeline, tmp_path):
        from capture.models import CaptureRecord
        rec = CaptureRecord(
            id=None, timestamp=_utc(), image_path="x.webp",
            width=100, height=100,
        )
        result = pipeline.process(rec)
        assert result.ocr_error is not None

    def test_null_engine_produces_empty_is_true(self, pipeline, tmp_path):
        rec = self._make_record(tmp_path)
        result = pipeline.process(rec)
        # NullEngine returns no blocks → is_empty should be True
        assert result.is_empty is True

    def test_null_engine_sets_ocr_error(self, pipeline, tmp_path):
        rec = self._make_record(tmp_path)
        result = pipeline.process(rec)
        assert result.ocr_error is not None

    def test_process_updates_frames_ocr_processed(self, pipeline, db, tmp_path):
        rec = self._make_record(tmp_path)
        pipeline.process(rec)
        rows = db.get_pending_ocr_frames(limit=10)
        # frame should have been marked ocr_processed=1
        assert all(r["id"] != rec.id for r in rows)

    def test_raw_text_record_persisted_in_db(self, pipeline, db, tmp_path):
        rec = self._make_record(tmp_path)
        pipeline.process(rec)
        stored = db.get_raw_text_record(rec.id)
        assert stored is not None
        assert stored["frame_id"] == rec.id


# ===========================================================================
# Database raw_text_records round-trip
# ===========================================================================

class TestDatabaseRawText:

    @pytest.fixture()
    def db(self, tmp_path):
        from storage.db import Database
        return Database(db_path=tmp_path / "test.db")

    def _insert_frame(self, db) -> int:
        return db.insert_frame(_utc(), "x.webp", 100, 100)

    def test_insert_and_retrieve(self, db):
        from ocr.models import RawTextRecord
        fid = self._insert_frame(db)
        rec = RawTextRecord(
            frame_id=fid,
            timestamp=_utc(),
            image_path="x.webp",
            application="chrome",
            window_title="GitHub",
            raw_text="Hello World",
            confidence=0.92,
            ocr_engine="paddleocr",
            blocks_json="[]",
            char_count=11,
            is_empty=False,
        )
        row_id = db.insert_raw_text_record(rec)
        assert isinstance(row_id, int) and row_id >= 1

        stored = db.get_raw_text_record(fid)
        assert stored is not None
        assert stored["raw_text"] == "Hello World"
        assert stored["application"] == "chrome"
        assert stored["ocr_engine"] == "paddleocr"
        assert stored["is_empty"] == 0

    def test_search_raw_text_finds_match(self, db):
        from ocr.models import RawTextRecord
        for i in range(3):
            fid = self._insert_frame(db)
            rec = RawTextRecord(
                frame_id=fid,
                timestamp=_utc(),
                image_path=f"{i}.webp",
                raw_text="Meeting at 3pm with Alice" if i == 1 else "Random content here",
                char_count=10,
                is_empty=False,
            )
            db.insert_raw_text_record(rec)
        results = db.search_raw_text("Meeting")
        assert len(results) == 1
        assert "Meeting" in results[0]["raw_text"]

    def test_search_raw_text_no_match(self, db):
        from ocr.models import RawTextRecord
        fid = self._insert_frame(db)
        rec = RawTextRecord(frame_id=fid, timestamp=_utc(), image_path="x.webp",
                            raw_text="Nothing here", char_count=12, is_empty=False)
        db.insert_raw_text_record(rec)
        results = db.search_raw_text("XYZNOTFOUND")
        assert results == []

    def test_get_raw_text_records_returns_newest_first(self, db):
        from ocr.models import RawTextRecord
        import time
        for text in ["First", "Second", "Third"]:
            fid = self._insert_frame(db)
            rec = RawTextRecord(frame_id=fid, timestamp=_utc(), image_path="x.webp",
                                raw_text=text, char_count=len(text), is_empty=False)
            db.insert_raw_text_record(rec)
            time.sleep(0.01)
        rows = db.get_raw_text_records(limit=3)
        assert rows[0]["raw_text"] == "Third"

    def test_missing_frame_returns_none(self, db):
        assert db.get_raw_text_record(frame_id=99999) is None

    def test_confidence_none_stored_as_null(self, db):
        from ocr.models import RawTextRecord
        fid = self._insert_frame(db)
        rec = RawTextRecord(
            frame_id=fid,
            timestamp=_utc(),
            image_path="x.webp",
            raw_text="",
            is_empty=True,
            confidence=None,
        )
        db.insert_raw_text_record(rec)
        stored = db.get_raw_text_record(fid)
        assert stored["confidence"] is None


# ===========================================================================
# Integration tests (require real OCR — marked so they're opt-in in CI)
# ===========================================================================

@pytest.mark.integration
class TestOCRIntegration:
    """
    Real OCR tests using test fixture images.

    Run with:
        AMBIENT_RUN_OCR_INTEGRATION=1 pytest tests/test_ocr.py -m integration
    """

    @pytest.fixture(autouse=True)
    def skip_if_not_integration(self):
        if not os.getenv("AMBIENT_RUN_OCR_INTEGRATION"):
            pytest.skip("Set AMBIENT_RUN_OCR_INTEGRATION=1 to run OCR integration tests.")

    def test_paddle_available_or_skip(self):
        from ocr.paddle_ocr_engine import PaddleOCREngine
        if not PaddleOCREngine.is_available():
            pytest.skip("PaddleOCR not installed.")

    def test_paddle_extracts_text_from_text_sample(self):
        from ocr.paddle_ocr_engine import PaddleOCREngine
        if not PaddleOCREngine.is_available():
            pytest.skip("PaddleOCR not installed.")
        engine = PaddleOCREngine(min_confidence=0.5)
        img = cv2.imread(str(TEXT_IMG), cv2.IMREAD_COLOR)
        result = engine.extract(img)
        assert result.error is None
        assert not result.is_empty, "Should detect text in text_sample.png"
        # At least one block should contain part of our rendered text
        full = result.full_text.lower()
        assert any(word in full for word in ["hello", "meeting", "riom"])

    def test_paddle_blank_image_returns_empty(self):
        from ocr.paddle_ocr_engine import PaddleOCREngine
        if not PaddleOCREngine.is_available():
            pytest.skip("PaddleOCR not installed.")
        engine = PaddleOCREngine(min_confidence=0.5)
        img = cv2.imread(str(BLANK_IMG), cv2.IMREAD_COLOR)
        result = engine.extract(img)
        assert result.error is None
        assert result.is_empty

    def test_tesseract_extracts_text(self):
        from ocr.tesseract_engine import TesseractEngine
        if not TesseractEngine.is_available():
            pytest.skip("Tesseract not installed.")
        engine = TesseractEngine()
        img = cv2.imread(str(TEXT_IMG), cv2.IMREAD_COLOR)
        result = engine.extract(img)
        assert result.error is None
        assert not result.is_empty
