"""
tests/test_pipeline_e2e.py

End-to-end integration tests for the RIOM ambient screen understanding pipeline.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from capture.change_detector import CaptureReason
from capture.models import CaptureRecord
from capture.simulation import run_simulation, generate_mock_screenshot
from metadata.extractor import MetadataExtractor
from metadata.llm_client import LLMClient
from metadata.schemas import (
    StructuredMetadata,
    Meeting,
    FileActivity,
    Appointment,
    Person,
    VerificationStatus,
)
from metadata.verifier import MetadataVerifier
from ocr.models import RawTextRecord
from ocr.pipeline import OCRPipeline, NullEngine
from processing.models import MergedTextRecord
from processing.pipeline_coordinator import PipelineCoordinator
from processing.text_processor import TextProcessor
from storage.db import Database
from storage.file_manager import FileManager


@pytest.fixture
def temp_env(tmp_path: Path):
    """Provides a fresh temporary Database and FileManager."""
    data_dir = tmp_path / "ambient_test"
    data_dir.mkdir(parents=True, exist_ok=True)
    db_path = data_dir / "test.db"
    db = Database(db_path=db_path)
    fm = FileManager(data_dir=data_dir, webp_quality=85)
    return db, fm, data_dir


class TestEndToEndPipeline:
    """Tests the entire integrated multi-stage pipeline."""

    def test_simulation_runs_and_populates_database(self, temp_env):
        db, fm, data_dir = temp_env
        count = run_simulation(db=db, file_manager=fm, data_dir=data_dir)
        assert count == 4

        # 1. Frames persisted in SQLite
        recent_frames = db.get_recent_frames(limit=10)
        assert len(recent_frames) == 4

        # 2. Raw text records created
        raw_texts = db.get_raw_text_records(limit=10)
        assert len(raw_texts) == 4

        # 3. Merged text records created
        merged = db.get_merged_text_records(limit=10)
        assert len(merged) >= 1

        # 4. Entities extracted and verified
        meetings = db.get_entities_by_type("meeting")
        assert len(meetings) >= 1
        assert "Sprint Planning" in meetings[0]["payload"]

        files = db.get_entities_by_type("file_activity")
        assert len(files) >= 1
        assert "project.py" in files[0]["payload"] or "Project Brief" in files[0]["payload"]

        appts = db.get_entities_by_type("appointment")
        assert len(appts) >= 1

        people = db.get_entities_by_type("person")
        assert len(people) >= 1

        # 5. Verification fact evidences stored
        conn = db.get_session()
        facts = conn.execute("SELECT * FROM fact_evidences").fetchall()
        assert len(facts) >= 1

    def test_pipeline_provenance_preservation(self, temp_env):
        db, fm, data_dir = temp_env
        run_simulation(db=db, file_manager=fm, data_dir=data_dir)

        # Check that meetings preserve source_frame_ids and timestamps
        meetings = db.get_entities_by_type("meeting")
        for m in meetings:
            payload = json.loads(m["payload"])
            assert "source_frame_ids" in payload
            assert len(payload["source_frame_ids"]) > 0
            assert "source_timestamps" in payload
            assert len(payload["source_timestamps"]) > 0

    def test_ocr_failure_does_not_crash_pipeline(self, temp_env):
        db, fm, data_dir = temp_env

        # Capture a record pointing to a nonexistent file
        rec = CaptureRecord(
            id=101,
            timestamp=datetime.now(timezone.utc),
            image_path="nonexistent_frame.webp",
            application="code.exe",
            window_title="Test Window",
            width=1280,
            height=720,
        )
        db.insert_frame(
            captured_at=rec.timestamp,
            image_path=rec.image_path,
            width=1280,
            height=720,
            application=rec.application,
            window_title=rec.window_title,
        )

        pipeline = OCRPipeline(db=db, data_dir=data_dir, engine=NullEngine())
        # Should gracefully return error record without raising
        result = pipeline.process(rec)
        assert result.ocr_error is not None
        assert result.is_empty is True

    def test_llm_failure_is_non_fatal_and_retryable(self, temp_env):
        db, fm, data_dir = temp_env

        # Create a frame in DB that has OCR but no LLM processing
        ts = datetime.now(timezone.utc)
        fid = db.insert_frame(
            captured_at=ts,
            image_path="test_frame.webp",
            width=1280,
            height=720,
            application="chrome.exe",
            window_title="Google Meet",
        )
        db.update_frame_ocr(fid, "Google Meet | Sprint Planning with Alice")

        # Mock LLM that raises an API error
        mock_client = MagicMock(spec=LLMClient)
        mock_client.complete.side_effect = RuntimeError("503 Service Unavailable")

        extractor = MetadataExtractor(llm_client=mock_client)
        result = extractor.extract("Google Meet | Sprint Planning with Alice", frame_id=fid)

        # Extraction returns None gracefully on failure
        assert result is None

        # Frame remains pending for retry in DB
        pending = db.get_pending_llm_frames(limit=5)
        assert any(r["id"] == fid for r in pending)

    def test_fact_verification_filters_hallucinations(self):
        verifier = MetadataVerifier()

        raw_text = "VS Code editing project.py in RIOM."
        metadata = StructuredMetadata(
            files=[
                FileActivity(file_name="project.py", source_frame_ids=[1]),
                FileActivity(file_name="hallucinated_secret.py", source_frame_ids=[1]),
            ],
            people=[
                Person(name="Bob Hallucinated", source_frame_ids=[1]),
            ]
        )

        verified, evidences = verifier.verify(
            metadata=metadata,
            raw_text_map={1: raw_text},
            timestamps_map={1: "2026-08-15T12:00:00+00:00"},
        )

        # Verified files should only contain project.py
        assert len(verified.files) == 1
        assert verified.files[0].file_name == "project.py"

        # Hallucinated person should be filtered out
        assert len(verified.people) == 0

        # Evidences should record unsupported status
        unsupported_evidences = [
            ev for ev in evidences if ev.verification_status == VerificationStatus.UNSUPPORTED
        ]
        assert len(unsupported_evidences) == 2
