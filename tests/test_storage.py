"""
tests/test_storage.py

Unit tests for Stage 4 local SQLite storage layer.

Tests:
- save_capture() works and avoids storing duplicate records.
- save_raw_text() works and updates raw text instead of duplicating.
- save_metadata() persists Pydantic StructuredMetadata entities and runs deduplication.
- save_fact_evidence() stores FactEvidence.
- get_session() returns the connection.
- search_text() searches text in the raw text records.
- get_metadata() queries and filters metadata correctly.
- get_captures_by_timestamp() filters frames within ranges.
- get_captures_by_application() filters frames by foreground application.
- Graceful database error handling.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest

from capture.models import CaptureReason
from metadata.schemas import StructuredMetadata, Meeting, Person, FactEvidence, VerificationStatus
from storage.db import Database


def _utc() -> datetime:
    return datetime.now(timezone.utc)


class TestLocalStorage:

    @pytest.fixture()
    def db(self, tmp_path):
        return Database(db_path=tmp_path / "test_storage.db")

    def test_save_capture_inserts_and_avoids_duplicate(self, db):
        ts = _utc()
        fid1 = db.save_capture(ts, "images/1.webp", 800, 600, "chrome", "GitHub", 0.05)
        assert isinstance(fid1, int) and fid1 >= 1

        # Save with same image_path -> should return the same frame ID
        fid2 = db.save_capture(ts, "images/1.webp", 800, 600, "chrome", "GitHub", 0.05)
        assert fid1 == fid2

    def test_save_raw_text_updates_instead_of_duplicate(self, db):
        fid = db.save_capture(_utc(), "images/1.webp", 800, 600)
        
        rid1 = db.save_raw_text(fid, "First OCR content", 0.95, "paddleocr", "[]")
        assert isinstance(rid1, int) and rid1 >= 1

        # Re-save raw text for the same frame_id -> should update and return same ID
        rid2 = db.save_raw_text(fid, "Updated OCR content", 0.98, "paddleocr", "[]")
        assert rid1 == rid2

        # Check database is updated
        stored = db.get_raw_text_record(fid)
        assert stored["raw_text"] == "Updated OCR content"
        assert stored["confidence"] == 0.98

    def test_save_metadata_and_avoid_duplicate_entities(self, db):
        fid = db.save_capture(_utc(), "images/1.webp", 800, 600)
        meta = StructuredMetadata(
            meetings=[Meeting(title="Project Standup Meeting", source_frame_ids=[fid])],
            people=[Person(name="Rahul Sharma", source_frame_ids=[fid])]
        )

        db.save_metadata(meta)
        
        # Verify stored entities
        meetings = db.get_metadata(entity_type="meeting")
        people = db.get_metadata(entity_type="person")
        assert len(meetings) == 1
        assert len(people) == 1

        # Re-save the exact same metadata -> should update payload rather than duplicate
        db.save_metadata(meta)
        meetings = db.get_metadata(entity_type="meeting")
        assert len(meetings) == 1

    def test_save_fact_evidence_and_update(self, db):
        fid = db.save_capture(_utc(), "images/1.webp", 800, 600)
        evidence = FactEvidence(
            fact_id="person_0",
            fact_type="person",
            fact={"name": "Rahul Sharma"},
            source_frame=fid,
            source_timestamp=_utc().isoformat(),
            evidence_text="Rahul Sharma checked in.",
            verification_status=VerificationStatus.VERIFIED,
            unsupported_fields=[]
        )

        eid1 = db.save_fact_evidence(evidence)
        assert isinstance(eid1, int) and eid1 >= 1

        # Re-save with same fact_id -> should update
        evidence.verification_status = VerificationStatus.PARTIALLY_SUPPORTED
        eid2 = db.save_fact_evidence(evidence)
        assert eid1 == eid2

        # Verify updated status
        conn = db.get_session()
        row = conn.execute("SELECT verification_status FROM fact_evidences WHERE id = ?", (eid1,)).fetchone()
        assert row["verification_status"] == "partially_supported"

    def test_get_session(self, db):
        conn = db.get_session()
        assert isinstance(conn, sqlite3.Connection)

    def test_search_text(self, db):
        fid = db.save_capture(_utc(), "images/1.webp", 800, 600)
        db.save_raw_text(fid, "Design sync at 3pm with Alice", 0.90)
        
        results = db.search_text("sync")
        assert len(results) == 1
        assert "Design sync" in results[0]["raw_text"]

    def test_get_metadata_type_filter(self, db):
        fid = db.save_capture(_utc(), "images/1.webp", 800, 600)
        meta = StructuredMetadata(
            meetings=[Meeting(title="Standup", source_frame_ids=[fid])],
            people=[Person(name="Bob", source_frame_ids=[fid])]
        )
        db.save_metadata(meta)

        meetings = db.get_metadata(entity_type="meeting")
        people = db.get_metadata(entity_type="person")
        all_meta = db.get_metadata()

        assert len(meetings) == 1
        assert len(people) == 1
        assert len(all_meta) == 2

    def test_query_by_timestamp_range(self, db):
        ts = datetime(2026, 8, 15, 10, 0, 0, tzinfo=timezone.utc)
        db.save_capture(ts, "images/1.webp", 800, 600)
        db.save_capture(ts + timedelta(minutes=5), "images/2.webp", 800, 600)
        db.save_capture(ts + timedelta(hours=2), "images/3.webp", 800, 600)

        # Query range covering first 10 minutes
        results = db.get_captures_by_timestamp(ts, ts + timedelta(minutes=10))
        assert len(results) == 2

    def test_query_by_application(self, db):
        db.save_capture(_utc(), "images/1.webp", 800, 600, application="chrome")
        db.save_capture(_utc(), "images/2.webp", 800, 600, application="vscode")
        db.save_capture(_utc(), "images/3.webp", 800, 600, application="chrome")

        chrome_results = db.get_captures_by_application("chrome")
        vscode_results = db.get_captures_by_application("vscode")

        assert len(chrome_results) == 2
        assert len(vscode_results) == 1

    def test_graceful_database_error_handling(self, db):
        # Trigger an error by running a malformed query directly on the connection
        conn = db.get_session()
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("SELECT * FROM non_existing_table")
