"""
tests/test_metadata.py

Unit tests for Stage 3 LLM metadata extraction.

Coverage:
- Pydantic schema validation for StructuredMetadata (Meeting, FileActivity,
  Appointment, Person, Organization, Project, URLReference).
- PrivacyFilter redaction of cards, SSNs, API keys.
- MetadataExtractor edge cases:
  * Empty OCR text returns None (skips LLM call).
  * Invalid JSON returns None (logs and fails gracefully).
  * Validation retries on malformed LLM response.
  * Correct provenance fallback/injection when LLM does not supply them.
  * Observed vs inferred tracking.
  * Configuration through environment variables.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from metadata.extractor import MetadataExtractor
from metadata.schemas import StructuredMetadata, ScreenContext, Meeting, FileActivity, Appointment
from ocr.models import RawTextRecord
from processing.privacy_filter import PrivacyFilter


def _utc() -> datetime:
    return datetime.now(timezone.utc)


class TestPrivacyFilter:
    def setup_method(self):
        self.f = PrivacyFilter()

    def test_redacts_card_number(self):
        text = "Card: 4111111111111111"
        result = self.f.filter(text)
        assert "4111111111111111" not in result
        assert "[REDACTED:CARD_NUMBER]" in result

    def test_redacts_ssn(self):
        text = "SSN: 123-45-6789"
        result = self.f.filter(text)
        assert "123-45-6789" not in result
        assert "[REDACTED:SSN]" in result

    def test_redacts_api_key(self):
        text = "Bearer sk-123456789012345678901234567890"
        result = self.f.filter(text)
        assert "sk-1234567890" not in result
        assert "[REDACTED:API_KEY]" in result

    def test_no_false_positive_on_clean_text(self):
        text = "Meeting with Alice at 10:00 AM about Project Phoenix"
        assert self.f.filter(text) == text


class TestSchemas:
    def test_structured_metadata_empty(self):
        meta = StructuredMetadata()
        assert meta.meetings == []
        assert meta.files == []
        assert meta.appointments == []
        assert meta.people == []

    def test_screen_context_compatibility_alias(self):
        ctx = ScreenContext(activity_summary="Reading slides")
        assert ctx.activity_summary == "Reading slides"
        assert ctx.meetings == []

    def test_meeting_schema_validation(self):
        m = Meeting(
            title="Design Sync",
            participants=["Bob", "Charlie"],
            time="14:00 UTC",
            platform="Zoom",
            discussion_points=["Discussing API design", "DB schema review"],
            action_items=["Charlie to write tests"],
            source_frame_ids=[7],
            source_timestamps=["2026-08-14T23:36:53Z"],
            is_inferred=True,
            inferred_rationale="Inferred platform Zoom from URL",
        )
        assert m.title == "Design Sync"
        assert m.is_inferred is True
        assert m.source_frame_ids == [7]

    def test_file_activity_schema(self):
        f = FileActivity(
            file_name="schemas.py",
            file_path="metadata/schemas.py",
            document_title="Metadata schemas definitions",
            application="VS Code",
            start_time="10:00",
            end_time="10:15",
            estimated_duration="15 minutes",
        )
        assert f.file_name == "schemas.py"
        assert f.estimated_duration == "15 minutes"


class TestMetadataExtractorMocked:
    """Test MetadataExtractor behavior with mocked LLM Client responses."""

    def test_empty_text_returns_none(self):
        mock_client = MagicMock()
        extractor = MetadataExtractor(llm_client=mock_client)
        result = extractor.extract("")
        assert result is None
        mock_client.complete.assert_not_called()

    def test_invalid_json_returns_none(self):
        mock_client = MagicMock()
        mock_client.complete.return_value = "This is not JSON {{{"
        extractor = MetadataExtractor(llm_client=mock_client, max_validation_retries=0)
        
        result = extractor.extract("Some raw OCR screen log text", frame_id=42)
        assert result is None
        assert mock_client.complete.call_count == 1

    def test_validation_retries_on_malformed_llm_json(self):
        mock_client = MagicMock()
        # First call: malformed JSON
        # Second call: valid JSON matching StructuredMetadata
        mock_client.complete.side_effect = [
            "invalid JSON",
            json.dumps({
                "meetings": [{"title": "Retry Standup", "participants": ["Dave"]}],
                "files": [],
                "appointments": [],
                "people": [],
                "organizations": [],
                "projects": [],
                "urls": []
            })
        ]
        extractor = MetadataExtractor(llm_client=mock_client, max_validation_retries=1)
        
        rec = RawTextRecord(frame_id=10, timestamp=_utc(), image_path="", raw_text="Help notes")
        result = extractor.extract(rec)
        
        assert result is not None
        assert result.meetings[0].title == "Retry Standup"
        assert mock_client.complete.call_count == 2

    def test_injects_provenance_fallback(self):
        mock_client = MagicMock()
        # LLM does not populate source_frame_ids/timestamps
        mock_client.complete.return_value = json.dumps({
            "meetings": [],
            "files": [],
            "appointments": [],
            "people": [{"name": "Eve"}],
            "organizations": [],
            "projects": [],
            "urls": []
        })
        extractor = MetadataExtractor(llm_client=mock_client)
        
        rec = RawTextRecord(frame_id=99, timestamp=_utc(), image_path="", raw_text="Visible person Eve")
        result = extractor.extract(rec)
        
        assert result is not None
        assert len(result.people) == 1
        assert result.people[0].name == "Eve"
        # Provenance should be automatically injected from source record
        assert result.people[0].source_frame_ids == [99]
        assert result.people[0].source_timestamps == [rec.timestamp.isoformat()]

    def test_observed_vs_inferred_tracking(self):
        mock_client = MagicMock()
        mock_client.complete.return_value = json.dumps({
            "meetings": [
                {
                    "title": "Observed meeting",
                    "is_inferred": False,
                },
                {
                    "title": "Inferred meeting",
                    "is_inferred": True,
                    "inferred_rationale": "Inferred meeting from calendar invite",
                }
            ],
            "files": [],
            "appointments": [],
            "people": [],
            "organizations": [],
            "projects": [],
            "urls": []
        })
        extractor = MetadataExtractor(llm_client=mock_client)
        result = extractor.extract("Some notes", frame_id=1)
        
        assert result is not None
        assert len(result.meetings) == 2
        assert result.meetings[0].is_inferred is False
        assert result.meetings[1].is_inferred is True
        assert result.meetings[1].inferred_rationale == "Inferred meeting from calendar invite"

    def test_heuristic_url_and_meeting_extraction(self):
        # Offline heuristic extraction
        extractor = MetadataExtractor(llm_client=None)
        rec = RawTextRecord(
            frame_id=1,
            timestamp=_utc(),
            image_path="",
            application="Brave",
            window_title="Meet - abc-defg-hij - Brave",
            raw_text="Google Meet | meet.google.com/abc-defg-hij\nSprint Planning\nAlso check github.com/owner/repo",
        )
        meta = extractor.extract(rec)
        assert meta is not None
        # Check URL extraction
        urls = [u.url for u in meta.urls]
        assert any("meet.google.com/abc-defg-hij" in u for u in urls)
        assert any("github.com/owner/repo" in u for u in urls)
        # Check meeting extraction
        assert len(meta.meetings) >= 1
        assert meta.meetings[0].meeting_link == "https://meet.google.com/abc-defg-hij"
