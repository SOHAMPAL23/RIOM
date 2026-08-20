"""
tests/test_verifier.py

Unit tests for Stage 3.5 Metadata Verification Layer.

Checks:
- normalize_text ignores case, spaces, and punctuation.
- is_supported finds substrings deterministically.
- VerificationStatus logic:
  * verified: all fields found.
  * partially_supported: core field (e.g. name) found, but details missing.
  * unsupported: core field not found.
- The "Rahul Sharma" observed vs inferred role check.
- Cleaning function:
  * unsupported facts are excluded.
  * partially_supported facts are cleaned.
- FactEvidence object population.
"""
from __future__ import annotations

import pytest

from metadata.verifier import MetadataVerifier, normalize_text, is_supported
from metadata.schemas import (
    StructuredMetadata,
    Meeting,
    Person,
    FileActivity,
    VerificationStatus,
)


class TestVerifierHelpers:

    def test_normalize_text_case_spaces_punctuation(self):
        text = "Hello, World!  This is: a test."
        assert normalize_text(text) == "hello world this is a test"

    def test_is_supported_exact_substring(self):
        text = "Meeting with Alice at 3pm"
        assert is_supported(text, "Alice") is True
        assert is_supported(text, "meeting with alice") is True
        assert is_supported(text, "Bob") is False

    def test_is_supported_inferred_phrase(self):
        text = "Rahul Sharma was visible."
        # Core name matches
        assert is_supported(text, "Rahul Sharma") is True
        # Inferred statement has words not present in the raw text -> should not match!
        assert is_supported(text, "Rahul is the project manager") is False


class TestMetadataVerifier:

    def setup_method(self):
        self.verifier = MetadataVerifier()

    def test_verify_person_status_verified(self):
        meta = StructuredMetadata(
            people=[Person(name="Rahul Sharma", email="rahul@example.com", source_frame_ids=[1])]
        )
        raw_text_map = {1: "Rahul Sharma (rahul@example.com) is in the chat."}
        verified, evidences = self.verifier.verify(meta, raw_text_map)

        assert len(verified.people) == 1
        assert verified.people[0].name == "Rahul Sharma"
        assert len(evidences) == 1
        assert evidences[0].verification_status == VerificationStatus.VERIFIED
        assert evidences[0].evidence_text is not None

    def test_verify_person_inferred_role_unsupported(self):
        """
        Observed: "Rahul Sharma"
        Inferred: "Rahul is the project manager" (stored as organization or checked)
        If the raw text only contains "Rahul Sharma", the inferred statement must be rejected.
        """
        # We model the inferred role as organization = "Project Manager"
        meta = StructuredMetadata(
            people=[Person(name="Rahul Sharma", organization="Project Manager", source_frame_ids=[1])]
        )
        raw_text_map = {1: "Rahul Sharma checked in."}
        verified, evidences = self.verifier.verify(meta, raw_text_map)

        # Core name "Rahul Sharma" is supported -> partially_supported
        # Organization "Project Manager" is NOT in raw text.
        assert len(verified.people) == 1
        assert verified.people[0].name == "Rahul Sharma"
        # The unsupported organization field must have been cleared!
        assert verified.people[0].organization is None
        
        assert len(evidences) == 1
        assert evidences[0].verification_status == VerificationStatus.PARTIALLY_SUPPORTED
        assert "organization" in evidences[0].unsupported_fields

    def test_verify_meeting_unsupported(self):
        meta = StructuredMetadata(
            meetings=[Meeting(title="Standup Sync Meeting", source_frame_ids=[1])]
        )
        # Raw text has unrelated info -> core title not matched -> unsupported
        raw_text_map = {1: "Just reading some docs on python."}
        verified, evidences = self.verifier.verify(meta, raw_text_map)

        # Unsupported facts are completely removed from the verified output list!
        assert len(verified.meetings) == 0
        assert len(evidences) == 1
        assert evidences[0].verification_status == VerificationStatus.UNSUPPORTED
        assert "title" in evidences[0].unsupported_fields

    def test_verify_file_activity_partially_supported(self):
        meta = StructuredMetadata(
            files=[
                FileActivity(
                    file_name="pipeline.py",
                    application="VS Code",
                    estimated_duration="5 minutes",
                    source_frame_ids=[1]
                )
            ]
        )
        # raw text has pipeline.py in VS Code, but does not explicitly state duration
        raw_text_map = {1: "VS Code editing pipeline.py."}
        verified, evidences = self.verifier.verify(meta, raw_text_map)

        assert len(verified.files) == 1
        assert verified.files[0].file_name == "pipeline.py"
        assert verified.files[0].application == "VS Code"
        # Estimated duration is unsupported, so it must be cleared!
        assert verified.files[0].estimated_duration is None

        assert len(evidences) == 1
        assert evidences[0].verification_status == VerificationStatus.PARTIALLY_SUPPORTED
        assert "estimated_duration" in evidences[0].unsupported_fields

    def test_timestamps_map_propagation(self):
        meta = StructuredMetadata(
            people=[Person(name="Alice", source_frame_ids=[5])]
        )
        raw_text_map = {5: "Alice is here."}
        timestamps_map = {5: "2026-08-14T23:40:00Z"}
        
        _, evidences = self.verifier.verify(meta, raw_text_map, timestamps_map)
        assert evidences[0].source_timestamp == "2026-08-14T23:40:00Z"

    def test_verify_url_scheme_agnostic(self):
        from metadata.schemas import URLReference
        meta = StructuredMetadata(
            urls=[
                URLReference(url="https://docs.google.com/document/d/riom-brief", title="Project Brief", source_frame_ids=[1]),
                URLReference(url="https://meet.google.com/abc-defg-hij", source_frame_ids=[1]),
            ]
        )
        # Raw text contains scheme-less URLs
        raw_text_map = {1: "Review docs.google.com/document/d/riom-brief on Google Docs. Also join meet.google.com/abc-defg-hij"}
        verified, evidences = self.verifier.verify(meta, raw_text_map)

        assert len(verified.urls) == 2
        assert verified.urls[0].url == "https://docs.google.com/document/d/riom-brief"
        assert verified.urls[1].url == "https://meet.google.com/abc-defg-hij"
        assert all(ev.verification_status in (VerificationStatus.VERIFIED, VerificationStatus.PARTIALLY_SUPPORTED) for ev in evidences)

    def test_verify_meeting_with_link(self):
        meta = StructuredMetadata(
            meetings=[
                Meeting(
                    title="Sprint Planning",
                    platform="Google Meet",
                    meeting_link="https://meet.google.com/riom-sprint-plan",
                    source_frame_ids=[1],
                )
            ]
        )
        raw_text_map = {1: "Google Meet | meet.google.com/riom-sprint-plan\nSprint Planning"}
        verified, evidences = self.verifier.verify(meta, raw_text_map)

        assert len(verified.meetings) == 1
        assert verified.meetings[0].meeting_link == "https://meet.google.com/riom-sprint-plan"
        assert evidences[0].verification_status == VerificationStatus.VERIFIED
