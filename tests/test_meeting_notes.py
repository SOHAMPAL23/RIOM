"""
tests/test_meeting_notes.py

Unit tests for Automated Ambient Meeting Notes Generation and File Persistence.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from metadata.schemas import Meeting, StructuredMetadata
from processing.meeting_notes import MeetingNotesGenerator, sanitize_filename


class TestMeetingNotes:

    @pytest.fixture
    def temp_notes_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            yield Path(tmp)

    def test_sanitize_filename(self):
        assert sanitize_filename("Sprint Planning: Q3 Sync / Review") == "Sprint_Planning_Q3_Sync_Review"
        assert sanitize_filename("Sprint Planning — Q3 RIOM Engineering") == "Sprint_Planning_-_Q3_RIOM_Engineering"
        assert sanitize_filename("Special * & ? <> | chars") == "Special_chars"
        assert sanitize_filename("") == "Meeting_Notes"

    def test_generate_markdown_structure(self, temp_notes_dir):
        meeting = Meeting(
            title="Q3 Roadmap Strategy",
            platform="Google Meet",
            meeting_link="https://meet.google.com/abc-defg-hij",
            participants=["Alice Chen", "Bob Smith"],
            emails=["alice@example.com", "bob@example.com"],
            discussion_points=[
                "Database indexing and migration strategy",
                "Frontend PySide6 dark mode polish",
            ],
            action_items=[
                "Alice to review PR #42",
                "Bob to benchmark OCR latency",
            ],
            source_frame_ids=[101, 102],
            source_timestamps=["2026-08-20T10:30:00Z"],
        )

        gen = MeetingNotesGenerator(output_dir=temp_notes_dir)
        raw_ctx = "Google Meet: Q3 Roadmap Strategy\nAlice Chen: Let's discuss database indexing."
        md = gen.generate_markdown(meeting, raw_context=raw_ctx)

        assert "# 🎙️ Meeting Notes: Q3 Roadmap Strategy" in md
        assert "Google Meet" in md
        assert "https://meet.google.com/abc-defg-hij" in md
        assert "Alice Chen" in md
        assert "bob@example.com" in md
        assert "Database indexing and migration strategy" in md
        assert "- [ ] Alice to review PR #42" in md
        assert "#101" in md
        assert "Ambient Screen Context Snippet" in md

    def test_save_meeting_notes_to_disk(self, temp_notes_dir):
        meeting = Meeting(
            title="Backend Architecture Sync",
            platform="Zoom",
            participants=["Rahul Sharma"],
            source_frame_ids=[55],
            source_timestamps=["2026-08-20T14:00:00Z"],
        )

        gen = MeetingNotesGenerator(output_dir=temp_notes_dir)
        saved_file = gen.save_meeting_notes(meeting)

        assert saved_file.exists()
        assert saved_file.suffix == ".md"
        assert "Backend_Architecture_Sync" in saved_file.name

        content = saved_file.read_text(encoding="utf-8")
        assert "Backend Architecture Sync" in content
        assert "Rahul Sharma" in content

    def test_process_and_save_all_meetings(self, temp_notes_dir):
        m1 = Meeting(title="Sprint Planning", participants=["Alice"], source_frame_ids=[1])
        m2 = Meeting(title="Client Intro Call", participants=["Carol"], source_frame_ids=[2])
        meta = StructuredMetadata(meetings=[m1, m2])

        gen = MeetingNotesGenerator(output_dir=temp_notes_dir)
        saved = gen.process_and_save_all(meta, raw_context_map={1: "Sprint Planning context", 2: "Client Call context"})

        assert len(saved) == 2
        for f in saved:
            assert f.exists()
            assert f.is_file()

        listed = gen.list_saved_notes()
        assert len(listed) == 2
        assert all("filename" in n and "path" in n for n in listed)
