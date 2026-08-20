"""
tests/test_meeting_notes.py

Unit tests for Raw Speech-to-Text Meeting Transcript Generation.
Verifies that only real OCR/speech content is captured — no AI filler.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from processing.meeting_notes import (
    MeetingNotesGenerator,
    MeetingTranscriptSaver,
    extract_speech_lines,
    sanitize_filename,
    _is_ui_noise,
    _looks_like_speech,
)


class TestSanitizeFilename:
    def test_replaces_special_chars(self):
        assert sanitize_filename("Sprint Planning: Q3 Sync / Review") == "Sprint_Planning_Q3_Sync_Review"

    def test_replaces_em_dash(self):
        assert sanitize_filename("Sprint Planning — Q3 RIOM Engineering") == "Sprint_Planning_-_Q3_RIOM_Engineering"

    def test_strips_symbols(self):
        assert sanitize_filename("Special * & ? <> | chars") == "Special_chars"

    def test_empty_returns_default(self):
        assert sanitize_filename("") == "Meeting_Transcript"


class TestUiNoiseFiltering:
    def test_blank_line_is_noise(self):
        assert _is_ui_noise("") is True
        assert _is_ui_noise("   ") is True

    def test_single_chars_are_noise(self):
        assert _is_ui_noise("v") is True
        assert _is_ui_noise("AB") is True

    def test_meeting_controls_are_noise(self):
        assert _is_ui_noise("Mute") is True
        assert _is_ui_noise("End Call") is True
        assert _is_ui_noise("Captions") is True
        assert _is_ui_noise("Leave") is True

    def test_browser_ui_is_noise(self):
        assert _is_ui_noise("Brave") is True
        assert _is_ui_noise("New Tab") is True
        assert _is_ui_noise("Home") is True

    def test_meet_room_code_is_noise(self):
        assert _is_ui_noise("cqi-vzht-ihq") is True

    def test_url_fragment_is_noise(self):
        assert _is_ui_noise("meet.google.com/cqi-vzht-ihq") is True
        assert _is_ui_noise("https://example.com") is True

    def test_real_sentence_is_not_noise(self):
        assert _is_ui_noise("So let's start with the sprint planning discussion") is False
        assert _is_ui_noise("Can you share your screen please?") is False


class TestSpeechDetection:
    def test_real_speech_passes(self):
        assert _looks_like_speech("So we need to finish the deployment by Friday") is True
        assert _looks_like_speech("I think this approach would work well for the team") is True
        assert _looks_like_speech("Can everyone see my screen right now?") is True

    def test_garbage_ocr_fails(self):
        assert _looks_like_speech("S (73) v") is False
        assert _looks_like_speech("wexbw") is False
        assert _looks_like_speech("rneet.google.corn") is False

    def test_too_short_fails(self):
        assert _looks_like_speech("hi") is False
        assert _looks_like_speech("ok") is False


class TestExtractSpeechLines:
    def test_extracts_real_speech(self):
        raw = """
S (73) v
Home A Shc
Mute
Can everyone see my screen right now?
I think we should push the deadline to next week.
cqi-vzht-ihq
End Call
So the main issue is with the authentication flow.
        """
        result = extract_speech_lines(raw)
        assert "Can everyone see my screen right now?" in result
        assert "I think we should push the deadline to next week." in result
        assert "So the main issue is with the authentication flow." in result

    def test_no_ui_chrome_in_output(self):
        raw = "Mute\nEnd Call\nChrome\nS (73)\nbroadcast"
        result = extract_speech_lines(raw)
        assert result == [] or all(len(l) > 4 for l in result)

    def test_deduplicates_lines(self):
        raw = """We need to fix the login bug.
We need to fix the login bug.
We need to fix the login bug."""
        result = extract_speech_lines(raw)
        assert len(result) == 1
        assert result[0] == "We need to fix the login bug."

    def test_replaces_shorter_with_longer_duplicate(self):
        raw = """I think we should
I think we should review the API contracts today."""
        result = extract_speech_lines(raw)
        # Should keep the longer version
        assert any("review the API contracts today" in l for l in result)


class TestMeetingTranscriptSaver:
    @pytest.fixture
    def tmp_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            yield Path(tmp)

    def test_generate_raw_transcript_has_header(self, tmp_dir):
        saver = MeetingTranscriptSaver(output_dir=tmp_dir)
        frames = [
            {
                "timestamp": "2026-08-20T10:18:05+00:00",
                "text": "So let's discuss the sprint goals for this quarter.",
                "window_title": "Sprint Planning",
            }
        ]
        txt = saver.generate_raw_transcript_txt(
            meeting_title="Sprint Planning",
            raw_frames=frames,
            platform="Google Meet",
            meeting_link="https://meet.google.com/cqi-vzht-ihq",
        )
        assert "Meeting: Sprint Planning" in txt
        assert "Platform: Google Meet" in txt
        assert "Link:     https://meet.google.com/cqi-vzht-ihq" in txt
        assert "─" * 20 in txt

    def test_generate_transcript_contains_speech(self, tmp_dir):
        saver = MeetingTranscriptSaver(output_dir=tmp_dir)
        frames = [
            {
                "timestamp": "2026-08-20T10:18:05+00:00",
                "text": "Mute\nEnd Call\nSo we need to review the API contracts before Friday.\nNew Tab",
                "window_title": "Meet",
            }
        ]
        txt = saver.generate_raw_transcript_txt("Team Sync", frames)
        assert "So we need to review the API contracts before Friday." in txt
        assert "Mute" not in txt
        assert "End Call" not in txt

    def test_save_transcript_creates_txt_file(self, tmp_dir):
        saver = MeetingTranscriptSaver(output_dir=tmp_dir)
        frames = [
            {"timestamp": "2026-08-20T10:18:05+00:00", "text": "This is the actual discussion from the meeting.", "window_title": "Meet"}
        ]
        p = saver.save_transcript_txt(
            meeting_title="Backend Sync",
            raw_frames=frames,
            source_timestamps=["2026-08-20T10:18:05+00:00"],
        )
        assert p.exists()
        assert p.suffix == ".txt"
        assert "Backend_Sync" in p.name
        content = p.read_text(encoding="utf-8")
        assert "Meeting: Backend Sync" in content

    def test_empty_frames_produces_fallback_message(self, tmp_dir):
        saver = MeetingTranscriptSaver(output_dir=tmp_dir)
        txt = saver.generate_raw_transcript_txt("Empty Meeting", raw_frames=[])
        assert "No speech or subtitle text captured" in txt

    def test_list_saved_transcripts(self, tmp_dir):
        saver = MeetingTranscriptSaver(output_dir=tmp_dir)
        frames = [{"timestamp": "2026-08-20T10:18:05+00:00", "text": "We should plan this properly.", "window_title": "Meet"}]
        saver.save_transcript_txt("Meeting A", frames, source_timestamps=["2026-08-20"])
        saver.save_transcript_txt("Meeting B", frames, source_timestamps=["2026-08-20"])
        listed = saver.list_saved_transcripts()
        assert len(listed) == 2
        assert all(n["format"] == ".txt" for n in listed)


class TestMeetingNotesGeneratorBackwardCompat:
    """Verify that the legacy MeetingNotesGenerator API still works and produces .txt output."""

    @pytest.fixture
    def tmp_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            yield Path(tmp)

    def test_save_meeting_notes_produces_txt(self, tmp_dir):
        from metadata.schemas import Meeting
        gen = MeetingNotesGenerator(output_dir=tmp_dir)
        meeting = Meeting(
            title="Backend Architecture Sync",
            platform="Zoom",
            participants=["Rahul Sharma"],
            source_frame_ids=[55],
            source_timestamps=["2026-08-20T14:00:00Z"],
        )
        saved_file = gen.save_meeting_notes(meeting, raw_context="I think we should consider microservices here.")
        assert saved_file.exists()
        assert saved_file.suffix == ".txt"
        assert "Backend_Architecture_Sync" in saved_file.name

    def test_process_and_save_all_meetings(self, tmp_dir):
        from metadata.schemas import Meeting, StructuredMetadata
        m1 = Meeting(title="Sprint Planning", source_frame_ids=[1], source_timestamps=["2026-08-20T10:00:00Z"])
        m2 = Meeting(title="Client Intro Call", source_frame_ids=[2], source_timestamps=["2026-08-20T11:00:00Z"])
        meta = StructuredMetadata(meetings=[m1, m2])

        gen = MeetingNotesGenerator(output_dir=tmp_dir)
        saved = gen.process_and_save_all(
            meta,
            raw_context_map={
                1: "We should plan the sprint goals for this quarter.",
                2: "Let me introduce our team and what we are building.",
            },
        )
        assert len(saved) == 2
        for f in saved:
            assert f.exists()
            assert f.suffix == ".txt"
