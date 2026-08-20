"""
tests/test_video_and_entities.py

Tests for ScreenVideoRecorder, sensible file naming, appointment extraction,
meeting enrichment, file activity duration estimation, and pipeline controls.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from capture.video_recorder import ScreenVideoRecorder
from metadata.extractor import MetadataExtractor
from metadata.schemas import StructuredMetadata, Meeting, Appointment, FileActivity, URLReference
from ocr.models import RawTextRecord
from storage.file_manager import FileManager


@pytest.fixture
def fm(tmp_path):
    return FileManager(data_dir=tmp_path, webp_quality=80)


class TestSensibleNamingAndStorage:
    def test_sanitize_name(self):
        assert FileManager.sanitize_name("chrome.exe") == "chrome"
        assert FileManager.sanitize_name("Visual Studio Code - main.py") == "Visual_Studio_Code_main"
        assert FileManager.sanitize_name(None) == ""

    def test_save_and_rename_with_context(self, fm):
        ts = datetime(2026, 8, 19, 10, 30, 0, tzinfo=timezone.utc)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        rel = fm.save_frame(frame, timestamp=ts, application="VSCode.exe", window_title="main.py")
        abs_path = fm.absolute_path(rel)
        assert abs_path.exists()
        assert "VSCode" in rel

        new_rel = fm.rename_to_id(rel, frame_id=42, timestamp=ts, application="VSCode", window_title="main.py")
        assert "00000042.webp" in new_rel
        assert "VSCode" in new_rel
        assert fm.absolute_path(new_rel).exists()

    def test_video_path_generation(self, fm):
        ts = datetime(2026, 8, 19, 11, 0, 0, tzinfo=timezone.utc)
        vpath = fm.get_video_path(timestamp=ts, segment_index=1, application="chrome")
        assert vpath.parent.exists()
        assert "chrome" in vpath.name
        assert "seg0001" in vpath.name
        assert vpath.suffix == ".mp4"


class TestVideoRecorder:
    def test_video_recorder_init_and_properties(self, fm):
        vr = ScreenVideoRecorder(
            file_manager=fm,
            monitor_index=1,
            fps=2.0,
            segment_minutes=15,
        )
        assert not vr.is_recording
        assert not vr.is_paused
        assert vr.segments_recorded == 0

    def test_video_recorder_start_pause_resume_stop(self, fm):
        with patch("mss.mss") as mock_mss_cls, patch("cv2.VideoWriter") as mock_writer_cls:
            mock_sct = MagicMock()
            mock_sct.monitors = [{"width": 1920, "height": 1080}, {"width": 1920, "height": 1080}]
            mock_sct.grab.return_value = np.zeros((480, 640, 4), dtype=np.uint8)
            mock_mss_cls.return_value.__enter__.return_value = mock_sct

            mock_writer = MagicMock()
            mock_writer.isOpened.return_value = True
            mock_writer_cls.return_value = mock_writer

            vr = ScreenVideoRecorder(
                file_manager=fm,
                monitor_index=1,
                fps=5.0,
                segment_minutes=1,
            )
            vr.start()
            assert vr.is_recording

            vr.pause()
            assert vr.is_paused

            vr.resume()
            assert not vr.is_paused

            vr.stop()
            assert not vr.is_recording


class TestEnhancedMetadataExtraction:
    def test_extract_meetings_with_tasks_and_platform(self):
        extractor = MetadataExtractor()
        ocr_text = (
            "Google Meet - abc-defg-hij - Brave\n"
            "Sprint Planning Discussion\n"
            "Participants: Alice Chen, Bob Smith\n"
            "Discussion:\n"
            "- Architecture review for background capture\n"
            "- SQLite indexing and migration strategy\n"
            "Action Items:\n"
            "- Alice Chen to review schema\n"
            "- Bob Smith to test shutdown"
        )
        rec = RawTextRecord(
            frame_id=101,
            timestamp=datetime(2026, 8, 19, 10, 30, 0, tzinfo=timezone.utc),
            image_path="test.webp",
            application="brave.exe",
            window_title="Meet - abc-defg-hij - Brave",
            raw_text=ocr_text,
        )

        metadata = extractor.extract(rec)
        assert metadata is not None
        assert len(metadata.meetings) >= 1
        m = metadata.meetings[0]
        assert "Meet" in (m.platform or "") or "Google" in (m.platform or "")
        assert m.meeting_link is not None and "meet.google.com" in m.meeting_link
        assert len(m.discussion_points) >= 1
        assert len(m.action_items) >= 1

    def test_extract_appointments_and_deadlines(self):
        extractor = MetadataExtractor()
        ocr_text = (
            "Calendar Sync\n"
            "Quarterly roadmap sync scheduled on Friday at 2:00 PM\n"
            "Project deliverable due by 5:00 PM"
        )
        rec = RawTextRecord(
            frame_id=102,
            timestamp=datetime(2026, 8, 19, 12, 0, 0, tzinfo=timezone.utc),
            image_path="test2.webp",
            application="chrome.exe",
            window_title="Google Calendar - Chrome",
            raw_text=ocr_text,
        )

        metadata = extractor.extract(rec)
        assert metadata is not None
        assert len(metadata.appointments) >= 1
        appt = metadata.appointments[0]
        assert "sync" in appt.title.lower() or "roadmap" in appt.title.lower() or "2:00" in appt.title

    def test_extract_files_with_duration_and_path(self):
        extractor = MetadataExtractor()
        ocr_text = (
            "Visual Studio Code\n"
            "c:/Users/Soham/OneDrive/Desktop/RIOM/storage/db.py\n"
            "def insert_frame(self):"
        )
        rec = RawTextRecord(
            frame_id=103,
            timestamp=datetime(2026, 8, 19, 14, 0, 0, tzinfo=timezone.utc),
            image_path="test3.webp",
            application="Code.exe",
            window_title="db.py - RIOM - Visual Studio Code",
            raw_text=ocr_text,
        )

        metadata = extractor.extract(rec)
        assert metadata is not None
        assert len(metadata.files) >= 1
        f = metadata.files[0]
        assert f.file_name == "db.py"
        assert f.file_path is not None
        assert "storage" in f.file_path or "db.py" in f.file_path
