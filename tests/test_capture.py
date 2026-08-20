"""
tests/test_capture.py

Unit tests for the capture module — Stage 1, including smart change detection.

Coverage
--------
ChangeDetector
    - Identical frames → NO_CHANGE, should_save=False
    - Slightly changed frames → NO_CHANGE below threshold
    - Significantly changed frames → VISUAL_CHANGE, should_save=True
    - Application change overrides visual threshold → APPLICATION_CHANGE
    - Periodic capture fires after max_capture_interval → PERIODIC_CAPTURE
    - Manual capture via force_next() → frame is accepted
    - Idle detection → IDLE reason returned after idle_threshold
    - compute_diff_score public utility returns correct range
    - reset() clears all state
    - Thread safety under concurrent calls

CaptureRecord (Pydantic model)
    - Construction, JSON round-trip, capture_reason / diff_score fields

WindowInfoProvider
    - Never raises on any platform
    - Correctly falls through to stub on unknown platform

FileManager
    - save_frame creates file, returns relative path
    - rename_to_id produces {id:08d}.webp filename
    - delete_frame / disk_usage / prune_before

Database
    - insert_frame stores capture_reason and diff_score
    - get_frames_by_reason filters correctly

ScreenRecorder lifecycle
    - start / stop / pause / resume / force_capture (mocked I/O)
    - reason_counts counter updated after captures
"""
from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Frame factories
# ---------------------------------------------------------------------------

def _bgr(color=(128, 128, 128), size=(640, 480)) -> np.ndarray:
    """Solid-colour BGR frame."""
    f = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    f[:] = color
    return f


def _noisy(base=(128, 128, 128), noise=5, size=(640, 480)) -> np.ndarray:
    """Slightly noisy frame — should be below typical threshold."""
    rng = np.random.default_rng(42)
    f = np.full((size[1], size[0], 3), base, dtype=np.int16)
    f += rng.integers(-noise, noise + 1, f.shape, dtype=np.int16)
    return np.clip(f, 0, 255).astype(np.uint8)


def _utc() -> datetime:
    return datetime.now(timezone.utc)


# ===========================================================================
# ChangeDetector tests
# ===========================================================================

class TestChangeDetectorIdenticalFrame:
    """Identical frames must be rejected (NO_CHANGE)."""

    def setup_method(self):
        from capture.change_detector import ChangeDetector
        self.det = ChangeDetector(visual_threshold=0.02)

    def test_first_frame_always_saved(self):
        a = self.det.analyse(_bgr())
        assert a.should_save is True

    def test_identical_second_frame_not_saved(self):
        frame = _bgr((64, 64, 64))
        self.det.analyse(frame)          # seed
        result = self.det.analyse(frame)
        assert result.should_save is False

    def test_identical_reason_is_no_change(self):
        from capture.change_detector import CaptureReason
        frame = _bgr()
        self.det.analyse(frame)
        result = self.det.analyse(frame)
        assert result.reason == CaptureReason.NO_CHANGE

    def test_diff_score_of_identical_frame_is_zero(self):
        frame = _bgr((200, 100, 50))
        self.det.analyse(frame)
        result = self.det.analyse(frame)
        assert result.diff_score == pytest.approx(0.0, abs=1e-6)


class TestChangeDetectorSlightChange:
    """Slightly changed frames that stay below the threshold."""

    def setup_method(self):
        from capture.change_detector import ChangeDetector
        self.det = ChangeDetector(visual_threshold=0.05)  # Higher threshold

    def test_small_noise_below_threshold_not_saved(self):
        base = _bgr((128, 128, 128))
        self.det.analyse(base)
        slightly_different = _noisy((128, 128, 128), noise=2)
        result = self.det.analyse(slightly_different)
        assert result.should_save is False

    def test_diff_score_is_low_for_slight_change(self):
        base = _bgr((128, 128, 128))
        self.det.analyse(base)
        slight = _noisy((128, 128, 128), noise=2)
        result = self.det.analyse(slight)
        assert result.diff_score < 0.05


class TestChangeDetectorSignificantChange:
    """Significantly changed frames must be accepted (VISUAL_CHANGE)."""

    def setup_method(self):
        from capture.change_detector import ChangeDetector
        self.det = ChangeDetector(visual_threshold=0.02)

    def test_black_to_white_is_accepted(self):
        from capture.change_detector import CaptureReason
        self.det.analyse(_bgr((0, 0, 0)))
        result = self.det.analyse(_bgr((255, 255, 255)))
        assert result.should_save is True
        assert result.reason == CaptureReason.VISUAL_CHANGE

    def test_diff_score_high_for_large_change(self):
        self.det.analyse(_bgr((0, 0, 0)))
        result = self.det.analyse(_bgr((255, 255, 255)))
        # Black → white = max possible diff
        assert result.diff_score > 0.9

    def test_moderate_change_above_threshold(self):
        from capture.change_detector import CaptureReason
        self.det.analyse(_bgr((100, 100, 100)))
        result = self.det.analyse(_bgr((200, 200, 200)))
        assert result.should_save is True
        assert result.reason == CaptureReason.VISUAL_CHANGE

    def test_consecutive_changes_each_accepted(self):
        self.det.analyse(_bgr((0, 0, 0)))
        r1 = self.det.analyse(_bgr((255, 255, 255)))
        r2 = self.det.analyse(_bgr((0, 0, 0)))
        assert r1.should_save is True
        assert r2.should_save is True


class TestChangeDetectorApplicationChange:
    """App change must override visual threshold — always save."""

    def setup_method(self):
        from capture.change_detector import ChangeDetector
        # High threshold so identical frames would normally be rejected
        self.det = ChangeDetector(visual_threshold=0.99)

    def test_app_change_saves_even_below_visual_threshold(self):
        from capture.change_detector import CaptureReason
        frame = _bgr()
        # Seed with app=chrome
        self.det.analyse(frame, application="chrome")
        # Same frame, different app
        result = self.det.analyse(frame, application="vscode")
        assert result.should_save is True
        assert result.reason == CaptureReason.APPLICATION_CHANGE
        assert result.app_changed is True

    def test_same_app_no_change_not_saved(self):
        from capture.change_detector import CaptureReason
        frame = _bgr()
        self.det.analyse(frame, application="chrome")
        result = self.det.analyse(frame, application="chrome")
        assert result.should_save is False

    def test_first_app_does_not_trigger_app_change(self):
        """First frame has no previous app — can't be an app-change save."""
        from capture.change_detector import CaptureReason
        result = self.det.analyse(_bgr(), application="chrome")
        # First frame is always VISUAL_CHANGE (diff=1.0)
        assert result.app_changed is False

    def test_app_change_recorded_after_no_previous_app(self):
        from capture.change_detector import CaptureReason
        frame = _bgr()
        self.det.analyse(frame, application=None)   # No app info
        self.det.analyse(frame, application=None)   # Still no info
        # Now app appears — can't be app-change without a prior known app
        result = self.det.analyse(frame, application="chrome")
        assert result.app_changed is False


class TestChangeDetectorPeriodicCapture:
    """Periodic forced capture fires after max_capture_interval."""

    def test_periodic_fires_after_interval(self):
        from capture.change_detector import ChangeDetector, CaptureReason
        det = ChangeDetector(visual_threshold=0.99, max_capture_interval=0.05)
        frame = _bgr()
        det.analyse(frame)            # Seed — sets last_accepted_time
        time.sleep(0.1)               # Exceed 0.05 s interval
        result = det.analyse(frame)   # Same frame, but timeout elapsed
        assert result.should_save is True
        assert result.reason == CaptureReason.PERIODIC_CAPTURE

    def test_periodic_disabled_when_zero(self):
        from capture.change_detector import ChangeDetector, CaptureReason
        det = ChangeDetector(visual_threshold=0.99, max_capture_interval=0.0)
        frame = _bgr()
        det.analyse(frame)
        time.sleep(0.05)
        result = det.analyse(frame)
        assert result.should_save is False

    def test_periodic_resets_after_save(self):
        from capture.change_detector import ChangeDetector, CaptureReason
        det = ChangeDetector(visual_threshold=0.99, max_capture_interval=0.05)
        frame = _bgr()
        det.analyse(frame)
        time.sleep(0.1)
        det.analyse(frame)   # Periodic save — resets timer
        # Should NOT fire again immediately
        result = det.analyse(frame)
        assert result.should_save is False


class TestChangeDetectorIdleDetection:
    """Idle detection tags frames with IDLE reason after threshold."""

    def test_idle_reason_returned_after_threshold(self):
        from capture.change_detector import ChangeDetector, CaptureReason
        det = ChangeDetector(visual_threshold=0.99, idle_threshold_seconds=0.05)
        frame = _bgr()
        det.analyse(frame)          # Seed (first call)
        det.analyse(frame)          # Second call sets _idle_since = now
        time.sleep(0.1)             # Exceed idle threshold
        result = det.analyse(frame) # Third call returns IDLE
        assert result.reason == CaptureReason.IDLE
        assert result.should_save is False


    def test_activity_clears_idle_state(self):
        from capture.change_detector import ChangeDetector, CaptureReason
        det = ChangeDetector(visual_threshold=0.02, idle_threshold_seconds=0.05)
        frame = _bgr()
        det.analyse(frame)
        time.sleep(0.1)
        # Now a big change — should clear idle and save
        result = det.analyse(_bgr((255, 255, 255)))
        assert result.reason == CaptureReason.VISUAL_CHANGE


class TestChangeDetectorUtilities:

    def test_compute_diff_score_same_frames(self):
        from capture.change_detector import ChangeDetector
        det = ChangeDetector()
        frame = _bgr()
        score = det.compute_diff_score(frame, frame.copy())
        assert score == pytest.approx(0.0, abs=1e-6)

    def test_compute_diff_score_opposite_frames(self):
        from capture.change_detector import ChangeDetector
        det = ChangeDetector()
        score = det.compute_diff_score(_bgr((0, 0, 0)), _bgr((255, 255, 255)))
        assert score > 0.9

    def test_diff_score_in_range(self):
        from capture.change_detector import ChangeDetector
        det = ChangeDetector()
        score = det.compute_diff_score(_bgr((50, 100, 150)), _bgr((200, 80, 30)))
        assert 0.0 <= score <= 1.0

    def test_reset_clears_state(self):
        from capture.change_detector import ChangeDetector
        det = ChangeDetector(visual_threshold=0.02)
        frame = _bgr()
        det.analyse(frame)
        det.analyse(frame)  # Rejected
        det.reset()
        result = det.analyse(frame)
        assert result.should_save is True

    def test_thread_safety(self):
        from capture.change_detector import ChangeDetector
        det = ChangeDetector(visual_threshold=0.01)
        errors = []

        def worker():
            try:
                for _ in range(20):
                    det.analyse(_bgr(), application="chrome")
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors, f"Thread errors: {errors}"


# ===========================================================================
# CaptureRecord model
# ===========================================================================

class TestCaptureRecord:

    def test_default_capture_reason_is_visual_change(self):
        from capture.models import CaptureRecord
        from capture.change_detector import CaptureReason
        rec = CaptureRecord(timestamp=_utc(), image_path="x.webp", width=1920, height=1080)
        assert rec.capture_reason == CaptureReason.VISUAL_CHANGE

    def test_application_change_reason_stored(self):
        from capture.models import CaptureRecord
        from capture.change_detector import CaptureReason
        rec = CaptureRecord(
            timestamp=_utc(),
            image_path="x.webp",
            width=1920,
            height=1080,
            capture_reason=CaptureReason.APPLICATION_CHANGE,
            diff_score=0.003,
        )
        assert rec.capture_reason == CaptureReason.APPLICATION_CHANGE
        assert rec.diff_score == pytest.approx(0.003)

    def test_json_roundtrip_preserves_reason(self):
        from capture.models import CaptureRecord
        from capture.change_detector import CaptureReason
        rec = CaptureRecord(
            timestamp=_utc(),
            image_path="x.webp",
            width=100,
            height=100,
            capture_reason=CaptureReason.PERIODIC_CAPTURE,
            diff_score=0.001,
        )
        restored = CaptureRecord.model_validate_json(rec.model_dump_json())
        assert restored.capture_reason == CaptureReason.PERIODIC_CAPTURE

    def test_manual_capture_reason(self):
        from capture.models import CaptureRecord
        from capture.change_detector import CaptureReason
        rec = CaptureRecord(
            timestamp=_utc(),
            image_path="x.webp",
            width=100,
            height=100,
            capture_reason=CaptureReason.MANUAL_CAPTURE,
        )
        assert rec.capture_reason == CaptureReason.MANUAL_CAPTURE


# ===========================================================================
# WindowInfoProvider
# ===========================================================================

class TestWindowInfoProvider:

    def test_never_raises(self):
        from capture.window_info import WindowInfoProvider
        p = WindowInfoProvider()
        result = p.get()   # Must not raise on any platform
        assert result is None or hasattr(result, "application")

    def test_unknown_platform_returns_none(self):
        from capture.window_info import WindowInfoProvider
        p = WindowInfoProvider()
        with patch("capture.window_info._OS", "AmigaOS"):
            assert p.get() is None


# ===========================================================================
# FileManager
# ===========================================================================

class TestFileManager:

    @pytest.fixture()
    def fm(self, tmp_path):
        from storage.file_manager import FileManager
        return FileManager(data_dir=tmp_path, webp_quality=80)

    def test_save_creates_webp(self, fm, tmp_path):
        rel = fm.save_frame(_bgr(), _utc())
        assert fm.absolute_path(rel).exists()
        assert rel.endswith(".webp")

    def test_relative_path_is_portable(self, fm):
        rel = fm.save_frame(_bgr(), _utc())
        assert not rel.startswith("/") and not (len(rel) > 1 and rel[1] == ":")

    def test_rename_to_id_uses_padded_id(self, fm):
        ts = _utc()
        rel = fm.save_frame(_bgr(), ts)
        new_rel = fm.rename_to_id(rel, frame_id=7, timestamp=ts)
        assert "00000007.webp" in new_rel
        assert fm.absolute_path(new_rel).exists()

    def test_delete_returns_true_then_false(self, fm):
        rel = fm.save_frame(_bgr(), _utc())
        assert fm.delete_frame(rel) is True
        assert fm.delete_frame(rel) is False

    def test_disk_usage_positive_after_save(self, fm):
        fm.save_frame(_bgr(), _utc())
        assert fm.disk_usage_bytes() > 0

    def test_prune_removes_old_dirs(self, fm, tmp_path):
        old = tmp_path / "images" / "2020-01-01"
        old.mkdir(parents=True)
        removed = fm.prune_before(datetime(2021, 1, 1))
        assert removed == 1
        assert not old.exists()


# ===========================================================================
# Database round-trip with new fields
# ===========================================================================

class TestDatabaseCaptureFields:

    @pytest.fixture()
    def db(self, tmp_path):
        from storage.db import Database
        return Database(db_path=tmp_path / "test.db")

    def test_insert_and_read_capture_reason(self, db):
        fid = db.insert_frame(
            _utc(), "x.webp", 1920, 1080,
            capture_reason="application_change",
            diff_score=0.003,
        )
        rows = db.get_capture_records(limit=1)
        assert rows[0]["capture_reason"] == "application_change"
        assert rows[0]["diff_score"] == pytest.approx(0.003, abs=1e-6)

    def test_get_frames_by_reason_filters_correctly(self, db):
        ts = _utc()
        db.insert_frame(ts, "a.webp", 100, 100, capture_reason="visual_change")
        db.insert_frame(ts, "b.webp", 100, 100, capture_reason="periodic_capture")
        db.insert_frame(ts, "c.webp", 100, 100, capture_reason="visual_change")
        rows = db.get_frames_by_reason("visual_change")
        assert len(rows) == 2
        assert all(r["capture_reason"] == "visual_change" for r in rows)

    def test_default_capture_reason_is_visual_change(self, db):
        db.insert_frame(_utc(), "x.webp", 100, 100)
        rows = db.get_capture_records(limit=1)
        assert rows[0]["capture_reason"] == "visual_change"
        assert rows[0]["diff_score"] == pytest.approx(0.0)

    def test_application_change_reason_persisted(self, db):
        db.insert_frame(
            _utc(), "x.webp", 100, 100,
            application="chrome",
            capture_reason="application_change",
            diff_score=0.0015,
        )
        rows = db.get_frames_by_reason("application_change")
        assert len(rows) == 1
        assert rows[0]["application"] == "chrome"


# ===========================================================================
# ScreenRecorder lifecycle (all I/O mocked)
# ===========================================================================

class TestScreenRecorderLifecycle:

    @pytest.fixture()
    def recorder(self, tmp_path):
        from capture.screen_recorder import ScreenRecorder
        mock_db = MagicMock()
        mock_db.insert_frame.return_value = 1
        mock_fm = MagicMock()
        mock_fm.save_frame.return_value = "images/2026-01-01/tmp.webp"
        mock_fm.rename_to_id.return_value = "images/2026-01-01/00000001.webp"
        mock_fm.absolute_path.return_value = tmp_path / "dummy.webp"
        return ScreenRecorder(
            db=mock_db,
            file_manager=mock_fm,
            interval_seconds=0.05,
            change_threshold=0.0,    # Accept all visual changes
            max_capture_interval=0.0,
            idle_threshold_seconds=0.0,
        )

    def _mock_mss(self):
        """Context manager that returns a minimal MSS mock."""
        mock_sct = MagicMock()
        mock_sct.__enter__ = lambda s: mock_sct
        mock_sct.__exit__ = MagicMock(return_value=False)
        mock_sct.monitors = [{}, {"width": 100, "height": 100}]
        raw = MagicMock()
        raw.__array__ = lambda *a, **k: np.zeros((100, 100, 4), dtype=np.uint8)
        mock_sct.grab.return_value = raw
        return mock_sct

    def test_initial_state_not_running(self, recorder):
        assert not recorder.is_running
        assert not recorder.is_paused

    def test_start_spawns_thread(self, recorder):
        with patch("mss.mss", return_value=self._mock_mss()):
            recorder.start()
            time.sleep(0.1)
            assert recorder._thread is not None
            assert recorder._thread.is_alive()
            recorder.stop()

    def test_double_start_is_noop(self, recorder):
        with patch("mss.mss", return_value=self._mock_mss()):
            recorder.start()
            first = recorder._thread
            recorder.start()
            assert recorder._thread is first
            recorder.stop()

    def test_pause_resume_flags(self, recorder):
        recorder.pause()
        assert recorder.is_paused
        recorder.resume()
        assert not recorder.is_paused

    def test_stop_joins_thread(self, recorder):
        with patch("mss.mss", return_value=self._mock_mss()):
            recorder.start()
            time.sleep(0.1)
            recorder.stop()
            assert not recorder._thread.is_alive()

    def test_force_capture_sets_event(self, recorder):
        recorder.force_capture()
        assert recorder._force_capture_event.is_set()

    def test_reason_counts_incremented(self, recorder):
        """reason_counts should have at least one entry after running briefly."""
        with patch("mss.mss", return_value=self._mock_mss()):
            recorder.start()
            time.sleep(0.2)
            recorder.stop()
        # At least the first frame (diff=1.0) should have been captured
        assert recorder.frames_captured >= 1
        assert sum(recorder.reason_counts.values()) == recorder.frames_captured
