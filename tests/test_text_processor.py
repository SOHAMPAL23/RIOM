"""
tests/test_text_processor.py

Unit tests for the Stage 2.5 text processing pipeline.

Coverage
--------
ArtifactCleaner
    - Removes symbol-only lines
    - Removes lines of repeated identical characters
    - Removes pipe sequences (toolbar separators)
    - Removes horizontal rules (dashes/underscores)
    - Removes scattered single-letter noise
    - Keeps meaningful short lines that look like artifacts
    - Preserves structural blank lines

UITextFilter
    - Lines that appear in many frames are filtered out
    - Lines that appear infrequently are kept
    - reset() clears history
    - Works correctly on empty text

jaccard_similarity
    - Identical texts → 1.0
    - Completely different texts → 0.0
    - Partial overlap → correct ratio
    - Both empty → 1.0
    - One empty → 0.0

SimilarityDeduplicator
    - Identical frames → one primary, one duplicate
    - Completely different frames → two primaries
    - Three frames with example from user story (Google Meet)
    - similarity_scores populated for merged pairs
    - Short texts below min_chars are not merged

FrameGroupMerger
    - Single record → wrapped as MergedTextRecord, no dedup
    - Identical records → one record, provenance has all frame IDs
    - Frame 3 adds new content → new lines appended to merged text
    - Provenance always complete (frame_ids, timestamps, image_paths)
    - is_deduplicated=True when duplicates exist

TextProcessor (full pipeline)
    - User story: Google Meet example → 3 frames → 1 merged, new content kept
    - Artifacts removed before similarity comparison
    - UI chrome stripped across a sequence of frames
    - Empty records handled gracefully
    - Single record passes through unchanged (no dedup needed)
    - reset_ui_filter() resets chrome history

MergedTextRecord (Pydantic)
    - JSON round-trip preserves all fields

Database (merged_text_records)
    - insert_merged_text_record / get_merged_text_records
    - search_merged_text LIKE query
    - application filter in get_merged_text_records
"""
from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from typing import Optional

import pytest

from ocr.models import RawTextRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts(offset_seconds: float = 0.0) -> datetime:
    base = datetime(2026, 8, 14, 10, 0, 0, tzinfo=timezone.utc)
    return base + timedelta(seconds=offset_seconds)


def _make_raw(
    frame_id: int,
    text: str,
    application: str = "vscode",
    window_title: str = "main.py",
    ts_offset: float = 0.0,
    ocr_engine: str = "paddleocr",
) -> RawTextRecord:
    """Create a RawTextRecord with the given text and metadata."""
    return RawTextRecord(
        id=frame_id,
        frame_id=frame_id,
        timestamp=_ts(ts_offset),
        image_path=f"images/frame_{frame_id:04d}.webp",
        application=application,
        window_title=window_title,
        raw_text=text,
        confidence=0.9,
        ocr_engine=ocr_engine,
        char_count=len(text),
        is_empty=not bool(text.strip()),
    )


# ===========================================================================
# ArtifactCleaner
# ===========================================================================

class TestArtifactCleaner:

    def setup_method(self):
        from processing.text_processor import ArtifactCleaner
        self.cleaner = ArtifactCleaner()

    def test_removes_symbol_only_lines(self):
        text = "Hello\n|||•|||·|||·\nWorld"
        result = self.cleaner.clean(text)
        assert "|||" not in result
        assert "Hello" in result
        assert "World" in result

    def test_removes_repeated_char_lines(self):
        text = "Title\n----\nContent"
        result = self.cleaner.clean(text)
        assert "----" not in result
        assert "Title" in result

    def test_removes_underline_runs(self):
        text = "Header\n__________\nBody text"
        result = self.cleaner.clean(text)
        assert "__________" not in result

    def test_removes_pipe_sequences(self):
        text = "Menu\n|  |\nItem"
        result = self.cleaner.clean(text)
        assert "|  |" not in result

    def test_removes_scattered_single_letter_noise(self):
        # "l l l l l" is a common OCR artefact from rendering glitches
        text = "Good line\nl l l l l\nAnother good line"
        result = self.cleaner.clean(text)
        assert "l l l l l" not in result

    def test_keeps_meaningful_short_lines(self):
        # "OK", "No", "ID" are real content, not artefacts
        text = "Click OK to continue\nOK\nCancel"
        result = self.cleaner.clean(text)
        assert "OK" in result

    def test_preserves_empty_lines_as_structure(self):
        text = "Paragraph one\n\nParagraph two"
        result = self.cleaner.clean(text)
        assert "\n\n" in result

    def test_keeps_normal_content_untouched(self):
        text = "Google Meet\nRahul Project Alpha\nAPI discussion at 3pm"
        result = self.cleaner.clean(text)
        assert result == text

    def test_empty_input_returns_empty(self):
        assert self.cleaner.clean("") == ""

    def test_does_not_remove_legitimate_dashes(self):
        # "2026-08-14" should not be removed as a "horizontal rule"
        text = "Date: 2026-08-14\nMeeting notes"
        result = self.cleaner.clean(text)
        assert "2026-08-14" in result


# ===========================================================================
# UITextFilter
# ===========================================================================

class TestUITextFilter:

    def setup_method(self):
        from processing.text_processor import UITextFilter
        # window=5, min_repeats=3: a line appearing in 3 of 5 frames is chrome
        self.filt = UITextFilter(window=5, min_repeats=3)

    def _feed_n(self, text: str, n: int) -> str:
        """Feed the same text n times, return last result."""
        result = ""
        for _ in range(n):
            result = self.filt.feed(text)
        return result

    def test_new_line_is_kept_initially(self):
        result = self.filt.feed("Status: Online\nContent line")
        assert "Status: Online" in result

    def test_line_kept_below_threshold(self):
        # Feed 2 times with window=5, min_repeats=3 → should NOT be filtered
        for _ in range(2):
            self.filt.feed("Status Bar\nContent")
        result = self.filt.feed("Status Bar\nNew content")
        assert "Status Bar" in result

    def test_repeated_line_filtered_after_threshold(self):
        for _ in range(4):
            self.filt.feed("File  Edit  View\nReal content")
        result = self.filt.feed("File  Edit  View\nNew real content")
        # "File  Edit  View" seen in 4 of 4 frames → chrome
        assert "File  Edit  View" not in result
        # Unique content is kept
        assert "New real content" in result

    def test_unique_lines_always_kept(self):
        # Feed chrome for 5 frames
        for _ in range(5):
            self.filt.feed("Chrome line\nUnique 1")
        # New frame has different unique content
        result = self.filt.feed("Chrome line\nNew unique content")
        assert "New unique content" in result

    def test_reset_clears_history(self):
        # Build up history
        for _ in range(5):
            self.filt.feed("Repeated line\nContent")
        self.filt.reset()
        # After reset, line should not be filtered
        result = self.filt.feed("Repeated line\nContent")
        assert "Repeated line" in result

    def test_empty_text_handled(self):
        result = self.filt.feed("")
        assert result == ""

    def test_case_insensitive_detection(self):
        # Feed "status bar" in various cases
        for _ in range(4):
            self.filt.feed("STATUS BAR\nContent")
        result = self.filt.feed("status bar\nMore content")
        # Normalised comparison: "STATUS BAR" == "status bar"
        assert "status bar" not in result.lower()


# ===========================================================================
# jaccard_similarity
# ===========================================================================

class TestJaccardSimilarity:

    def setup_method(self):
        from processing.text_processor import jaccard_similarity
        self.jaccard = jaccard_similarity

    def test_identical_text_is_one(self):
        text = "Google Meet Rahul Project Alpha"
        assert self.jaccard(text, text) == pytest.approx(1.0)

    def test_completely_different_text_is_zero(self):
        assert self.jaccard("apple banana cherry", "dog elephant fox") == pytest.approx(0.0)

    def test_partial_overlap(self):
        # A = {hello, world}, B = {hello, python}
        # intersection = {hello} = 1
        # union = {hello, world, python} = 3
        # Jaccard = 1/3
        sim = self.jaccard("hello world", "hello python")
        assert sim == pytest.approx(1 / 3, abs=0.01)

    def test_both_empty_is_one(self):
        assert self.jaccard("", "") == pytest.approx(1.0)

    def test_one_empty_is_zero(self):
        assert self.jaccard("hello world", "") == pytest.approx(0.0)
        assert self.jaccard("", "hello world") == pytest.approx(0.0)

    def test_superset_has_score_less_than_one(self):
        a = "Google Meet Rahul Project Alpha"
        b = "Google Meet Rahul Project Alpha API discussion"
        sim = self.jaccard(a, b)
        assert 0.5 < sim < 1.0

    def test_symmetry(self):
        a = "the quick brown fox"
        b = "the slow brown dog"
        assert self.jaccard(a, b) == pytest.approx(self.jaccard(b, a))


# ===========================================================================
# SimilarityDeduplicator
# ===========================================================================

class TestSimilarityDeduplicator:

    def setup_method(self):
        from processing.text_processor import SimilarityDeduplicator
        self.dedup = SimilarityDeduplicator(threshold=0.85, min_chars=5)

    def test_identical_frames_one_primary_one_duplicate(self):
        text = "Google Meet Rahul Project Alpha"
        r1 = _make_raw(1, text, ts_offset=0)
        r2 = _make_raw(2, text, ts_offset=5)
        primaries, duplicates, scores = self.dedup.deduplicate([r1, r2])
        assert len(primaries) == 1
        assert len(duplicates) == 1

    def test_different_frames_two_primaries(self):
        r1 = _make_raw(1, "Hello World from Python", ts_offset=0)
        r2 = _make_raw(2, "Git commit staged files ready", ts_offset=5)
        primaries, duplicates, scores = self.dedup.deduplicate([r1, r2])
        assert len(primaries) == 2
        assert len(duplicates) == 0

    def test_user_story_google_meet(self):
        """
        Frame 1 = Frame 2 = "Google Meet Rahul Project Alpha"
        Frame 3 = "Google Meet Rahul Project Alpha API discussion"

        Expected: Frames 1 and 2 are duplicates of each other.
        Frame 3 is above threshold vs 1 and 2, so also merged.
        → 1 primary group total.
        """
        from processing.text_processor import SimilarityDeduplicator
        dedup = SimilarityDeduplicator(threshold=0.70, min_chars=5)
        base = "Google Meet Rahul Project Alpha"
        extended = base + " API discussion"
        r1 = _make_raw(1, base,     ts_offset=0)
        r2 = _make_raw(2, base,     ts_offset=5)
        r3 = _make_raw(3, extended, ts_offset=10)
        primaries, duplicates, scores = dedup.deduplicate([r1, r2, r3])
        # All three are similar at threshold=0.70 → 1 primary cluster
        assert len(primaries) == 1
        assert len(duplicates) == 2


    def test_similarity_scores_populated_for_merged_pairs(self):
        text = "Some shared content here"
        r1 = _make_raw(1, text, ts_offset=0)
        r2 = _make_raw(2, text, ts_offset=5)
        _, _, scores = self.dedup.deduplicate([r1, r2])
        assert len(scores) >= 1
        assert all(0.0 <= v <= 1.0 for v in scores.values())

    def test_short_texts_below_min_chars_not_merged(self):
        from processing.text_processor import SimilarityDeduplicator
        dedup = SimilarityDeduplicator(threshold=0.85, min_chars=50)
        r1 = _make_raw(1, "hi", ts_offset=0)
        r2 = _make_raw(2, "hi", ts_offset=5)
        primaries, duplicates, scores = dedup.deduplicate([r1, r2])
        # Both too short to compare → both become primaries
        assert len(primaries) == 2
        assert len(duplicates) == 0

    def test_empty_list_returns_empty(self):
        primaries, duplicates, scores = self.dedup.deduplicate([])
        assert primaries == []
        assert duplicates == []
        assert scores == {}

    def test_single_record_is_its_own_primary(self):
        r = _make_raw(1, "Hello World Python test", ts_offset=0)
        primaries, duplicates, scores = self.dedup.deduplicate([r])
        assert primaries == [0]
        assert duplicates == []


# ===========================================================================
# FrameGroupMerger
# ===========================================================================

class TestFrameGroupMerger:

    def setup_method(self):
        from processing.text_processor import FrameGroupMerger, SimilarityDeduplicator
        self.merger = FrameGroupMerger(SimilarityDeduplicator(threshold=0.85))

    def test_single_record_wrapped_as_merged(self):
        r = _make_raw(1, "Hello World", ts_offset=0)
        result = self.merger.merge([r])
        assert result.merged_text == "Hello World"
        assert result.frame_count == 1
        assert result.is_deduplicated is False
        assert result.contributing_frame_ids == [1]

    def test_identical_records_provenance_contains_all_ids(self):
        text = "Google Meet Rahul Project Alpha"
        r1 = _make_raw(1, text, ts_offset=0)
        r2 = _make_raw(2, text, ts_offset=5)
        r3 = _make_raw(3, text, ts_offset=10)
        result = self.merger.merge([r1, r2, r3])
        # All 3 frame IDs in provenance
        assert set(result.contributing_frame_ids) == {1, 2, 3}
        assert result.frame_count == 3
        assert result.is_deduplicated is True

    def test_frame_3_new_content_appended(self):
        base = "Google Meet Rahul Project Alpha"
        r1 = _make_raw(1, base,                          ts_offset=0)
        r2 = _make_raw(2, base,                          ts_offset=5)
        r3 = _make_raw(3, base + "\nAPI discussion",     ts_offset=10)
        result = self.merger.merge([r1, r2, r3])
        assert "API discussion" in result.merged_text

    def test_provenance_timestamps_are_sorted(self):
        r1 = _make_raw(1, "Content A", ts_offset=10)
        r2 = _make_raw(2, "Content B", ts_offset=0)   # Earlier
        result = self.merger.merge([r1, r2])
        # Timestamps should be in chronological order
        ts_list = result.contributing_timestamps
        assert ts_list == sorted(ts_list)

    def test_provenance_image_paths_preserved(self):
        r1 = _make_raw(1, "Text here", ts_offset=0)
        r2 = _make_raw(2, "Text here", ts_offset=5)
        result = self.merger.merge([r1, r2])
        assert "images/frame_0001.webp" in result.contributing_image_paths
        assert "images/frame_0002.webp" in result.contributing_image_paths

    def test_first_and_last_timestamp(self):
        r1 = _make_raw(1, "A", ts_offset=0)
        r2 = _make_raw(2, "B", ts_offset=100)
        result = self.merger.merge([r1, r2])
        assert result.first_timestamp == _ts(0)
        assert result.last_timestamp  == _ts(100)

    def test_application_from_latest_frame(self):
        r1 = _make_raw(1, "Text", application="chrome",  ts_offset=0)
        r2 = _make_raw(2, "Text", application="vscode",  ts_offset=5)
        result = self.merger.merge([r1, r2])
        assert result.application == "vscode"

    def test_ocr_engines_collected(self):
        r1 = _make_raw(1, "Text A", ocr_engine="paddleocr", ts_offset=0)
        r2 = _make_raw(2, "Text B", ocr_engine="tesseract",  ts_offset=5)
        result = self.merger.merge([r1, r2])
        assert set(result.ocr_engines) == {"paddleocr", "tesseract"}

    def test_empty_records_raise(self):
        with pytest.raises(ValueError):
            self.merger.merge([])

    def test_is_empty_flag(self):
        r = _make_raw(1, "", ts_offset=0)
        r.is_empty = True
        result = self.merger.merge([r])
        assert result.is_empty is True


# ===========================================================================
# TextProcessor — full pipeline
# ===========================================================================

class TestTextProcessor:

    def setup_method(self):
        from processing.text_processor import TextProcessor, TextProcessorConfig
        self.proc = TextProcessor(TextProcessorConfig(
            ui_chrome_window=5,
            ui_chrome_min_repeats=3,
            similarity_threshold=0.85,
            min_chars_to_compare=5,
        ))

    def test_user_story_google_meet(self):
        """
        User story:
            Frame 1: "Google Meet Rahul Project Alpha"
            Frame 2: "Google Meet Rahul Project Alpha"   ← duplicate
            Frame 3: "Google Meet Rahul Project Alpha API discussion"  ← new content

        Expected:
            - Single MergedTextRecord
            - "API discussion" present in merged_text
            - All 3 frame IDs in contributing_frame_ids
            - is_deduplicated=True
        """
        base = "Google Meet Rahul Project Alpha"
        r1 = _make_raw(1, base,              ts_offset=0)
        r2 = _make_raw(2, base,              ts_offset=5)
        r3 = _make_raw(3, base + "\nAPI discussion", ts_offset=10)

        result = self.proc.process([r1, r2, r3])

        assert "API discussion" in result.merged_text
        assert set(result.contributing_frame_ids) == {1, 2, 3}
        assert result.is_deduplicated is True
        assert result.frame_count == 3

    def test_artifacts_removed_before_similarity(self):
        """Artifact lines should not prevent correct similarity detection."""
        clean = "Google Meet Rahul Project Alpha"
        r1 = _make_raw(1, clean + "\n----\n|  |", ts_offset=0)
        r2 = _make_raw(2, clean + "\n----\n|  |", ts_offset=5)
        result = self.proc.process([r1, r2])
        assert "----" not in result.merged_text
        assert "|  |" not in result.merged_text
        assert "Google Meet" in result.merged_text

    def test_single_record_passes_through(self):
        r = _make_raw(1, "Hello World meeting notes", ts_offset=0)
        result = self.proc.process([r])
        assert result.frame_count == 1
        assert result.is_deduplicated is False
        assert "Hello World" in result.merged_text

    def test_provenance_never_lost(self):
        """Even when 3 frames are fully deduplicated, all IDs appear in provenance."""
        text = "Identical screen content for all frames"
        records = [_make_raw(i, text, ts_offset=i*5) for i in range(1, 6)]
        result = self.proc.process(records)
        assert set(result.contributing_frame_ids) == {1, 2, 3, 4, 5}
        assert len(result.contributing_image_paths) == 5
        assert len(result.contributing_timestamps) == 5

    def test_completely_different_frames_not_merged(self):
        """Frames with unrelated content should each contribute to merged text."""
        r1 = _make_raw(1, "Python code editor window with syntax highlighting", ts_offset=0)
        r2 = _make_raw(2, "Browser showing GitHub repository commits list",     ts_offset=5)
        result = self.proc.process([r1, r2])
        # Both texts should appear (they are different content)
        assert "Python code" in result.merged_text
        assert "GitHub repository" in result.merged_text

    def test_ui_chrome_stripped_over_sequence(self):
        """Lines repeated across many frames should be stripped."""
        from processing.text_processor import TextProcessor, TextProcessorConfig
        proc = TextProcessor(TextProcessorConfig(
            ui_chrome_window=5,
            ui_chrome_min_repeats=3,
            similarity_threshold=0.5,  # Low threshold for this test
        ))
        chrome_line = "File  Edit  View  Go  Run  Terminal  Help"
        # Build history with chrome present in 4 frames
        for i in range(1, 5):
            proc.process([_make_raw(i, f"{chrome_line}\nUnique content {i}", ts_offset=i*5)])
        # 5th frame — chrome should now be filtered
        result = proc.process([_make_raw(5, f"{chrome_line}\nFinal unique", ts_offset=25)])
        assert chrome_line not in result.merged_text
        assert "Final unique" in result.merged_text

    def test_empty_records_handled(self):
        r = _make_raw(1, "", ts_offset=0)
        r.is_empty = True
        result = self.proc.process([r])
        assert result.is_empty is True
        assert result.contributing_frame_ids == [1]

    def test_reset_ui_filter_clears_history(self):
        from processing.text_processor import TextProcessor, TextProcessorConfig
        proc = TextProcessor(TextProcessorConfig(
            ui_chrome_window=5,
            ui_chrome_min_repeats=3,
        ))
        chrome = "Repeated chrome line here"
        for i in range(5):
            proc.process([_make_raw(i + 1, f"{chrome}\nContent {i}", ts_offset=i * 5)])
        proc.reset_ui_filter()
        result = proc.process([_make_raw(10, f"{chrome}\nNew content", ts_offset=50)])
        # After reset, chrome line should not be filtered
        assert chrome in result.merged_text


# ===========================================================================
# MergedTextRecord (Pydantic)
# ===========================================================================

class TestMergedTextRecord:

    def test_json_roundtrip(self):
        from processing.models import MergedTextRecord
        rec = MergedTextRecord(
            contributing_frame_ids   = [1, 2, 3],
            contributing_timestamps  = [_ts(0).isoformat(), _ts(5).isoformat(), _ts(10).isoformat()],
            contributing_image_paths = ["a.webp", "b.webp", "c.webp"],
            first_timestamp          = _ts(0),
            last_timestamp           = _ts(10),
            application              = "vscode",
            window_title             = "main.py",
            merged_text              = "Hello World\nAPI discussion",
            char_count               = 27,
            is_empty                 = False,
            is_deduplicated          = True,
            frame_count              = 3,
            similarity_scores        = {"1:2": 0.95, "1:3": 0.87},
            ocr_engines              = ["paddleocr"],
        )
        restored = MergedTextRecord.model_validate_json(rec.model_dump_json())
        assert restored.contributing_frame_ids == [1, 2, 3]
        assert restored.is_deduplicated is True
        assert restored.similarity_scores == {"1:2": 0.95, "1:3": 0.87}

    def test_default_is_empty_true(self):
        from processing.models import MergedTextRecord
        rec = MergedTextRecord(
            first_timestamp=_ts(0),
            last_timestamp=_ts(0),
        )
        assert rec.is_empty is True


# ===========================================================================
# Database — merged_text_records
# ===========================================================================

class TestDatabaseMergedText:

    @pytest.fixture()
    def db(self, tmp_path):
        from storage.db import Database
        return Database(db_path=tmp_path / "test.db")

    def _make_merged(self, frame_ids: list[int]) -> "MergedTextRecord":
        from processing.models import MergedTextRecord
        return MergedTextRecord(
            contributing_frame_ids   = frame_ids,
            contributing_timestamps  = [_ts(i).isoformat() for i in frame_ids],
            contributing_image_paths = [f"img_{i}.webp" for i in frame_ids],
            first_timestamp          = _ts(frame_ids[0]),
            last_timestamp           = _ts(frame_ids[-1]),
            application              = "chrome",
            window_title             = "Google Meet",
            merged_text              = "Google Meet Rahul Project Alpha\nAPI discussion",
            char_count               = 47,
            is_empty                 = False,
            is_deduplicated          = True,
            frame_count              = len(frame_ids),
            similarity_scores        = {"1:2": 0.97},
            ocr_engines              = ["paddleocr"],
        )

    def test_insert_and_retrieve(self, db):
        rec = self._make_merged([1, 2, 3])
        row_id = db.insert_merged_text_record(rec)
        assert isinstance(row_id, int) and row_id >= 1
        rows = db.get_merged_text_records(limit=1)
        assert len(rows) == 1
        assert rows[0]["frame_count"] == 3
        assert rows[0]["is_deduplicated"] == 1
        assert json.loads(rows[0]["contributing_frame_ids"]) == [1, 2, 3]

    def test_search_merged_text(self, db):
        rec = self._make_merged([1, 2])
        db.insert_merged_text_record(rec)
        results = db.search_merged_text("API discussion")
        assert len(results) == 1
        assert "API discussion" in results[0]["merged_text"]

    def test_search_no_match(self, db):
        rec = self._make_merged([1])
        db.insert_merged_text_record(rec)
        results = db.search_merged_text("XYZ_NOTFOUND")
        assert results == []

    def test_filter_by_application(self, db):
        from processing.models import MergedTextRecord
        # Insert one "chrome" record and one "vscode" record
        rec_chrome = self._make_merged([1, 2])
        db.insert_merged_text_record(rec_chrome)

        rec_vscode = MergedTextRecord(
            contributing_frame_ids   = [3],
            contributing_timestamps  = [_ts(30).isoformat()],
            contributing_image_paths = ["img_3.webp"],
            first_timestamp          = _ts(30),
            last_timestamp           = _ts(30),
            application              = "vscode",
            window_title             = "editor.py",
            merged_text              = "def main(): pass",
            char_count               = 16,
            is_empty                 = False,
            is_deduplicated          = False,
            frame_count              = 1,
            similarity_scores        = {},
            ocr_engines              = ["tesseract"],
        )
        db.insert_merged_text_record(rec_vscode)

        chrome_rows = db.get_merged_text_records(application="chrome")
        vscode_rows = db.get_merged_text_records(application="vscode")
        assert len(chrome_rows) == 1
        assert len(vscode_rows) == 1
        assert chrome_rows[0]["application"] == "chrome"
        assert vscode_rows[0]["application"] == "vscode"

    def test_get_returns_newest_first(self, db):
        import time
        for i in range(3):
            rec = self._make_merged([i + 1])
            rec.first_timestamp = _ts(i * 10)
            rec.last_timestamp  = _ts(i * 10)
            db.insert_merged_text_record(rec)
            time.sleep(0.01)

        rows = db.get_merged_text_records(limit=3)
        timestamps = [r["last_timestamp"] for r in rows]
        assert timestamps == sorted(timestamps, reverse=True)
