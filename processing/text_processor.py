from __future__ import annotations

import logging
import re
import unicodedata
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import timezone
from typing import Optional, Sequence

from ocr.models import RawTextRecord
from processing.models import MergedTextRecord

logger = logging.getLogger(__name__)


# ===========================================================================
# Configuration
# ===========================================================================

@dataclass
class TextProcessorConfig:
    # Artifact cleaning
    clean_artifacts:       bool  = True

    # UI chrome
    filter_ui_chrome:      bool  = True
    ui_chrome_window:      int   = 10    # Look back this many frames
    ui_chrome_min_repeats: int   = 5     # Line appears in ≥N of window frames

    # Similarity dedup
    deduplicate:           bool  = True
    similarity_threshold:  float = 0.85
    min_chars_to_compare:  int   = 10

    # Merging
    merge_groups:          bool  = True


# ===========================================================================
# ArtifactCleaner
# ===========================================================================

# Compiled artifact patterns.  Each entry is (description, pattern).
# Order matters: more aggressive patterns run first.
_ARTIFACT_PATTERNS: list[tuple[str, re.Pattern]] = [
    # Lines that are only punctuation, symbols, or box-drawing characters
    ("symbol_only",   re.compile(r"^[\s\W_\|•·°·×÷±—–—]+$")),
    # Lines with 3+ repeated identical non-alphanumeric characters
    ("repeated_char", re.compile(r"^(.)\1{3,}$")),
    # Pipe/bar sequences (toolbar separators)
    ("pipe_seq",      re.compile(r"^\|+\s*\|+$")),
    # Lines that are mostly underscores or dashes (horizontal rules mistaken as text)
    ("horiz_rule",    re.compile(r"^[-_=]{4,}$")),
    # Lone whitespace or invisible characters only
    ("whitespace",    re.compile(r"^\s+$")),
    # OCR noise: scattered single letters/numbers with spaces between them
    ("scattered",     re.compile(r"^(?:[A-Za-z0-9]\s){3,}[A-Za-z0-9]?$")),
]

# Lines shorter than this (non-whitespace chars) are considered artefacts
_MIN_CONTENT_CHARS = 2


class ArtifactCleaner:
    """
    Cleans OCR noise while strictly preserving URLs, filenames, dates, times,
    and meaningful domain terms.
    """

    def clean(self, text: str) -> str:
        if not text:
            return text

        lines = text.splitlines()
        cleaned: list[str] = []
        for line in lines:
            stripped = line.strip()
            if self._is_artifact(stripped):
                logger.debug("ArtifactCleaner: dropped %r", stripped[:60])
                continue
            cleaned.append(line)

        return "\n".join(cleaned)

    @staticmethod
    def _is_artifact(line: str) -> bool:
        """Return True if the line is an OCR artefact to be dropped."""
        if not line:
            return False   # Keep empty lines — they are structural

        # Never drop lines containing URLs, emails, filenames, or meet codes
        if any(marker in line.lower() for marker in ["http://", "https://", "@", "meet.", "zoom.", ".py", ".md", ".json", ".ts", ".js"]):
            return False

        # Too few meaningful characters
        content = line.replace(" ", "").replace("\t", "")
        if len(content) < _MIN_CONTENT_CHARS:
            return True

        # Matches any artefact pattern
        for _, pattern in _ARTIFACT_PATTERNS:
            if pattern.match(line):
                return True

        return False


# Alias for spec compliance
TextCleaner = ArtifactCleaner


# ===========================================================================
# UITextFilter
# ===========================================================================

def _normalise_for_ui(line: str) -> str:
    """Lowercase + collapse whitespace for UI chrome comparison."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", line).lower()).strip()


class UITextFilter:
    def __init__(self, window: int = 10, min_repeats: int = 5) -> None:
        self._window      = window
        self._min_repeats = min_repeats
        # Each element is a frozenset of normalised lines from one frame
        self._history: deque[frozenset[str]] = deque(maxlen=window)

    def feed(self, text: str) -> str:
        if not text:
            self._history.append(frozenset())
            return text

        lines = text.splitlines()
        norm_lines = [_normalise_for_ui(l) for l in lines]

        # Count how many past frames contain each normalised line
        line_counts = self._count_line_occurrences(norm_lines)

        # Keep lines that are NOT ui chrome
        kept: list[str] = []
        for orig, norm in zip(lines, norm_lines):
            count = line_counts.get(norm, 0)
            if count >= self._min_repeats:
                logger.debug("UITextFilter: dropped chrome %r (seen in %d frames)", norm[:60], count)
            else:
                kept.append(orig)

        # Update history AFTER filtering decision (using original normalised lines)
        self._history.append(frozenset(n for n in norm_lines if n))

        return "\n".join(kept)

    def reset(self) -> None:
        self._history.clear()

    def _count_line_occurrences(self, norm_lines: list[str]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for norm in norm_lines:
            if not norm:
                continue
            counts[norm] = sum(1 for frame_set in self._history if norm in frame_set)
        return counts

# SimilarityDeduplicator

def _tokenise(text: str) -> frozenset[str]:
    return frozenset(re.findall(r"[a-z0-9]+", text.lower()))


def jaccard_similarity(text_a: str, text_b: str) -> float:
    tokens_a = _tokenise(text_a)
    tokens_b = _tokenise(text_b)
    if not tokens_a and not tokens_b:
        return 1.0   # Both empty → identical
    if not tokens_a or not tokens_b:
        return 0.0   # One empty → completely different
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


@dataclass
class DeduplicationResult:
    primary_index:     int               # Index of the "richest" frame kept
    duplicate_indices: list[int]         # Indices of frames merged into primary
    similarity_scores: dict[str, float]  # "fid_A:fid_B" → Jaccard score


class SimilarityDeduplicator:
    def __init__(self, threshold: float = 0.85, min_chars: int = 10) -> None:
        self._threshold = threshold
        self._min_chars = min_chars

    def deduplicate(
        self, records: list[RawTextRecord]
    ) -> tuple[list[int], list[int], dict[str, float]]:
        if not records:
            return [], [], {}

        n = len(records)
        scores: dict[str, float] = {}

        # Build sorted order: richest text first (for greedy clustering)
        order = sorted(range(n), key=lambda i: len(records[i].raw_text), reverse=True)

        # Cluster assignment: index → cluster_id (= index of primary)
        cluster: dict[int, int] = {}

        for idx in order:
            rec = records[idx]
            text = rec.raw_text

            if len(text) < self._min_chars:
                # Too short — treat as its own cluster (won't merge)
                cluster[idx] = idx
                continue

            # Try to assign to an existing cluster
            assigned = False
            for primary_idx, members in self._iter_clusters(cluster, order):
                for member_idx in members:
                    ref_text = records[member_idx].raw_text
                    if len(ref_text) < self._min_chars:
                        continue
                    sim = jaccard_similarity(text, ref_text)
                    key = f"{rec.frame_id}:{records[member_idx].frame_id}"
                    scores[key] = round(sim, 4)
                    if sim >= self._threshold:
                        cluster[idx] = primary_idx
                        assigned = True
                        break
                if assigned:
                    break

            if not assigned:
                cluster[idx] = idx   # New cluster

        # Collect primaries and duplicates
        primary_set = set(cluster.values())
        primaries   = [i for i in range(n) if i in primary_set]
        duplicates  = [i for i in range(n) if i not in primary_set]

        return primaries, duplicates, scores

    @staticmethod
    def _iter_clusters(
        cluster: dict[int, int],
        order: list[int],
    ):
        from collections import defaultdict
        groups: dict[int, list[int]] = defaultdict(list)
        for idx, primary in cluster.items():
            groups[primary].append(idx)
        for primary in order:
            if primary in groups:
                yield primary, groups[primary]


# ===========================================================================
# FrameGroupMerger
# ===========================================================================

class FrameGroupMerger:
    def __init__(self, deduplicator: SimilarityDeduplicator) -> None:
        self._dedup = deduplicator

    def merge(self, records: list[RawTextRecord]) -> MergedTextRecord:
        if not records:
            raise ValueError("merge() requires at least one record.")

        if len(records) == 1:
            return self._single_record_merge(records[0])

        # Deduplicate
        primaries, duplicates, scores = self._dedup.deduplicate(records)

        # Order all records chronologically for provenance lists
        sorted_records = sorted(records, key=lambda r: r.timestamp)

        # Build merged text: union of unique lines
        merged_text = self._union_merge(records, primaries)

        # Metadata from the most recent record
        latest = max(records, key=lambda r: r.timestamp)
        earliest = min(records, key=lambda r: r.timestamp)

        engines = sorted({r.ocr_engine for r in records if r.ocr_engine})

        return MergedTextRecord(
            contributing_frame_ids   = [r.frame_id for r in sorted_records],
            contributing_timestamps  = [r.timestamp.isoformat() for r in sorted_records],
            contributing_image_paths = [r.image_path for r in sorted_records],
            first_timestamp          = earliest.timestamp,
            last_timestamp           = latest.timestamp,
            application              = latest.application,
            window_title             = latest.window_title,
            merged_text              = merged_text,
            char_count               = len(merged_text),
            is_empty                 = not bool(merged_text.strip()),
            is_deduplicated          = bool(duplicates),
            frame_count              = len(records),
            similarity_scores        = scores,
            ocr_engines              = engines,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _single_record_merge(record: RawTextRecord) -> MergedTextRecord:
        return MergedTextRecord(
            contributing_frame_ids   = [record.frame_id],
            contributing_timestamps  = [record.timestamp.isoformat()],
            contributing_image_paths = [record.image_path],
            first_timestamp          = record.timestamp,
            last_timestamp           = record.timestamp,
            application              = record.application,
            window_title             = record.window_title,
            merged_text              = record.raw_text,
            char_count               = len(record.raw_text),
            is_empty                 = record.is_empty,
            is_deduplicated          = False,
            frame_count              = 1,
            similarity_scores        = {},
            ocr_engines              = [record.ocr_engine] if record.ocr_engine else [],
        )

    @staticmethod
    def _union_merge(records: list[RawTextRecord], primary_indices: list[int]) -> str:
        if not records:
            return ""

        # The primary with the most text is the base
        base_idx = max(primary_indices, key=lambda i: len(records[i].raw_text))
        base_text = records[base_idx].raw_text

        # Normalised set of lines already in the merged text
        def _norm(line: str) -> str:
            return re.sub(r"\s+", " ", line.lower()).strip()

        seen: set[str] = {_norm(l) for l in base_text.splitlines() if l.strip()}
        extra_lines: list[str] = []

        # Iterate remaining records in chronological order
        chron_order = sorted(
            (i for i in range(len(records)) if i != base_idx),
            key=lambda i: records[i].timestamp,
        )
        for idx in chron_order:
            for line in records[idx].raw_text.splitlines():
                norm = _norm(line)
                if norm and norm not in seen:
                    extra_lines.append(line)
                    seen.add(norm)

        if extra_lines:
            return base_text.rstrip() + "\n" + "\n".join(extra_lines)
        return base_text


# Alias for spec compliance
TextStitcher = FrameGroupMerger


# ===========================================================================
# TextProcessor — top-level orchestrator
# ===========================================================================

class TextProcessor:

    def __init__(self, config: TextProcessorConfig | None = None) -> None:
        self._cfg        = config or TextProcessorConfig()
        self._cleaner    = ArtifactCleaner()
        self._ui_filter  = UITextFilter(
            window      = self._cfg.ui_chrome_window,
            min_repeats = self._cfg.ui_chrome_min_repeats,
        )
        self._dedup      = SimilarityDeduplicator(
            threshold  = self._cfg.similarity_threshold,
            min_chars  = self._cfg.min_chars_to_compare,
        )
        self._merger     = FrameGroupMerger(self._dedup)

    def process(self, records: list[RawTextRecord]) -> MergedTextRecord:
        if not records:
            raise ValueError("process() requires at least one RawTextRecord.")

        # Sort by timestamp so UI filter sees records chronologically
        sorted_records = sorted(records, key=lambda r: r.timestamp)

        # Steps 1–2: clean each record and create a working copy
        cleaned_records: list[RawTextRecord] = []
        for rec in sorted_records:
            text = rec.raw_text

            if self._cfg.clean_artifacts:
                text = self._cleaner.clean(text)

            if self._cfg.filter_ui_chrome:
                text = self._ui_filter.feed(text)

            # Clone the record with cleaned text (original raw_text untouched)
            cleaned = rec.model_copy()
            cleaned.raw_text = text
            cleaned.char_count = len(text)
            cleaned.is_empty = not bool(text.strip())
            cleaned_records.append(cleaned)

        logger.debug(
            "TextProcessor: %d records in, cleaning complete.",
            len(cleaned_records),
        )

        # Steps 4–5: deduplicate and merge
        if not self._cfg.merge_groups or len(cleaned_records) == 1:
            merged = self._merger.merge(cleaned_records)
        else:
            merged = self._merger.merge(cleaned_records)

        if merged.is_deduplicated:
            logger.info(
                "TextProcessor: merged %d frames → 1 record (deduped=%s).",
                len(records),
                merged.is_deduplicated,
            )
        return merged

    def reset_ui_filter(self) -> None:
        self._ui_filter.reset()
