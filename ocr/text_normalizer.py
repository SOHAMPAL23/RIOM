"""
ocr/text_normalizer.py

Normalises raw OCR output text before it is stored and passed to the
LLM stage.

Why normalise?
--------------
OCR engines often produce:
- Multiple consecutive blank lines from whitespace-heavy UIs.
- Trailing spaces on every line.
- Lines that are only punctuation or single characters (artefacts).
- Mixed line-endings (\\r\\n on Windows screenshots).
- Duplicate adjacent lines (e.g. status bars repeated at top and bottom).

None of these are useful to the LLM and inflate token counts.

Pipeline (applied in order)
----------------------------
1. Strip leading/trailing whitespace from each line.
2. Normalise line endings to \\n.
3. Collapse runs of 3+ blank lines to a single blank line.
4. Drop lines that are only whitespace, punctuation, or a single char.
   (Configurable: min_line_length threshold.)
5. Deduplicate consecutive identical lines (common with repeated UI elements).
6. Strip leading/trailing blank lines from the whole text.

The normalizer is stateless and pure — the same input always produces
the same output.  All steps are individually toggleable for testing.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field


@dataclass
class NormalizerConfig:
    """Controls which normalisation steps are applied."""
    strip_lines:          bool = True   # Strip per-line whitespace
    collapse_blank_lines: bool = True   # Collapse 3+ blanks → 1 blank
    drop_short_lines:     bool = True   # Drop lines shorter than min_line_length
    min_line_length:      int  = 2      # Minimum non-whitespace chars to keep
    deduplicate_lines:    bool = True   # Remove consecutive duplicate lines
    normalise_unicode:    bool = True   # NFC unicode normalisation


class TextNormalizer:
    """
    Cleans and normalises raw OCR text.

    Args:
        config: NormalizerConfig controlling which steps run.
    """

    def __init__(self, config: NormalizerConfig | None = None) -> None:
        self._cfg = config or NormalizerConfig()

    def normalize(self, raw: str) -> str:
        """
        Apply the normalisation pipeline to raw OCR text.

        Args:
            raw: Raw concatenated OCR text (may contain \\r\\n).

        Returns:
            Cleaned text string.
        """
        if not raw:
            return ""

        # Step 0 — Unicode normalisation
        if self._cfg.normalise_unicode:
            raw = unicodedata.normalize("NFC", raw)

        # Step 1 — normalise line endings
        text = raw.replace("\r\n", "\n").replace("\r", "\n")

        lines = text.split("\n")

        # Step 2 — strip per-line whitespace
        if self._cfg.strip_lines:
            lines = [l.strip() for l in lines]

        # Step 3 — drop lines that are too short (noise/artefacts)
        if self._cfg.drop_short_lines:
            lines = [
                l for l in lines
                if len(l.replace(" ", "").replace("\t", "")) >= self._cfg.min_line_length
                or l == ""
            ]

        # Step 4 — collapse runs of 3+ blank lines to a single blank
        if self._cfg.collapse_blank_lines:
            lines = self._collapse_blanks(lines)

        # Step 5 — remove consecutive duplicate lines
        if self._cfg.deduplicate_lines:
            lines = self._dedup_consecutive(lines)

        # Step 6 — strip leading/trailing blank lines
        text = "\n".join(lines).strip()

        return text

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _collapse_blanks(lines: list[str]) -> list[str]:
        """Replace runs of ≥3 consecutive blank lines with a single blank."""
        result: list[str] = []
        blank_run = 0
        for line in lines:
            if line == "":
                blank_run += 1
                if blank_run <= 2:
                    result.append(line)
            else:
                blank_run = 0
                result.append(line)
        return result

    @staticmethod
    def _dedup_consecutive(lines: list[str]) -> list[str]:
        """Remove consecutive duplicate lines (case-sensitive)."""
        result: list[str] = []
        prev = object()  # Sentinel that never equals a string
        for line in lines:
            if line != prev:
                result.append(line)
            prev = line
        return result
