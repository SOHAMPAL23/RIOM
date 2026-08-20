"""
processing/meeting_notes.py

Raw Speech-to-Text Meeting Transcript Generator.
Captures and saves the actual OCR-extracted speech/subtitles from video calls
(Google Meet, Zoom, Microsoft Teams, Webex) directly to plain .txt files on disk.

No AI-generated filler — only what was actually seen on screen.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

from config.settings import settings

logger = logging.getLogger(__name__)

# Indian Standard Time — UTC+05:30
_IST = timezone(timedelta(hours=5, minutes=30))


def sanitize_filename(name: str) -> str:
    """Convert title/name to a clean, safe filename string."""
    clean = name.replace("—", "-").replace("–", "-")
    clean = re.sub(r'[\\/*?:"<>|#%!$@&^=+;,\'`~]', "", clean)
    clean = re.sub(r"\s+", "_", clean.strip())
    clean = re.sub(r"_+", "_", clean)
    return clean[:64].strip("._-") or "Meeting_Transcript"


# ---------------------------------------------------------------------------
# UI chrome patterns — lines that are never actual spoken content
# ---------------------------------------------------------------------------

_UI_NOISE_PATTERNS: list[re.Pattern] = [
    re.compile(r"^\s*$"),
    re.compile(r"^(https?://|meet\.|zoom\.|teams\.)", re.IGNORECASE),
    re.compile(r"^\W{1,3}$"),
    re.compile(
        r"^(mute|unmute|camera|end call|present now|chat|participants|more options|"
        r"pin|reactions|settings|leave|join now|raise hand|turn off|turn on|"
        r"captions?|subtitles?|closed captions?|live captions?)$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(home|back|forward|refresh|new tab|bookmarks|extensions|"
        r"chrome|brave|firefox|edge|google|windows|explorer|taskbar|"
        r"start|search|cortana)$",
        re.IGNORECASE,
    ),
    re.compile(r"^\d{1,3}$"),
    re.compile(r"^[a-z]{3}-[a-z]{4}-[a-z]{3}$", re.IGNORECASE),  # bare Meet room codes
    re.compile(r"^meet\.google\.com", re.IGNORECASE),
    re.compile(r"^S \(\d+\)", re.IGNORECASE),      # OCR taskbar artifacts
]

# Common English function words — at least one means the line is likely real speech
_SPEECH_SIGNAL = re.compile(
    r"\b(is|are|was|were|will|would|can|could|should|have|has|had|"
    r"the|and|but|so|that|this|for|with|from|about|just|going|"
    r"need|want|think|know|see|like|make|work|here|there|when|"
    r"then|how|what|why|who|our|we|they|you|I|me|my|it|its|"
    r"said|told|asked|showed|let|got|get|put|take|keep|come|"
    r"okay|yeah|alright|actually|basically|probably|definitely|"
    r"right|sure|good|great|cool|also|even|really|very)\b",
    re.IGNORECASE,
)


def _is_ui_noise(line: str) -> bool:
    """Return True if this line is browser/meeting UI chrome, not spoken content."""
    stripped = line.strip()
    if not stripped or len(stripped) <= 2:
        return True
    for pat in _UI_NOISE_PATTERNS:
        if pat.match(stripped):
            return True
    return False


def _looks_like_speech(line: str) -> bool:
    """Heuristic: does this line resemble a natural speech sentence or caption?"""
    stripped = line.strip()
    if len(stripped) < 5:
        return False
    if not _SPEECH_SIGNAL.search(stripped):
        return False
    # Reject lines that are mostly non-letter characters (OCR garbage)
    non_alpha = sum(1 for c in stripped if not (c.isalpha() or c.isspace() or c in ".,!?'-:;"))
    if len(stripped) > 0 and non_alpha / len(stripped) > 0.35:
        return False
    return True


def extract_speech_lines(raw_ocr_text: str) -> list[str]:
    """
    Extract lines that are likely actual spoken content / subtitles from a
    raw OCR dump of a video call screen.  Returns a deduplicated, ordered list.
    """
    seen: set[str] = set()
    results: list[str] = []

    for line in raw_ocr_text.splitlines():
        line = line.strip()
        if _is_ui_noise(line):
            continue
        if not _looks_like_speech(line):
            continue
        if line in seen:
            continue
        # Skip if this is a substring / superset of the immediately previous line (OCR re-reads)
        if results and (line in results[-1] or results[-1] in line):
            if len(line) > len(results[-1]):
                results[-1] = line  # replace shorter with longer
            continue
        seen.add(line)
        results.append(line)

    return results


# ---------------------------------------------------------------------------
# Core saver
# ---------------------------------------------------------------------------

class MeetingTranscriptSaver:
    """
    Saves raw speech-to-text transcripts from video call OCR frames to .txt files.
    No AI generation — only the text actually seen on screen.
    """

    def __init__(self, output_dir: Optional[Path] = None) -> None:
        self._output_dir = output_dir or settings.data_dir / "meeting_notes"
        self._output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def output_dir(self) -> Path:
        return self._output_dir

    def generate_raw_transcript_txt(
        self,
        meeting_title: str,
        raw_frames: list[dict],
        meeting_link: Optional[str] = None,
        platform: Optional[str] = None,
    ) -> str:
        """
        Build a plain-text, timestamped speech transcript from raw OCR frames.

        Output format:
            Meeting: <title>
            Platform: <platform>
            Link: <url>
            Date: YYYY-MM-DD HH:MM UTC
            ────────────────────────────────────────────────────────────
            [HH:MM:SS]  <speech line>
            [HH:MM:SS]  <speech line>
            ...
        """
        now_str = datetime.now(_IST).strftime("%Y-%m-%d %H:%M IST")

        header_lines = [f"Meeting: {meeting_title}"]
        if platform:
            header_lines.append(f"Platform: {platform}")
        if meeting_link:
            header_lines.append(f"Link:     {meeting_link}")
        header_lines.append(f"Date:     {now_str}")
        header_lines.append("─" * 60)
        header_lines.append("")

        transcript_lines: list[str] = []
        seen_lines: set[str] = set()

        for frame in raw_frames:
            raw_text = frame.get("text", "")
            ts_str = frame.get("timestamp", "")
            if not raw_text.strip():
                continue

            time_tag = ""
            if ts_str:
                try:
                    dt = datetime.fromisoformat(ts_str)
                    time_tag = dt.astimezone(_IST).strftime("%H:%M:%S")
                except Exception:
                    time_tag = ts_str[:8] if len(ts_str) >= 8 else ts_str

            for line in extract_speech_lines(raw_text):
                if line in seen_lines:
                    continue
                seen_lines.add(line)
                prefix = f"[{time_tag}]  " if time_tag else "  "
                transcript_lines.append(f"{prefix}{line}")

        if not transcript_lines:
            transcript_lines.append("(No speech or subtitle text captured during this session)")

        return "\n".join(header_lines + transcript_lines) + "\n"

    def save_transcript_txt(
        self,
        meeting_title: str,
        raw_frames: list[dict],
        meeting_link: Optional[str] = None,
        platform: Optional[str] = None,
        output_dir: Optional[Path] = None,
        source_timestamps: Optional[list[str]] = None,
    ) -> Path:
        """Save the raw speech-to-text transcript as a .txt file on disk."""
        target_dir = output_dir or self._output_dir
        target_dir.mkdir(parents=True, exist_ok=True)

        date_prefix = datetime.now(_IST).strftime("%Y-%m-%d")
        if source_timestamps and len(source_timestamps[0]) >= 10:
            date_prefix = source_timestamps[0][:10]

        safe_title = sanitize_filename(meeting_title)
        filename = f"{date_prefix}_{safe_title}.txt"
        file_path = target_dir / filename

        txt_content = self.generate_raw_transcript_txt(
            meeting_title=meeting_title,
            raw_frames=raw_frames,
            meeting_link=meeting_link,
            platform=platform,
        )
        file_path.write_text(txt_content, encoding="utf-8")
        logger.info("[MEETING_TRANSCRIPT] Saved raw speech-to-text transcript: %s", file_path)
        return file_path

    def list_saved_transcripts(self) -> list[dict]:
        """Return all saved .txt and .md files, newest first."""
        if not self._output_dir.exists():
            return []

        files = []
        for p in sorted(
            list(self._output_dir.glob("*.txt")) + list(self._output_dir.glob("*.md")),
            key=lambda f: f.stat().st_mtime,
            reverse=True,
        ):
            files.append({
                "filename": p.name,
                "path": str(p.resolve()),
                "modified": datetime.fromtimestamp(p.stat().st_mtime, _IST).isoformat(),
                "size_bytes": p.stat().st_size,
                "format": p.suffix,
            })
        return files


# ---------------------------------------------------------------------------
# Backward-compatible MeetingNotesGenerator wrapper
# ---------------------------------------------------------------------------

class MeetingNotesGenerator:
    """
    Backward-compatible wrapper around MeetingTranscriptSaver.
    Now generates raw .txt speech transcripts instead of AI-written markdown.
    """

    def __init__(self, output_dir: Optional[Path] = None) -> None:
        self._output_dir = output_dir or settings.data_dir / "meeting_notes"
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._saver = MeetingTranscriptSaver(output_dir=self._output_dir)

    @property
    def output_dir(self) -> Path:
        return self._output_dir

    def generate_markdown(self, meeting, raw_context: str = "") -> str:  # type: ignore[override]
        """Returns a plain-text transcript (not AI-generated markdown)."""
        raw_frames = []
        if raw_context.strip():
            ts = (getattr(meeting, "source_timestamps", None) or [""])[0]
            raw_frames.append({"timestamp": ts, "text": raw_context, "window_title": meeting.title})
        return self._saver.generate_raw_transcript_txt(
            meeting_title=meeting.title,
            raw_frames=raw_frames,
            meeting_link=getattr(meeting, "meeting_link", None),
            platform=getattr(meeting, "platform", None),
        )

    def save_meeting_notes(
        self,
        meeting,
        raw_context: str = "",
        output_dir: Optional[Path] = None,
    ) -> Path:
        """Save a raw .txt transcript for this meeting."""
        raw_frames = []
        if raw_context.strip():
            ts = (getattr(meeting, "source_timestamps", None) or [""])[0]
            raw_frames.append({"timestamp": ts, "text": raw_context, "window_title": meeting.title})
        return self._saver.save_transcript_txt(
            meeting_title=meeting.title,
            raw_frames=raw_frames,
            meeting_link=getattr(meeting, "meeting_link", None),
            platform=getattr(meeting, "platform", None),
            output_dir=output_dir,
            source_timestamps=getattr(meeting, "source_timestamps", None),
        )

    def process_and_save_all(
        self,
        metadata,
        raw_context_map: Optional[dict[int, str]] = None,
    ) -> list[Path]:
        """Save raw .txt transcript files for all meetings in a StructuredMetadata object."""
        saved_paths: list[Path] = []
        raw_map = raw_context_map or {}

        for m in metadata.meetings:
            try:
                tss = getattr(m, "source_timestamps", None) or []
                fids = getattr(m, "source_frame_ids", None) or []

                raw_frames: list[dict] = []
                for i, fid in enumerate(fids):
                    raw_text = raw_map.get(fid, "")
                    ts = tss[i] if i < len(tss) else ""
                    if raw_text.strip():
                        raw_frames.append({"timestamp": ts, "text": raw_text, "window_title": m.title})

                p = self._saver.save_transcript_txt(
                    meeting_title=m.title,
                    raw_frames=raw_frames,
                    meeting_link=getattr(m, "meeting_link", None),
                    platform=getattr(m, "platform", None),
                    source_timestamps=tss,
                )
                saved_paths.append(p)
            except Exception as exc:  # noqa: BLE001
                logger.warning("[MEETING_TRANSCRIPT] Failed to save transcript for '%s': %s", m.title, exc)

        return saved_paths

    def list_saved_notes(self) -> list[dict]:
        """Return list of all saved transcript files."""
        return self._saver.list_saved_transcripts()
