"""
processing/meeting_notes.py

Automated Ambient Meeting Notes Generator.
Generates structured, professional markdown meeting notes in the background
whenever a meeting (Google Meet, Zoom, Microsoft Teams, Webex) is detected on screen,
and saves them directly to disk as Markdown (.md) files.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

from config.settings import settings
from metadata.schemas import Meeting, StructuredMetadata

logger = logging.getLogger(__name__)


def sanitize_filename(name: str) -> str:
    """Convert title/name to a clean, safe filename string."""
    clean = name.replace("—", "-").replace("–", "-")
    clean = re.sub(r'[\\/*?:"<>|#%!$@&^=+;,\'`~]', "", clean)
    clean = re.sub(r"\s+", "_", clean.strip())
    clean = re.sub(r"_+", "_", clean)
    return clean[:64].strip("._-") or "Meeting_Notes"


class MeetingNotesGenerator:
    """
    Formats and persists structured meeting notes from verified Meeting metadata.
    """

    def __init__(self, output_dir: Optional[Path] = None) -> None:
        self._output_dir = output_dir or settings.data_dir / "meeting_notes"
        self._output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def output_dir(self) -> Path:
        return self._output_dir

    def generate_markdown(self, meeting: Meeting, raw_context: str = "") -> str:
        """
        Generate a comprehensive, executive-ready Markdown document from a Meeting entity.
        """
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        time_display = meeting.time or meeting.start_time or "Observed during working session"
        if meeting.end_time and meeting.end_time != time_display:
            time_display = f"{time_display} – {meeting.end_time}"

        lines: list[str] = [
            f"# 🎙️ Meeting Notes: {meeting.title}",
            "",
            f"> **Generated ambiently by RIOM Screen Understanding** • *{now_str}*",
            "",
            "## 📌 Meeting Overview",
            f"- **Platform**: {meeting.platform or 'Online Video Conference'}",
            f"- **Time / Duration**: {time_display}",
        ]

        if meeting.meeting_link:
            lines.append(f"- **Meeting Link**: [{meeting.meeting_link}]({meeting.meeting_link})")

        # Participants
        lines.append("")
        lines.append("## 👥 Participants & Attendees")
        if meeting.participants:
            for p in meeting.participants:
                lines.append(f"- **{p}**")
        else:
            lines.append("- *No named participants detected from video tiles or chat*")

        if meeting.emails:
            lines.append("")
            lines.append("### 📧 Associated Email Addresses")
            for em in meeting.emails:
                lines.append(f"- `{em}`")

        # Discussion Points & Topics
        lines.append("")
        lines.append("## 💬 Key Discussion Points & Topics")
        if meeting.discussion_points:
            for pt in meeting.discussion_points:
                lines.append(f"- {pt}")
        else:
            lines.append("- General team sync and collaboration observed.")

        # Action Items
        lines.append("")
        lines.append("## ✅ Action Items & Next Steps")
        if meeting.action_items:
            for item in meeting.action_items:
                lines.append(f"- [ ] {item}")
        else:
            lines.append("- [ ] Review notes and follow up with participants if needed")

        # Grounded Evidence & Provenance
        lines.append("")
        lines.append("## 🔍 Truth Grounding & Provenance")
        if meeting.source_frame_ids:
            fids_str = ", ".join(f"#{fid}" for fid in meeting.source_frame_ids)
            lines.append(f"- **Source Frame IDs**: {fids_str}")
        if meeting.source_timestamps:
            tss_str = ", ".join(meeting.source_timestamps[:3])
            lines.append(f"- **Observed Timestamps**: {tss_str}")
        if meeting.is_inferred:
            lines.append(f"- **Inference Note**: {meeting.inferred_rationale or 'Inferred from window title / URL context'}")

        if raw_context.strip():
            lines.append("")
            lines.append("### 📝 Ambient Screen Context Snippet")
            lines.append("```text")
            # Limit snippet length
            snippet = "\n".join(raw_context.strip().splitlines()[:15])
            lines.append(snippet)
            lines.append("```")

        lines.append("")
        lines.append("---")
        lines.append("*Meeting notes automatically recorded and grounded by RIOM Ambient Screen Intelligence.*")
        lines.append("")

        return "\n".join(lines)

    def save_meeting_notes(
        self,
        meeting: Meeting,
        raw_context: str = "",
        output_dir: Optional[Path] = None,
    ) -> Path:
        """
        Save meeting notes as a Markdown file on disk.

        Returns:
            Path to the created/updated Markdown file.
        """
        target_dir = output_dir or self._output_dir
        target_dir.mkdir(parents=True, exist_ok=True)

        date_prefix = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if meeting.source_timestamps and len(meeting.source_timestamps[0]) >= 10:
            date_prefix = meeting.source_timestamps[0][:10]

        safe_title = sanitize_filename(meeting.title)
        filename = f"{date_prefix}_{safe_title}.md"
        file_path = target_dir / filename

        md_content = self.generate_markdown(meeting, raw_context)
        file_path.write_text(md_content, encoding="utf-8")
        logger.info("[MEETING_NOTES] Saved ambient meeting notes to: %s", file_path)
        return file_path

    def process_and_save_all(
        self,
        metadata: StructuredMetadata,
        raw_context_map: Optional[dict[int, str]] = None,
    ) -> list[Path]:
        """
        Save meeting notes files for all meetings inside a StructuredMetadata object.
        """
        saved_paths: list[Path] = []
        raw_map = raw_context_map or {}

        for m in metadata.meetings:
            try:
                # Combine raw text from source frames
                ctx_snippets = [raw_map.get(fid, "") for fid in m.source_frame_ids if fid in raw_map]
                raw_ctx = "\n".join(s for s in ctx_snippets if s)
                p = self.save_meeting_notes(m, raw_context=raw_ctx)
                saved_paths.append(p)
            except Exception as exc:  # noqa: BLE001
                logger.warning("[MEETING_NOTES] Failed to save notes for meeting '%s': %s", m.title, exc)

        return saved_paths

    def list_saved_notes(self) -> list[dict]:
        """
        Return list of all existing saved markdown meeting notes files.
        """
        if not self._output_dir.exists():
            return []

        notes = []
        for p in sorted(self._output_dir.glob("*.md"), key=lambda f: f.stat().st_mtime, reverse=True):
            notes.append({
                "filename": p.name,
                "path": str(p.resolve()),
                "modified": datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).isoformat(),
                "size_bytes": p.stat().st_size,
            })
        return notes
