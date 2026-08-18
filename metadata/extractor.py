"""
metadata/extractor.py

Stage 3 Metadata Extractor — Coordinates LLM calls to translate raw screen OCR
text logs into structured, validated JSON metadata.

Adheres strictly to extraction guidelines:
- Never invents info.
- Uses validation retries on validation failure.
- Maintains strict provenance (source_frame_ids/timestamps).
- Distinguishes observed vs inferred facts.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Optional, Union, Sequence, Any

from metadata.llm_client import LLMClient
from metadata.schemas import (
    StructuredMetadata,
    Meeting,
    FileActivity,
    Appointment,
    Person,
    Organization,
    Project,
    URLReference,
)
from ocr.models import RawTextRecord
from processing.models import MergedTextRecord
from processing.privacy_filter import PrivacyFilter

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """
You are an expert information extraction assistant.
You will receive a series of timestamped raw text sections extracted from a computer screen via OCR.
Each section is labeled with:
  "FRAME ID: <id>"
  "TIMESTAMP: <timestamp>"
  "APPLICATION: <app>"
  "WINDOW TITLE: <title>"

Your task is to identify and extract structured metadata from the text, matching the requested JSON schema.

EXTRACT THE FOLLOWING SCHEMAS:
1. Meetings: title, participants, time, platform, discussion_points, action_items.
2. File Activity: file_name, file_path, document_title, application, start_time, end_time, estimated_duration.
3. Appointments: title, date, time, deadline, reminder.
4. Entities: people (name, email, organization), organizations (name, domain), projects (name, description), URLs (url, title).

CRITICAL RULES:
1. NEVER invent information. Only extract facts explicitly supported by the supplied text.
2. If information or a field is missing, return null or an empty list [].
3. For every extracted item, you MUST populate "source_frame_ids" (list of integers) and "source_timestamps" (list of strings) matching the FRAME ID and TIMESTAMP of the section(s) where the fact was observed.
4. Distinguish observed facts from inferred facts: if a fact is not explicitly stated but can be reliably inferred (e.g. platform Zoom inferred from a zoom.us URL), set "is_inferred": true and provide a brief explanation in "inferred_rationale". Otherwise, keep "is_inferred": false.
5. Do NOT assume a person's identity or convert uncertain information into a definite fact.

Return ONLY a valid JSON object matching the StructuredMetadata schema. Do not output markdown code blocks or any other commentary.
""".strip()

_USER_PROMPT_TEMPLATE = """
Timestamped OCR logs:
---
{formatted_logs}
---

Extract structured metadata. Return ONLY valid JSON.
""".strip()


class MetadataExtractor:
    """
    Extracts structured entities, file activities, meetings, and appointments
    from screen text records. Supports both live LLM calls and intelligent
    local heuristic extraction when offline or when no API key is present.

    Args:
        llm_client:      Optional LLMClient instance (e.g. OpenAI / Groq).
        privacy_filter:  PrivacyFilter applied before sending text externally.
        max_validation_retries: Number of retries if the LLM output fails Pydantic validation.
    """

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        privacy_filter: Optional[PrivacyFilter] = None,
        max_validation_retries: int = 2,
    ) -> None:
        self._client = llm_client
        self._filter = privacy_filter or PrivacyFilter()
        self._max_validation_retries = max_validation_retries

    def extract(
        self,
        raw_text_records: Union[RawTextRecord, MergedTextRecord, list[RawTextRecord], list[MergedTextRecord], str],
        frame_id: Optional[int] = None,  # For backward compatibility
    ) -> Optional[StructuredMetadata]:
        """
        Extract structured metadata from raw text records.

        Args:
            raw_text_records: One or more RawTextRecords/MergedTextRecords or a raw string.
            frame_id:         Optional fallback frame ID.

        Returns:
            StructuredMetadata on success, None on total failure.
        """
        records: list[Union[RawTextRecord, MergedTextRecord]] = []

        if isinstance(raw_text_records, str):
            if not raw_text_records.strip():
                return None
            from datetime import datetime, timezone
            records = [
                RawTextRecord(
                    frame_id=frame_id or 0,
                    timestamp=datetime.now(timezone.utc),
                    image_path="",
                    raw_text=raw_text_records,
                )
            ]
        elif isinstance(raw_text_records, (RawTextRecord, MergedTextRecord)):
            records = [raw_text_records]
        elif isinstance(raw_text_records, list):
            records = raw_text_records
        else:
            logger.error("Invalid input type to MetadataExtractor.extract: %s", type(raw_text_records))
            return None

        if not records:
            return None

        # If LLM client is configured, attempt live LLM extraction
        if self._client is not None:
            return self._extract_with_llm(records)

        # Fallback / Offline / Heuristic Mode: parse actual live OCR text and window titles
        logger.info("[METADATA] Running dynamic heuristic extraction on screen records...")
        return self._heuristic_extract(records)

    def _extract_with_llm(
        self, records: list[Union[RawTextRecord, MergedTextRecord]]
    ) -> Optional[StructuredMetadata]:
        """Runs LLM extraction via prompt."""
        formatted_segments: list[str] = []
        for rec in records:
            fid = getattr(rec, "frame_id", None) or getattr(rec, "id", None) or 0
            ts_str = rec.timestamp.isoformat() if hasattr(rec, "timestamp") else ""
            app = getattr(rec, "application", "") or ""
            title = getattr(rec, "window_title", "") or ""

            if isinstance(rec, MergedTextRecord):
                text_content = rec.merged_text
            else:
                text_content = rec.raw_text

            if not text_content.strip():
                continue

            filtered = self._filter.filter(text_content)
            seg = (
                f"FRAME ID: {fid}\n"
                f"TIMESTAMP: {ts_str}\n"
                f"APPLICATION: {app}\n"
                f"WINDOW TITLE: {title}\n"
                f"TEXT:\n{filtered}"
            )
            formatted_segments.append(seg)

        if not formatted_segments:
            return None

        formatted_logs = "\n\n---\n\n".join(formatted_segments)
        user_prompt = _USER_PROMPT_TEMPLATE.format(formatted_logs=formatted_logs)

        raw_json = ""
        for attempt in range(1, self._max_validation_retries + 2):
            try:
                assert self._client is not None
                raw_json = self._client.complete(_SYSTEM_PROMPT, user_prompt)
                data = json.loads(raw_json)
                metadata = StructuredMetadata.model_validate(data)
                self._ensure_provenance(metadata, records)
                return metadata
            except Exception as exc:  # noqa: BLE001
                logger.warning("Validation attempt %d failed: %s", attempt, exc)
                if attempt > self._max_validation_retries:
                    return None

        return None

    def _heuristic_extract(
        self, records: list[Union[RawTextRecord, MergedTextRecord]]
    ) -> StructuredMetadata:
        """
        Dynamically extracts structured entities from real OCR text & window titles
        using high-precision pattern recognition, strict noise filtering, and semantic heuristics.
        """
        meetings: list[Meeting] = []
        files: list[FileActivity] = []
        appointments: list[Appointment] = []
        people: list[Person] = []
        organizations: list[Organization] = []
        projects: list[Project] = []
        urls: list[URLReference] = []

        seen_files: set[str] = set()
        seen_people: set[str] = set()
        seen_urls: set[str] = set()
        seen_meetings: set[str] = set()
        seen_appts: set[str] = set()
        seen_orgs: set[str] = set()
        seen_projects: set[str] = set()

        # Domains to ignore as client organizations
        _IGNORE_ORGS = {
            "whatsapp", "webwhatsapp", "gmail", "gmad", "google", "brave", "netmirror",
            "cypress", "github", "youtube", "facebook", "instagram", "twitter", "microsoft",
            "apple", "yahoo", "localhost", "python", "pyside", "pyside6", "gitlab", "bitbucket",
            "amazon", "aws", "cloudflare", "openai", "groq", "anthropic", "npm", "pip", "pypi"
        }

        # Ignored window title substrings for meetings and appointments
        _IGNORED_TITLES = [
            "work memory dashboard", "riom", "whatsapp", "brave", "chrome", "firefox",
            "edge", "explorer", "terminal", "powershell", "cmd.exe", "netmirror", "new tab",
            "open", "home", "settings", "task manager", "downloads", "inbox"
        ]

        def _is_noise_file(name: str) -> bool:
            """Filter out OCR gibberish filenames."""
            name = name.strip()
            if len(name) < 4 or len(name) > 64:
                return True
            # Reject filenames with excessive punctuation or starting with weird prefixes
            if name.startswith(("_", "-", ".")) and not name.startswith(".env"):
                return True
            parts = name.rsplit(".", 1)
            if len(parts) != 2:
                return True
            stem, ext = parts[0], parts[1].lower()
            if ext not in {"py", "js", "ts", "tsx", "jsx", "html", "css", "json", "md", "sql", "txt", "csv", "pdf", "docx", "xlsx", "env"}:
                return True
            # Reject OCR artifacts with consecutive weird consonants or random symbols
            if re.search(r"[^a-zA-Z0-9_\-.]", stem):
                return True
            if len(stem) < 2 and name != ".env":
                return True
            return False

        for rec in records:
            fid = getattr(rec, "frame_id", None) or getattr(rec, "id", None) or 0
            ts_str = rec.timestamp.isoformat() if hasattr(rec, "timestamp") else ""
            app = getattr(rec, "application", "") or ""
            title = getattr(rec, "window_title", "") or ""

            # Skip self-inspection of the RIOM dashboard
            if "RIOM" in title and "Dashboard" in title:
                title = ""

            if isinstance(rec, MergedTextRecord):
                text = rec.merged_text
                fids = rec.contributing_frame_ids or [fid]
                tss = rec.contributing_timestamps or [ts_str]
            else:
                text = rec.raw_text
                fids = [fid]
                tss = [ts_str]

            combined_str = f"{title}\n{text}"

            # 1. URL & Web Link Extraction (Support both https?:// and scheme-less domain paths)
            url_candidates: list[tuple[str, str]] = []  # (canonical_url, raw_found)

            # A. Full URL match
            for m in re.finditer(r"https?://[a-zA-Z0-9\-._~:/?#\[\]@!$&'()*+,;=%]+", combined_str):
                raw_u = m.group(0).rstrip(".,;)\"'>")
                if len(raw_u) > 8:
                    url_candidates.append((raw_u, raw_u))

            # B. Scheme-less domain & path match (e.g. meet.google.com/abc-xyz, github.com/user/repo, docs.google.com/document/d/...)
            domain_url_pattern = r"\b((?:[a-zA-Z0-9\-]+\.)+(?:com|org|net|io|ai|app|so|dev|co|edu|gov|in|us|uk|de|ca|au|me|xyz|tech|info)(?:/[^\s,;\)\"\'<>|]*)?)"
            for m in re.finditer(domain_url_pattern, combined_str, re.IGNORECASE):
                raw_u = m.group(1).rstrip(".,;)\"'>")
                # Filter out obvious non-URL strings like filenames (script.py, doc.pdf)
                if not re.search(r"\.(?:py|js|ts|tsx|jsx|css|html|json|md|sql|txt|csv|pdf|docx|xlsx|pptx|png|jpg|jpeg|webp|gif|zip|tar|gz|exe|dll)$", raw_u, re.IGNORECASE):
                    if len(raw_u) >= 6:
                        canonical = f"https://{raw_u}"
                        url_candidates.append((canonical, raw_u))

            # C. Google Meet 3-part codes in window title or text (e.g. Meet - abc-defg-hij)
            meet_code_match = re.search(r"\b([a-z]{3}-[a-z]{4}-[a-z]{3})\b", combined_str, re.IGNORECASE)
            if meet_code_match and ("meet" in combined_str.lower() or "google" in combined_str.lower()):
                m_code = meet_code_match.group(1).lower()
                url_candidates.append((f"https://meet.google.com/{m_code}", f"meet.google.com/{m_code}"))

            # Clean and deduplicate URLs
            for canonical_u, raw_u in url_candidates:
                # Ignore localhost, internal noise, and empty strings
                if any(junk in canonical_u.lower() for junk in ["localhost", "127.0.0.1", "favicon.ico", "riom-dashboard", "wexbw"]):
                    continue
                # Normalize trailing slash
                canonical_clean = canonical_u.rstrip("/")
                if canonical_clean not in seen_urls and len(canonical_clean) > 10:
                    seen_urls.add(canonical_clean)
                    # Deduce meaningful title: document title or window title or domain
                    doc_title = title if title and not any(ig in title.lower() for ig in _IGNORED_TITLES) else None
                    urls.append(
                        URLReference(
                            url=canonical_clean,
                            title=doc_title,
                            source_frame_ids=fids,
                            source_timestamps=tss,
                        )
                    )

            # 2. File Activity Extraction with OCR Noise Filtering & Substring Deduplication
            file_candidates: list[str] = []
            file_pattern = r"\b([a-zA-Z0-9_.\-]+\.(?:py|js|ts|tsx|jsx|html|css|json|md|yaml|yml|sql|txt|csv|pdf|docx|xlsx|pptx|env))\b"
            for m in re.finditer(file_pattern, combined_str, re.IGNORECASE):
                fname = m.group(1).strip()
                if not _is_noise_file(fname):
                    file_candidates.append(fname)

            # Deduplicate suffix fragments (e.g. drop 'en_recorder.py' if 'screen_recorder.py' is present)
            file_candidates = sorted(list(set(file_candidates)), key=len, reverse=True)
            clean_files: list[str] = []
            for fc in file_candidates:
                if any(cf != fc and (cf.endswith(fc) or fc in cf) for cf in clean_files):
                    continue
                clean_files.append(fc)

            for fname in clean_files:
                if fname not in seen_files:
                    seen_files.add(fname)
                    files.append(
                        FileActivity(
                            file_name=fname,
                            document_title=title if title and fname not in title else None,
                            application=app or None,
                            source_frame_ids=fids,
                            source_timestamps=tss,
                        )
                    )

            # 3. Meeting Extraction (Strict matching for real meetings only)
            title_lower = title.lower().strip()
            is_ignored_window = any(ig in title_lower for ig in _IGNORED_TITLES)

            platform_found = None
            meeting_link = None

            # Detect platform
            if "meet.google.com" in combined_str.lower() or "google meet" in combined_str.lower() or re.search(r"\bmeet\s*-\s*[a-z]{3}-[a-z]{4}-[a-z]{3}\b", combined_str, re.IGNORECASE):
                platform_found = "Google Meet"
            elif "zoom.us" in combined_str.lower() or "zoom meeting" in combined_str.lower():
                platform_found = "Zoom"
            elif "teams.microsoft.com" in combined_str.lower() or "microsoft teams" in combined_str.lower():
                platform_found = "Microsoft Teams"
            elif "webex.com" in combined_str.lower():
                platform_found = "Webex"

            # 1. Detect explicit meeting URL
            meet_link_match = re.search(
                r"(https?://(?:meet\.google\.com/[a-zA-Z0-9\-_]+|(?:[a-zA-Z0-9.\-_]+\.)?zoom\.us/(?:j|my)/[0-9a-zA-Z\-_?=&]+|teams\.microsoft\.com/[^\s\)\"\'<]+|teams\.live\.com/meet/[a-zA-Z0-9]+|(?:[a-zA-Z0-9.\-_]+\.)?webex\.com/[^\s\)\"\'<]+))",
                combined_str,
                re.IGNORECASE,
            )
            if meet_link_match:
                meeting_link = meet_link_match.group(1).rstrip(".,;)")
            else:
                meet_url_match = re.search(r"\b(meet\.google\.com/[a-zA-Z0-9\-_]+|zoom\.us/j/[0-9]+)\b", combined_str, re.IGNORECASE)
                if meet_url_match:
                    meeting_link = f"https://{meet_url_match.group(1)}"

            # 2. Detect Google Meet 3-part code (e.g. abc-defg-hij or Meet - abc-defg-hij in browser tab)
            if not meeting_link and (platform_found == "Google Meet" or "meet" in title_lower or "meet" in combined_str.lower()):
                code_match = re.search(r"\b([a-z]{3}-[a-z]{4}-[a-z]{3})\b", combined_str, re.IGNORECASE)
                if code_match:
                    meet_code = code_match.group(1).lower()
                    meeting_link = f"https://meet.google.com/{meet_code}"
                    if not platform_found:
                        platform_found = "Google Meet"

            # Ensure meeting_link has https:// scheme
            if meeting_link and not meeting_link.startswith(("http://", "https://")):
                meeting_link = f"https://{meeting_link}"

            has_meeting_intent = bool(
                re.search(
                    r"\b(?:sprint planning|daily standup|1:1 sync|roadmap review|all hands|team meeting|client sync|interview|design review)\b",
                    combined_str,
                    re.IGNORECASE,
                )
            )

            if (platform_found or has_meeting_intent or meeting_link) and not (is_ignored_window and not platform_found and not meeting_link):
                # Clean up title if it's a browser window title like 'Meet - abc-defg-hij - Brave'
                m_title = title
                if " - " in m_title:
                    parts = [p.strip() for p in m_title.split(" - ") if p.strip().lower() not in {"brave", "google chrome", "chrome", "edge", "firefox"}]
                    if parts:
                        m_title = " — ".join(parts)

                if not m_title or is_ignored_window or m_title.lower() in {"google meet", "meet", "zoom", "teams"}:
                    if meeting_link and "meet.google.com/" in meeting_link:
                        meet_code = meeting_link.split("meet.google.com/")[-1]
                        m_title = f"Google Meet ({meet_code})"
                    else:
                        m_title = f"{platform_found or 'Team'} Meeting"

                if m_title and m_title not in seen_meetings and len(m_title) > 4:
                    seen_meetings.add(m_title)
                    participants: list[str] = []
                    name_matches = re.findall(r"\b([A-Z][a-z]{2,} [A-Z][a-z]{2,})\b", combined_str)
                    for n in name_matches:
                        if n not in participants and n.lower() not in [
                            "visual studio", "google chrome", "sprint planning", "microsoft teams",
                            "project brief", "work memory", "brave browser", "windows terminal"
                        ]:
                            participants.append(n)

                    # Extract emails associated with meeting
                    meeting_emails: list[str] = []
                    for em in re.findall(r"\b([a-zA-Z0-9_.+\-]+@[a-zA-Z0-9\-]+\.[a-zA-Z0-9\-.]+)\b", combined_str):
                        if em.lower() not in [e.lower() for e in meeting_emails] and not em.lower().startswith(("noreply", "no-reply", "support")):
                            meeting_emails.append(em)

                    # Extract discussion points and action items
                    discussion_pts: list[str] = []
                    action_items: list[str] = []
                    in_action_section = False
                    in_disc_section = False

                    for line in combined_str.splitlines():
                        line_s = line.strip()
                        if not line_s:
                            continue
                        if re.search(r"\b(?:action items?|tasks?|to-?do)\b", line_s, re.IGNORECASE):
                            in_action_section = True
                            in_disc_section = False
                            continue
                        elif re.search(r"\b(?:discussion|agenda|notes?|topics?)\b", line_s, re.IGNORECASE):
                            in_disc_section = True
                            in_action_section = False
                            continue

                        if line_s.startswith(("-", "*", "•", "–")) or re.match(r"^\d+[\.\)]\s+", line_s):
                            clean_pt = re.sub(r"^[-*•–\d\.\)]+\s*", "", line_s).strip()
                            if len(clean_pt) > 5:
                                if in_action_section or re.search(r"\b(?:to review|to test|to build|to follow up|assigned to)\b", clean_pt, re.IGNORECASE):
                                    action_items.append(clean_pt)
                                elif in_disc_section or len(discussion_pts) < 4:
                                    discussion_pts.append(clean_pt)

                    meetings.append(
                        Meeting(
                            title=m_title,
                            participants=participants[:5],
                            time=ts_str[11:16] if ts_str else None,
                            platform=platform_found,
                            meeting_link=meeting_link,
                            emails=meeting_emails[:5],
                            discussion_points=discussion_pts[:5],
                            action_items=action_items[:5],
                            source_frame_ids=fids,
                            source_timestamps=tss,
                        )
                    )

            # 4. Appointments & Deadlines (Strict matching — requires explicit event + time)
            time_match = re.search(
                r"\b(\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?|\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b(?:\s+\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?)?|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2}(?:,\s*\d{4})?(?:\s+at\s+\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?)?)\b",
                combined_str,
                re.IGNORECASE,
            )
            has_appt_kw = bool(
                re.search(
                    r"\b(?:deadline|due date|scheduled sync|roadmap sync|calendar invite|quarterly sync|dentist|doctor|interview at|sync at|on friday|on monday|due by)\b",
                    combined_str,
                    re.IGNORECASE,
                )
            )
            if time_match and has_appt_kw and not is_ignored_window:
                # Find the line containing the keyword for a descriptive title
                appt_title = ""
                for line in combined_str.splitlines():
                    if re.search(r"\b(?:deadline|due|sync|schedule|calendar|roadmap|interview|meeting)\b", line, re.IGNORECASE):
                        clean_line = re.sub(r"[^\w\s\-:–—@.]", " ", line).strip()
                        if 4 < len(clean_line) <= 70:
                            appt_title = clean_line
                            break

                if not appt_title:
                    appt_title = f"Scheduled Event ({time_match.group(1)})"

                if appt_title not in seen_appts and not any(ig in appt_title.lower() for ig in _IGNORED_TITLES):
                    seen_appts.add(appt_title)
                    appointments.append(
                        Appointment(
                            title=appt_title,
                            time=time_match.group(1),
                            deadline=time_match.group(1) if "due" in combined_str.lower() or "deadline" in combined_str.lower() else None,
                            source_frame_ids=fids,
                            source_timestamps=tss,
                        )
                    )


            # 5. People Extraction
            email_matches = re.findall(r"\b([a-zA-Z0-9_.+\-]+@[a-zA-Z0-9\-]+\.[a-zA-Z0-9\-.]+)\b", combined_str)
            for em in email_matches:
                prefix = em.split("@")[0].lower()
                # Skip bot/system/numeric usernames
                if prefix in {"noreply", "no-reply", "support", "info", "admin", "service", "system", "mailer-daemon", "git"}:
                    continue
                if re.match(r"^[0-9]+$", prefix) or len(prefix) < 3:
                    continue

                clean_name = re.sub(r"[0-9_\-]+", " ", prefix).strip().title()
                if len(clean_name) >= 3 and clean_name not in seen_people:
                    seen_people.add(clean_name)
                    people.append(
                        Person(
                            name=clean_name,
                            email=em,
                            source_frame_ids=fids,
                            source_timestamps=tss,
                        )
                    )

            # 6. Projects & Organizations
            for kw in ["RIOM", "Antigravity", "AI Work Memory"]:
                if re.search(rf"\b{re.escape(kw)}\b", combined_str, re.IGNORECASE) and kw not in seen_projects:
                    seen_projects.add(kw)
                    projects.append(
                        Project(
                            name=kw,
                            description=f"Project observed in {app or 'workspace'}",
                            source_frame_ids=fids,
                            source_timestamps=tss,
                        )
                    )

            for domain_match in re.findall(r"\b([a-zA-Z0-9\-]+)\.(?:com|org|io|ai|net)\b", combined_str):
                org_name = domain_match.capitalize().strip()
                if (
                    len(org_name) >= 3
                    and org_name.lower() not in _IGNORE_ORGS
                    and org_name not in seen_orgs
                    and not any(ig in org_name.lower() for ig in _IGNORED_TITLES)
                ):
                    seen_orgs.add(org_name)
                    organizations.append(
                        Organization(
                            name=org_name,
                            domain=f"{domain_match}.com",
                            source_frame_ids=fids,
                            source_timestamps=tss,
                        )
                    )

        metadata = StructuredMetadata(
            meetings=meetings,
            files=files,
            appointments=appointments,
            people=people,
            organizations=organizations,
            projects=projects,
            urls=urls,
        )
        self._ensure_provenance(metadata, records)
        return metadata

    def _ensure_provenance(
        self,
        metadata: StructuredMetadata,
        records: list[Union[RawTextRecord, MergedTextRecord]],
    ) -> None:
        """
        Post-process extracted entities to make sure source_frame_ids and
        source_timestamps are populated.
        """
        default_ids: list[int] = []
        default_ts: list[str] = []

        for rec in records:
            if isinstance(rec, MergedTextRecord):
                default_ids.extend(rec.contributing_frame_ids)
                default_ts.extend(rec.contributing_timestamps)
            else:
                fid = getattr(rec, "frame_id", None) or getattr(rec, "id", None) or 0
                default_ids.append(fid)
                default_ts.append(rec.timestamp.isoformat())

        entity_groups = [
            metadata.meetings,
            metadata.files,
            metadata.appointments,
            metadata.people,
            metadata.organizations,
            metadata.projects,
            metadata.urls,
        ]

        for group in entity_groups:
            for entity in group:
                if not getattr(entity, "source_frame_ids", None):
                    entity.source_frame_ids = default_ids
                if not getattr(entity, "source_timestamps", None):
                    entity.source_timestamps = default_ts
