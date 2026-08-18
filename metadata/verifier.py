from __future__ import annotations

import re
import unicodedata
from typing import Any, Optional, Union

from metadata.schemas import (
    StructuredMetadata,
    FactEvidence,
    VerificationStatus,
    Meeting,
    FileActivity,
    Appointment,
    Person,
    Organization,
    Project,
    URLReference,
)


def normalize_text(text: str) -> str:
    """Lowercase and collapse all whitespace/punctuation for comparison."""
    if not text:
        return ""
    # NFC normalisation
    text = unicodedata.normalize("NFC", text).lower()
    # Replace non-alphanumeric characters with spaces
    text = re.sub(r"[^\w\s]", " ", text)
    # Collapse multiple spaces
    return " ".join(text.split())


def is_supported(raw_text: str, query: str) -> bool:
    """
    Deterministic check: is the query phrase present in the raw text?
    Uses substring matching after normalization.
    """
    norm_query = normalize_text(query)
    if not norm_query:
        return True
    norm_raw = normalize_text(raw_text)
    return norm_query in norm_raw


def is_supported_url(raw_text: str, url: str) -> bool:
    """
    Checks if a URL or its primary path/host components are grounded in raw screen text.
    Handles differences in scheme (http vs https vs none), subdomains, and trailing slashes.
    """
    if not url:
        return True
    
    # 1. Exact or normalized full URL match
    if is_supported(raw_text, url):
        return True
    
    # 2. Scheme-stripped & www-stripped match (e.g. 'meet.google.com/abc-xyz' or 'docs.google.com/document/d/...')
    clean_url = re.sub(r"^https?://", "", url.strip(), flags=re.IGNORECASE).rstrip("/")
    if is_supported(raw_text, clean_url):
        return True
    
    clean_no_www = re.sub(r"^www\.", "", clean_url, flags=re.IGNORECASE)
    if is_supported(raw_text, clean_no_www):
        return True

    # 3. Path / Code match (e.g. 'abc-defg-hij' or 'riom-sprint-plan' or 'riom-brief' or repo name)
    segments = [seg for seg in clean_no_www.split("/") if len(seg) >= 6]
    for seg in segments:
        if is_supported(raw_text, seg):
            return True

    # 4. Host match if domain is specific
    host = clean_no_www.split("/")[0]
    if len(host) >= 8 and is_supported(raw_text, host):
        return True

    return False


class MetadataVerifier:
    """
    Verifies StructuredMetadata against the original raw OCR text.
    """

    def verify(
        self,
        metadata: StructuredMetadata,
        raw_text_map: dict[int, str],
        timestamps_map: Optional[dict[int, str]] = None,
    ) -> tuple[StructuredMetadata, list[FactEvidence]]:
        """
        Verify metadata against raw screen text logs.

        Args:
            metadata:       The StructuredMetadata returned by the LLM.
            raw_text_map:   Dict mapping frame_id → raw_text.
            timestamps_map: Optional dict mapping frame_id → ISO timestamp.

        Returns:
            verified_metadata: A clean StructuredMetadata (unsupported items removed,
                               partially supported items cleaned).
            evidences:         List of FactEvidence detail records.
        """
        verified_meetings: list[Meeting] = []
        verified_files: list[FileActivity] = []
        verified_appointments: list[Appointment] = []
        verified_people: list[Person] = []
        verified_organizations: list[Organization] = []
        verified_projects: list[Project] = []
        verified_urls: list[URLReference] = []
        evidences: list[FactEvidence] = []

        # Helper to get consolidated raw text for an entity's source frames
        def _get_raw_text(source_frame_ids: list[int]) -> str:
            texts = [raw_text_map.get(fid, "") for fid in source_frame_ids if fid in raw_text_map]
            return "\n".join(texts)

        # ── 1. Verify Meetings ──────────────────────────────────────────────
        for idx, m in enumerate(metadata.meetings):
            fact_id = f"meeting_{idx}"
            raw_text = _get_raw_text(m.source_frame_ids)
            
            # Check fields
            unsupported: list[str] = []
            
            title_ok = is_supported(raw_text, m.title)
            link_ok = bool(m.meeting_link and is_supported_url(raw_text, m.meeting_link))
            platform_ok = bool(m.platform and is_supported(raw_text, m.platform))

            if not title_ok and not link_ok:
                unsupported.append("title")

            if m.meeting_link and not link_ok:
                unsupported.append("meeting_link")
            
            for p in m.participants:
                if not is_supported(raw_text, p):
                    unsupported.append(f"participant:{p}")
            
            if m.time and not is_supported(raw_text, m.time):
                unsupported.append("time")
            
            if m.platform and not platform_ok:
                unsupported.append("platform")
            
            for dp in m.discussion_points:
                if not is_supported(raw_text, dp):
                    unsupported.append(f"discussion_point:{dp}")
            
            for ai in m.action_items:
                if not is_supported(raw_text, ai):
                    unsupported.append(f"action_item:{ai}")

            # Assign status
            core_supported = ("title" not in unsupported) or link_ok
            total_f = 1 + bool(m.meeting_link) + len(m.participants) + bool(m.time) + bool(m.platform) + len(m.discussion_points) + len(m.action_items)
            status = self._calc_status(
                core_supported=core_supported,
                total_fields=total_f,
                unsupported_count=len(unsupported)
            )

            evidence = self._make_evidence(fact_id, "meeting", m.model_dump(), m.source_frame_ids, timestamps_map, raw_text, status, unsupported)
            evidences.append(evidence)

            if status == VerificationStatus.VERIFIED:
                verified_meetings.append(m)
            elif status == VerificationStatus.PARTIALLY_SUPPORTED:
                # Clean unsupported fields
                cleaned_m = m.model_copy()
                cleaned_m.participants = [p for p in m.participants if f"participant:{p}" not in unsupported]
                cleaned_m.discussion_points = [dp for dp in m.discussion_points if f"discussion_point:{dp}" not in unsupported]
                cleaned_m.action_items = [ai for ai in m.action_items if f"action_item:{ai}" not in unsupported]
                if "time" in unsupported:
                    cleaned_m.time = None
                if "platform" in unsupported:
                    cleaned_m.platform = None
                if "meeting_link" in unsupported:
                    cleaned_m.meeting_link = None
                verified_meetings.append(cleaned_m)

        # ── 2. Verify FileActivity ──────────────────────────────────────────
        for idx, f in enumerate(metadata.files):
            fact_id = f"file_{idx}"
            raw_text = _get_raw_text(f.source_frame_ids)
            
            unsupported = []
            # Core file identification is file_name, document_title or path
            core_ok = False
            if f.file_name:
                if is_supported(raw_text, f.file_name):
                    core_ok = True
                else:
                    unsupported.append("file_name")
            if f.document_title:
                if is_supported(raw_text, f.document_title):
                    core_ok = True
                else:
                    unsupported.append("document_title")
            if f.file_path:
                if is_supported(raw_text, f.file_path):
                    core_ok = True
                else:
                    unsupported.append("file_path")

            if not f.file_name and not f.document_title and not f.file_path:
                unsupported.append("no_identifier")

            if f.application and not is_supported(raw_text, f.application):
                unsupported.append("application")
            
            if f.estimated_duration and not is_supported(raw_text, f.estimated_duration):
                unsupported.append("estimated_duration")

            total_fields = bool(f.file_name) + bool(f.document_title) + bool(f.file_path) + bool(f.application) + bool(f.estimated_duration)
            status = self._calc_status(core_ok, total_fields, len(unsupported))

            evidence = self._make_evidence(fact_id, "file_activity", f.model_dump(), f.source_frame_ids, timestamps_map, raw_text, status, unsupported)
            evidences.append(evidence)

            if status == VerificationStatus.VERIFIED:
                verified_files.append(f)
            elif status == VerificationStatus.PARTIALLY_SUPPORTED:
                cleaned_f = f.model_copy()
                if "file_name" in unsupported:
                    cleaned_f.file_name = None
                if "document_title" in unsupported:
                    cleaned_f.document_title = None
                if "file_path" in unsupported:
                    cleaned_f.file_path = None
                if "application" in unsupported:
                    cleaned_f.application = None
                if "estimated_duration" in unsupported:
                    cleaned_f.estimated_duration = None
                verified_files.append(cleaned_f)

        # ── 3. Verify Appointments ──────────────────────────────────────────
        for idx, appt in enumerate(metadata.appointments):
            fact_id = f"appointment_{idx}"
            raw_text = _get_raw_text(appt.source_frame_ids)
            
            unsupported = []
            if not is_supported(raw_text, appt.title):
                unsupported.append("title")
            
            if appt.date and not is_supported(raw_text, appt.date):
                unsupported.append("date")
            if appt.time and not is_supported(raw_text, appt.time):
                unsupported.append("time")
            if appt.deadline and not is_supported(raw_text, appt.deadline):
                unsupported.append("deadline")
            if appt.reminder and not is_supported(raw_text, appt.reminder):
                unsupported.append("reminder")

            total_fields = 1 + bool(appt.date) + bool(appt.time) + bool(appt.deadline) + bool(appt.reminder)
            status = self._calc_status("title" not in unsupported, total_fields, len(unsupported))

            evidence = self._make_evidence(fact_id, "appointment", appt.model_dump(), appt.source_frame_ids, timestamps_map, raw_text, status, unsupported)
            evidences.append(evidence)

            if status == VerificationStatus.VERIFIED:
                verified_appointments.append(appt)
            elif status == VerificationStatus.PARTIALLY_SUPPORTED:
                cleaned_appt = appt.model_copy()
                if "date" in unsupported:
                    cleaned_appt.date = None
                if "time" in unsupported:
                    cleaned_appt.time = None
                if "deadline" in unsupported:
                    cleaned_appt.deadline = None
                if "reminder" in unsupported:
                    cleaned_appt.reminder = None
                verified_appointments.append(cleaned_appt)

        # ── 4. Verify People ────────────────────────────────────────────────
        for idx, p in enumerate(metadata.people):
            fact_id = f"person_{idx}"
            raw_text = _get_raw_text(p.source_frame_ids)
            
            unsupported = []
            if not is_supported(raw_text, p.name):
                unsupported.append("name")
            
            if p.email and not is_supported(raw_text, p.email):
                unsupported.append("email")
            
            if p.organization and not is_supported(raw_text, p.organization):
                unsupported.append("organization")

            total_fields = 1 + bool(p.email) + bool(p.organization)
            status = self._calc_status("name" not in unsupported, total_fields, len(unsupported))

            evidence = self._make_evidence(fact_id, "person", p.model_dump(), p.source_frame_ids, timestamps_map, raw_text, status, unsupported)
            evidences.append(evidence)

            if status == VerificationStatus.VERIFIED:
                verified_people.append(p)
            elif status == VerificationStatus.PARTIALLY_SUPPORTED:
                cleaned_p = p.model_copy()
                if "email" in unsupported:
                    cleaned_p.email = None
                if "organization" in unsupported:
                    cleaned_p.organization = None
                verified_people.append(cleaned_p)

        # ── 5. Verify Organizations ─────────────────────────────────────────
        for idx, org in enumerate(metadata.organizations):
            fact_id = f"organization_{idx}"
            raw_text = _get_raw_text(org.source_frame_ids)
            
            unsupported = []
            if not is_supported(raw_text, org.name):
                unsupported.append("name")
            if org.domain and not is_supported(raw_text, org.domain):
                unsupported.append("domain")

            total_fields = 1 + bool(org.domain)
            status = self._calc_status("name" not in unsupported, total_fields, len(unsupported))

            evidence = self._make_evidence(fact_id, "organization", org.model_dump(), org.source_frame_ids, timestamps_map, raw_text, status, unsupported)
            evidences.append(evidence)

            if status == VerificationStatus.VERIFIED:
                verified_organizations.append(org)
            elif status == VerificationStatus.PARTIALLY_SUPPORTED:
                cleaned_org = org.model_copy()
                if "domain" in unsupported:
                    cleaned_org.domain = None
                verified_organizations.append(cleaned_org)

        # ── 6. Verify Projects ──────────────────────────────────────────────
        for idx, proj in enumerate(metadata.projects):
            fact_id = f"project_{idx}"
            raw_text = _get_raw_text(proj.source_frame_ids)
            
            unsupported = []
            if not is_supported(raw_text, proj.name):
                unsupported.append("name")
            if proj.description and not is_supported(raw_text, proj.description):
                unsupported.append("description")

            total_fields = 1 + bool(proj.description)
            status = self._calc_status("name" not in unsupported, total_fields, len(unsupported))

            evidence = self._make_evidence(fact_id, "project", proj.model_dump(), proj.source_frame_ids, timestamps_map, raw_text, status, unsupported)
            evidences.append(evidence)

            if status == VerificationStatus.VERIFIED:
                verified_projects.append(proj)
            elif status == VerificationStatus.PARTIALLY_SUPPORTED:
                cleaned_proj = proj.model_copy()
                if "description" in unsupported:
                    cleaned_proj.description = None
                verified_projects.append(cleaned_proj)

        # ── 7. Verify URLs ──────────────────────────────────────────────────
        for idx, url_ref in enumerate(metadata.urls):
            fact_id = f"url_{idx}"
            raw_text = _get_raw_text(url_ref.source_frame_ids)
            
            unsupported = []
            if not is_supported_url(raw_text, url_ref.url):
                unsupported.append("url")
            if url_ref.title:
                # Flexible title check: match full title or substantial keyword
                title_clean = normalize_text(url_ref.title)
                title_words = [w for w in title_clean.split() if len(w) > 3]
                if not is_supported(raw_text, url_ref.title) and not any(is_supported(raw_text, w) for w in title_words):
                    unsupported.append("title")

            total_fields = 1 + bool(url_ref.title)
            status = self._calc_status("url" not in unsupported, total_fields, len(unsupported))

            evidence = self._make_evidence(fact_id, "url", url_ref.model_dump(), url_ref.source_frame_ids, timestamps_map, raw_text, status, unsupported)
            evidences.append(evidence)

            if status == VerificationStatus.VERIFIED:
                verified_urls.append(url_ref)
            elif status == VerificationStatus.PARTIALLY_SUPPORTED:
                cleaned_url = url_ref.model_copy()
                if "title" in unsupported:
                    cleaned_url.title = None
                verified_urls.append(cleaned_url)

        # Return consolidated verified metadata container
        verified_metadata = StructuredMetadata(
            meetings=verified_meetings,
            files=verified_files,
            appointments=verified_appointments,
            people=verified_people,
            organizations=verified_organizations,
            projects=verified_projects,
            urls=verified_urls,
        )

        return verified_metadata, evidences

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _calc_status(
        core_supported: bool,
        total_fields: int,
        unsupported_count: int,
    ) -> VerificationStatus:
        """Helper to decide verification status based on query matches."""
        if not core_supported:
            return VerificationStatus.UNSUPPORTED
        if unsupported_count == 0:
            return VerificationStatus.VERIFIED
        return VerificationStatus.PARTIALLY_SUPPORTED

    @staticmethod
    def _make_evidence(
        fact_id: str,
        fact_type: str,
        fact: dict[str, Any],
        source_frame_ids: list[int],
        timestamps_map: Optional[dict[int, str]],
        raw_text: str,
        status: VerificationStatus,
        unsupported: list[str],
    ) -> FactEvidence:
        """Build a FactEvidence Pydantic object."""
        primary_frame = source_frame_ids[0] if source_frame_ids else None
        primary_ts = None
        if primary_frame is not None and timestamps_map:
            primary_ts = timestamps_map.get(primary_frame)

        # Find matching snippets in raw text (evidence text)
        evidence_snippets: list[str] = []
        lines = raw_text.splitlines()

        # Find matching lines for any non-unsupported string field
        for key, val in fact.items():
            if key in unsupported or not val:
                continue
            if isinstance(val, str) and len(val) > 3:
                for line in lines:
                    if is_supported(line, val):
                        evidence_snippets.append(line.strip())
                        break

        evidence_text = " | ".join(evidence_snippets) if evidence_snippets else None

        return FactEvidence(
            fact_id=fact_id,
            fact_type=fact_type,
            fact=fact,
            source_frame=primary_frame,
            source_timestamp=primary_ts,
            evidence_text=evidence_text,
            verification_status=status,
            unsupported_fields=unsupported,
        )
