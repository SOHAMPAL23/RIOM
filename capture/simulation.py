"""
capture/simulation.py

Simulated working session for demonstrating the complete end-to-end RIOM pipeline.

Generates a realistic multi-app workflow across a simulated workday:
  1. 10:30 — Google Meet: "Sprint Planning with Alice Chen & Bob Smith"
  2. 11:15 — VS Code: "project.py — RIOM Ambient Screen Engine"
  3. 12:00 — Gmail: "Email from Sarah Connor about Q3 Roadmap Sync on Friday 2:00 PM"
  4. 13:30 — Google Docs: "Project Brief: Antigravity AI Work Memory"

Renders synthetic screenshot images to disk, inserts capture records, runs
OCR, Text Cleaning, Metadata Extraction (real LLM or deterministic fallback),
Fact Verification, and SQLite persistence.
"""
from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ── Make package importable when run directly ──────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from capture.change_detector import CaptureReason
from capture.models import CaptureRecord
from config.settings import settings
from metadata.extractor import MetadataExtractor
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
from metadata.verifier import MetadataVerifier
from ocr.models import RawTextRecord
from processing.privacy_filter import PrivacyFilter
from processing.text_processor import TextProcessor
from storage.db import Database
from storage.file_manager import FileManager

logger = logging.getLogger(__name__)


# ===========================================================================
# Simulated Session Scenarios
# ===========================================================================

SIMULATED_SCENARIOS = [
    {
        "time_str": "2026-08-15T10:30:00+00:00",
        "app": "chrome.exe",
        "window_title": "Google Meet — Sprint Planning — Google Chrome",
        "bg_color": (30, 25, 20),      # Dark Blue-ish (BGR)
        "accent_color": (255, 166, 88), # Light Blue
        "text": (
            "Google Meet | meet.google.com\n"
            "Sprint Planning — Q3 RIOM Engineering\n"
            "Participants (3): Alice Chen, Bob Smith, Soham Pal\n"
            "Platform: Google Meet\n"
            "Discussion:\n"
            "- SQLite WAL mode and schema indexing verification\n"
            "- Multi-stage pipeline resilience and non-blocking background workers\n"
            "- PySide6 Work Memory Dashboard dark mode interface\n"
            "Action Items:\n"
            "- Alice Chen to review SQLite entity schema\n"
            "- Bob Smith to test background thread graceful shutdown\n"
        ),
        "entities": {
            "meetings": [
                Meeting(
                    title="Sprint Planning — Q3 RIOM Engineering",
                    participants=["Alice Chen", "Bob Smith", "Soham Pal"],
                    time="10:30 AM",
                    platform="Google Meet",
                    meeting_link="https://meet.google.com",
                    emails=["alice.chen@riom.ai", "bob.smith@riom.ai", "soham@riom.ai"],
                    discussion_points=[
                        "SQLite WAL mode and schema indexing verification",
                        "Multi-stage pipeline resilience and non-blocking background workers",
                        "PySide6 Work Memory Dashboard dark mode interface",
                    ],
                    action_items=[
                        "Alice Chen to review SQLite entity schema",
                        "Bob Smith to test background thread graceful shutdown",
                    ],
                )
            ],
            "people": [
                Person(name="Alice Chen", organization="RIOM Team"),
                Person(name="Bob Smith", organization="RIOM Team"),
            ],
            "urls": [],
        },
    },
    {
        "time_str": "2026-08-15T11:15:00+00:00",
        "app": "code.exe",
        "window_title": "project.py — RIOM — Visual Studio Code",
        "bg_color": (25, 20, 15),
        "accent_color": (80, 185, 63), # Green
        "text": (
            "Visual Studio Code — github.com/microsoft/vscode\n"
            "File: project.py\n"
            "Path: c:/Users/Soham/OneDrive/Desktop/RIOM/project.py\n"
            "class PipelineCoordinator:\n"
            "    def __init__(self, db: Database, file_manager: FileManager):\n"
            "        self._db = db\n"
            "        self._file_manager = file_manager\n"
            "    def start_pipeline(self):\n"
            "        logger.info('Continuous ambient screen understanding active')\n"
        ),
        "entities": {
            "files": [
                FileActivity(
                    file_name="project.py",
                    file_path="c:/Users/Soham/OneDrive/Desktop/RIOM/project.py",
                    document_title="project.py — Visual Studio Code",
                    application="VS Code",
                    start_time="11:15 AM",
                    estimated_duration="45 minutes",
                )
            ],
            "projects": [
                Project(name="RIOM", description="Continuous ambient screen understanding system"),
            ],
            "urls": [
                URLReference(url="https://github.com/microsoft/vscode", title="Visual Studio Code Repository"),
            ],
        },
    },
    {
        "time_str": "2026-08-15T12:00:00+00:00",
        "app": "chrome.exe",
        "window_title": "Inbox (1) — Gmail — Sarah Connor: Q3 Roadmap Sync — Google Chrome",
        "bg_color": (20, 22, 28),
        "accent_color": (34, 153, 210), # Amber
        "text": (
            "Gmail — mail.google.com\n"
            "From: Sarah Connor <sarah@antigravity.io>\n"
            "To: team@riom.ai\n"
            "Subject: Q3 Roadmap Sync\n"
            "Hi team, let's schedule our Q3 Roadmap Sync for Friday at 2:00 PM.\n"
            "Deadline for slide proposals is Thursday 5:00 PM.\n"
            "Organization: Antigravity Systems\n"
        ),
        "entities": {
            "appointments": [
                Appointment(
                    title="Q3 Roadmap Sync",
                    date="Friday",
                    time="2:00 PM",
                    deadline="Thursday 5:00 PM",
                )
            ],
            "people": [
                Person(name="Sarah Connor", email="sarah@antigravity.io", organization="Antigravity Systems"),
            ],
            "organizations": [
                Organization(name="Antigravity Systems", domain="antigravity.io"),
            ],
            "urls": [
                URLReference(url="https://mail.google.com", title="Gmail Inbox"),
            ],
        },
    },
    {
        "time_str": "2026-08-15T13:30:00+00:00",
        "app": "chrome.exe",
        "window_title": "Project Brief: Antigravity AI Work Memory — Google Docs",
        "bg_color": (28, 20, 24),
        "accent_color": (255, 214, 165),
        "text": (
            "Google Docs — docs.google.com\n"
            "Document: Project Brief: Antigravity AI Work Memory\n"
            "Organization: Antigravity Systems\n"
            "Project: AI Work Memory Engine\n"
            "Overview: Construct a lightweight memory layer for desktop knowledge workers.\n"
        ),
        "entities": {
            "files": [
                FileActivity(
                    document_title="Project Brief: Antigravity AI Work Memory",
                    application="Google Docs",
                    start_time="1:30 PM",
                )
            ],
            "projects": [
                Project(name="AI Work Memory Engine", description="Lightweight memory layer for desktop knowledge workers"),
            ],
            "urls": [
                URLReference(url="https://docs.google.com", title="Project Brief: Antigravity AI Work Memory"),
            ],
        },
    },
]


# ===========================================================================
# Synthetic Screenshot Generator
# ===========================================================================

def generate_mock_screenshot(
    app: str,
    window_title: str,
    text_content: str,
    bg_color: tuple[int, int, int] = (25, 20, 15),
    accent_color: tuple[int, int, int] = (255, 166, 88),
    width: int = 1280,
    height: int = 720,
) -> np.ndarray:
    """
    Renders a clean synthetic desktop frame with title bar, window chrome,
    and text content for OCR and visual verification.
    """
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = bg_color

    # Title bar
    cv2.rectangle(frame, (0, 0), (width, 40), (45, 40, 35), -1)
    cv2.circle(frame, (20, 20), 6, (70, 70, 230), -1)  # Red close
    cv2.circle(frame, (38, 20), 6, (70, 200, 230), -1) # Yellow min
    cv2.circle(frame, (56, 20), 6, (70, 200, 70), -1)  # Green zoom

    cv2.putText(
        frame,
        window_title[:80],
        (80, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )

    # Content Box
    cv2.rectangle(frame, (40, 60), (width - 40, height - 40), (35, 30, 25), -1)
    cv2.rectangle(frame, (40, 60), (width - 40, height - 40), accent_color, 1)

    # Render lines of text
    y = 100
    for line in text_content.splitlines():
        if y > height - 60:
            break
        cv2.putText(
            frame,
            line[:100],
            (60, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (230, 235, 240),
            1,
            cv2.LINE_AA,
        )
        y += 28

    return frame


# ===========================================================================
# Simulation Runner
# ===========================================================================

def run_simulation(
    db: Optional[Database] = None,
    file_manager: Optional[FileManager] = None,
    data_dir: Optional[Path] = None,
) -> int:
    """
    Runs the complete simulated work session end-to-end.

    Flow:
      1. Renders synthetic frames.
      2. Saves WebP files via FileManager.
      3. Inserts CaptureRecords into SQLite frames table.
      4. Saves RawTextRecords in raw_text_records table.
      5. Runs TextProcessor for deduplication/cleaning.
      6. Runs Metadata Extraction (using LLM or deterministic fallback).
      7. Runs MetadataVerifier to generate verified facts and FactEvidence.
      8. Persists all entities, fact evidences, and marks frames as done.

    Returns:
        Number of simulated frames successfully processed.
    """
    data_dir = data_dir or settings.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    db = db or Database(db_path=settings.db_path)
    file_manager = file_manager or FileManager(data_dir=data_dir, webp_quality=settings.webp_quality)

    logger.info("🎬 Starting simulated work session (%d scenarios)...", len(SIMULATED_SCENARIOS))

    text_processor = TextProcessor()
    verifier = MetadataVerifier()

    # Optional real LLM client
    llm_client: Optional[LLMClient] = None
    if settings.llm_api_key:
        try:
            llm_client = LLMClient(api_key=settings.llm_api_key, model=settings.llm_model)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not initialise LLM for simulation: %s", exc)

    # Clean up prior simulation frames if re-running so entries don't duplicate
    try:
        conn = db.get_session()
        with conn:
            conn.execute("DELETE FROM fact_evidences WHERE fact_id LIKE 'sim_%' OR fact_id LIKE 'meeting_%'")
            conn.execute("DELETE FROM entities WHERE frame_id IN (SELECT id FROM frames WHERE application IN ('chrome.exe', 'code.exe') AND capture_reason='application_change')")
    except Exception:  # noqa: BLE001
        pass

    metadata_extractor = MetadataExtractor(llm_client=llm_client) if llm_client else None

    processed_count = 0
    all_raw_records: list[RawTextRecord] = []

    for idx, sc in enumerate(SIMULATED_SCENARIOS):
        ts = datetime.fromisoformat(sc["time_str"])
        app = sc["app"]
        title = sc["window_title"]
        text = sc["text"]

        logger.info("▶ [Stage 1: Capture] Generating simulated frame: %s | %s", app, title)

        # 1. Render frame image
        frame_img = generate_mock_screenshot(
            app=app,
            window_title=title,
            text_content=text,
            bg_color=sc["bg_color"],
            accent_color=sc["accent_color"],
        )

        # 2. Save image to disk
        rel_path = file_manager.save_frame(
            frame=frame_img, timestamp=ts, application=app, window_title=title
        )

        # 3. Insert CaptureRecord to SQLite
        frame_id = db.insert_frame(
            captured_at=ts,
            image_path=rel_path,
            width=1280,
            height=720,
            application=app,
            window_title=title,
            monitor=1,
            capture_reason=CaptureReason.APPLICATION_CHANGE.value,
            diff_score=0.15,
        )

        # Rename image to ID-based filename
        final_path = file_manager.rename_to_id(
            rel_path, frame_id, ts, application=app, window_title=title
        )
        if final_path != rel_path:
            db.update_frame_image_path(frame_id, final_path)

        # 4. Stage 2: OCR Output Persistence
        logger.info("▶ [Stage 2: OCR] Storing OCR text for frame #%d", frame_id)
        raw_rec = RawTextRecord(
            frame_id=frame_id,
            timestamp=ts,
            image_path=final_path,
            application=app,
            window_title=title,
            raw_text=text,
            confidence=0.98,
            ocr_engine="simulation",
            char_count=len(text),
            is_empty=False,
        )
        db.insert_raw_text_record(raw_rec)
        db.update_frame_ocr(frame_id, text)
        all_raw_records.append(raw_rec)
        processed_count += 1

    # 5. Stage 2.5: Text Processing & Merging
    logger.info("▶ [Stage 2.5: TextProcessor] Merging & deduplicating session text...")
    merged_rec = text_processor.process(all_raw_records)
    db.insert_merged_text_record(merged_rec)

    # 6. Stage 3: Metadata Extraction
    logger.info("▶ [Stage 3: LLM Extraction] Extracting structured metadata...")
    extracted_metadata: Optional[StructuredMetadata] = None

    if metadata_extractor:
        try:
            extracted_metadata = metadata_extractor.extract(merged_rec)
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM extraction error during simulation: %s", exc)

    if not extracted_metadata:
        logger.info("ℹ Using deterministic simulated metadata payload.")
        # Construct combined structured metadata from scenarios
        all_meetings = []
        all_files = []
        all_appts = []
        all_people = []
        all_orgs = []
        all_projects = []
        all_urls = []

        for sc_idx, sc in enumerate(SIMULATED_SCENARIOS):
            fid = all_raw_records[sc_idx].frame_id
            ts_str = sc["time_str"]
            ents = sc.get("entities", {})

            for m in ents.get("meetings", []):
                m.source_frame_ids = [fid]
                m.source_timestamps = [ts_str]
                all_meetings.append(m)
            for f in ents.get("files", []):
                f.source_frame_ids = [fid]
                f.source_timestamps = [ts_str]
                all_files.append(f)
            for a in ents.get("appointments", []):
                a.source_frame_ids = [fid]
                a.source_timestamps = [ts_str]
                all_appts.append(a)
            for p in ents.get("people", []):
                p.source_frame_ids = [fid]
                p.source_timestamps = [ts_str]
                all_people.append(p)
            for org in ents.get("organizations", []):
                org.source_frame_ids = [fid]
                org.source_timestamps = [ts_str]
                all_orgs.append(org)
            for proj in ents.get("projects", []):
                proj.source_frame_ids = [fid]
                proj.source_timestamps = [ts_str]
                all_projects.append(proj)
            for u in ents.get("urls", []):
                u.source_frame_ids = [fid]
                u.source_timestamps = [ts_str]
                all_urls.append(u)

        extracted_metadata = StructuredMetadata(
            meetings=all_meetings,
            files=all_files,
            appointments=all_appts,
            people=all_people,
            organizations=all_orgs,
            projects=all_projects,
            urls=all_urls,
        )

    # 7. Stage 3.5: Fact Verification
    logger.info("▶ [Stage 3.5: Verification] Verifying extracted facts against raw text...")
    raw_text_map = {r.frame_id: r.raw_text for r in all_raw_records}
    timestamps_map = {r.frame_id: r.timestamp.isoformat() for r in all_raw_records}

    verified_metadata, evidences = verifier.verify(
        metadata=extracted_metadata,
        raw_text_map=raw_text_map,
        timestamps_map=timestamps_map,
    )

    # 8. Stage 4: SQLite Storage
    logger.info("▶ [Stage 4: Storage] Storing verified metadata and fact evidence...")
    db.save_metadata(verified_metadata)
    for ev in evidences:
        db.save_fact_evidence(ev)

    for rec in all_raw_records:
        db.mark_frame_llm_done(rec.frame_id)

    logger.info(
        "[SIMULATION] Session completed successfully. %d frames captured and processed.",
        processed_count,
    )
    return processed_count


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run_simulation()
