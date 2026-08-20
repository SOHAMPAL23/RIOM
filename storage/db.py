from __future__ import annotations

import json
import re
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from storage.models import SCHEMA_SQL


class Database:

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._local = threading.local()
        # Ensure schema exists
        self._init_schema()

    # ------------------------------------------------------------------
    # Connection management (per-thread)
    # ------------------------------------------------------------------

    def _conn(self) -> sqlite3.Connection:
        """Return (or create) this thread's connection."""
        if not hasattr(self._local, "conn"):
            conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return self._local.conn

    def _init_schema(self) -> None:
        conn = self._conn()
        conn.executescript(SCHEMA_SQL)
        conn.commit()

    def close(self) -> None:
        """Close this thread's connection if open."""
        conn = getattr(self._local, "conn", None)
        if conn:
            conn.close()
            del self._local.conn

    # ------------------------------------------------------------------
    # frames table
    # ------------------------------------------------------------------

    def insert_frame(
        self,
        captured_at: datetime,
        image_path: str,
        width: int,
        height: int,
        image_hash: Optional[str] = None,
        application: Optional[str] = None,
        window_title: Optional[str] = None,
        monitor: int = 1,
        capture_reason: str = "visual_change",
        diff_score: float = 0.0,
    ) -> int:
        """Insert a new frame record and return its auto-incremented ID."""
        conn = self._conn()
        with conn:
            cur = conn.execute(
                """
                INSERT INTO frames
                    (captured_at, image_path, image_hash, width, height,
                     application, window_title, monitor, capture_reason, diff_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    captured_at.isoformat(),
                    image_path,
                    image_hash,
                    width,
                    height,
                    application,
                    window_title,
                    monitor,
                    capture_reason,
                    diff_score,
                ),
            )
        return cur.lastrowid  # type: ignore

    def update_frame_image_path(self, frame_id: int, image_path: str) -> None:
        conn = self._conn()
        with conn:
            conn.execute(
                "UPDATE frames SET image_path = ? WHERE id = ?",
                (image_path, frame_id),
            )

    def get_capture_records(self, limit: int = 50) -> list[dict]:
        """Return recent CaptureRecord-compatible rows, newest first."""
        conn = self._conn()
        rows = conn.execute(
            """
            SELECT id, captured_at, image_path, application, window_title,
                   monitor, width, height, capture_reason, diff_score
            FROM frames
            ORDER BY captured_at DESC LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_frames_by_application(self, application: str, limit: int = 100) -> list[dict]:
        """Return frames where the foreground application matches."""
        conn = self._conn()
        rows = conn.execute(
            """
            SELECT id, captured_at, image_path, application, window_title,
                   monitor, width, height, capture_reason, diff_score
            FROM frames
            WHERE application = ?
            ORDER BY captured_at DESC LIMIT ?
            """,
            (application, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_frames_by_reason(self, reason: str, limit: int = 100) -> list[dict]:
        """Return frames saved for a specific CaptureReason."""
        conn = self._conn()
        rows = conn.execute(
            """
            SELECT id, captured_at, image_path, application, window_title,
                   monitor, width, height, capture_reason, diff_score
            FROM frames
            WHERE capture_reason = ?
            ORDER BY captured_at DESC LIMIT ?
            """,
            (reason, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def update_frame_ocr(self, frame_id: int, raw_text: str) -> None:
        """Store the OCR full-text result and mark OCR as done."""
        conn = self._conn()
        with conn:
            conn.execute(
                "UPDATE frames SET raw_text = ?, ocr_processed = 1 WHERE id = ?",
                (raw_text, frame_id),
            )

    def mark_frame_llm_done(self, frame_id: int) -> None:
        conn = self._conn()
        with conn:
            conn.execute(
                "UPDATE frames SET llm_processed = 1 WHERE id = ?",
                (frame_id,),
            )

    def get_pending_ocr_frames(self, limit: int = 10) -> list[sqlite3.Row]:
        """Return frames that have not yet been OCR-processed."""
        conn = self._conn()
        return conn.execute(
            "SELECT * FROM frames WHERE ocr_processed = 0 ORDER BY captured_at LIMIT ?",
            (limit,),
        ).fetchall()

    def get_pending_llm_frames(self, limit: int = 5) -> list[sqlite3.Row]:
        """Return frames with OCR text that have not yet been LLM-processed."""
        conn = self._conn()
        return conn.execute(
            """
            SELECT * FROM frames
            WHERE ocr_processed = 1 AND llm_processed = 0 AND raw_text IS NOT NULL
            ORDER BY captured_at LIMIT ?
            """,
            (limit,),
        ).fetchall()

    def get_recent_frames(self, limit: int = 50) -> list[sqlite3.Row]:
        conn = self._conn()
        return conn.execute(
            "SELECT * FROM frames ORDER BY captured_at DESC LIMIT ?",
            (limit,),
        ).fetchall()

    # ------------------------------------------------------------------
    # ocr_blocks table
    # ------------------------------------------------------------------

    def insert_ocr_blocks(self, frame_id: int, blocks: list[dict]) -> None:
        """Bulk-insert OCR text blocks for a frame."""
        conn = self._conn()
        with conn:
            conn.executemany(
                """
                INSERT INTO ocr_blocks (frame_id, text, confidence, x1, y1, x2, y2)
                VALUES (:frame_id, :text, :confidence, :x1, :y1, :x2, :y2)
                """,
                [
                    {
                        "frame_id":   frame_id,
                        "text":       b["text"],
                        "confidence": b["confidence"],
                        "x1":         b["bbox"]["x1"],
                        "y1":         b["bbox"]["y1"],
                        "x2":         b["bbox"]["x2"],
                        "y2":         b["bbox"]["y2"],
                    }
                    for b in blocks
                ],
            )

    # ------------------------------------------------------------------
    # raw_text_records table  (Stage 2 — OCR output)
    # ------------------------------------------------------------------

    def insert_raw_text_record(self, record: "RawTextRecord") -> int:  # type: ignore[name-defined]
        """
        Insert a RawTextRecord and return its auto-incremented id.

        The caller must have already imported RawTextRecord from ocr.models.
        We accept Any-typed record to avoid a circular import here.
        """
        conn = self._conn()
        now  = datetime.now(timezone.utc).isoformat()
        with conn:
            cur = conn.execute(
                """
                INSERT INTO raw_text_records
                    (frame_id, timestamp, image_path, application, window_title,
                     raw_text, confidence, ocr_engine, blocks_json,
                     char_count, is_empty, ocr_error, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.frame_id,
                    record.timestamp.isoformat(),
                    record.image_path,
                    record.application,
                    record.window_title,
                    record.raw_text,
                    record.confidence,
                    record.ocr_engine,
                    record.blocks_json,
                    record.char_count,
                    int(record.is_empty),
                    record.ocr_error,
                    now,
                ),
            )
        return cur.lastrowid  # type: ignore

    def get_raw_text_record(self, frame_id: int) -> Optional[dict]:
        """Return the RawTextRecord for a specific frame, or None."""
        conn = self._conn()
        row = conn.execute(
            "SELECT * FROM raw_text_records WHERE frame_id = ? LIMIT 1",
            (frame_id,),
        ).fetchone()
        return dict(row) if row else None

    def get_raw_text_records(self, limit: int = 50) -> list[dict]:
        """Return recent RawTextRecords, newest first."""
        conn = self._conn()
        rows = conn.execute(
            """
            SELECT r.*, f.capture_reason, f.diff_score
            FROM raw_text_records r
            JOIN frames f ON f.id = r.frame_id
            ORDER BY r.timestamp DESC LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def search_raw_text(self, query: str, limit: int = 50) -> list[dict]:
        """
        Full-text search across raw_text column using SQLite LIKE.

        Returns rows joined with frame provenance (capture_reason, diff_score,
        application, image_path).
        """
        conn = self._conn()
        rows = conn.execute(
            """
            SELECT r.id, r.frame_id, r.timestamp, r.image_path,
                   r.application, r.window_title, r.raw_text,
                   r.confidence, r.ocr_engine, r.char_count, r.is_empty,
                   f.capture_reason, f.diff_score
            FROM raw_text_records r
            JOIN frames f ON f.id = r.frame_id
            WHERE r.raw_text LIKE ?
            ORDER BY r.timestamp DESC LIMIT ?
            """,
            (f"%{query}%", limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_pending_ocr_frames_full(self, limit: int = 10) -> list[dict]:
        """Return full frame rows not yet OCR-processed, as dicts."""
        conn = self._conn()
        rows = conn.execute(
            """
            SELECT id, captured_at, image_path, application, window_title,
                   monitor, width, height, capture_reason, diff_score
            FROM frames
            WHERE ocr_processed = 0
            ORDER BY captured_at
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]




    # ------------------------------------------------------------------
    # merged_text_records table  (Stage 2.5 — TextProcessor output)
    # ------------------------------------------------------------------

    def insert_merged_text_record(self, record: "MergedTextRecord") -> int:  # type: ignore[name-defined]
        """
        Persist a MergedTextRecord and return its auto-incremented id.
        Accepts any duck-typed object with the expected fields to avoid
        a circular import.
        """
        conn = self._conn()
        now  = datetime.now(timezone.utc).isoformat()
        with conn:
            cur = conn.execute(
                """
                INSERT INTO merged_text_records
                    (contributing_frame_ids, contributing_timestamps,
                     contributing_image_paths, first_timestamp, last_timestamp,
                     application, window_title, merged_text, char_count,
                     is_empty, is_deduplicated, frame_count,
                     similarity_scores, ocr_engines, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    json.dumps(record.contributing_frame_ids),
                    json.dumps(record.contributing_timestamps),
                    json.dumps(record.contributing_image_paths),
                    record.first_timestamp.isoformat(),
                    record.last_timestamp.isoformat(),
                    record.application,
                    record.window_title,
                    record.merged_text,
                    record.char_count,
                    int(record.is_empty),
                    int(record.is_deduplicated),
                    record.frame_count,
                    json.dumps(record.similarity_scores),
                    json.dumps(record.ocr_engines),
                    now,
                ),
            )
        return cur.lastrowid  # type: ignore

    def get_merged_text_records(
        self,
        limit: int = 50,
        application: Optional[str] = None,
    ) -> list[dict]:
        """
        Return recent MergedTextRecords, newest first.

        Args:
            limit:       Maximum rows to return.
            application: Filter by application name if provided.
        """
        conn = self._conn()
        if application:
            rows = conn.execute(
                """
                SELECT * FROM merged_text_records
                WHERE application = ?
                ORDER BY last_timestamp DESC LIMIT ?
                """,
                (application, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT * FROM merged_text_records
                ORDER BY last_timestamp DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def search_merged_text(self, query: str, limit: int = 50) -> list[dict]:
        """SQLite LIKE search across merged_text column."""
        conn = self._conn()
        rows = conn.execute(
            """
            SELECT id, contributing_frame_ids, first_timestamp, last_timestamp,
                   application, window_title, merged_text, char_count,
                   is_empty, is_deduplicated, frame_count
            FROM merged_text_records
            WHERE merged_text LIKE ?
            ORDER BY last_timestamp DESC LIMIT ?
            """,
            (f"%{query}%", limit),
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    # screen_contexts table
    # ------------------------------------------------------------------

    def insert_screen_context(
        self, frame_id: int, application: Optional[str], activity_summary: str
    ) -> int:
        conn = self._conn()
        now = datetime.now(timezone.utc).isoformat()
        with conn:
            cur = conn.execute(
                """
                INSERT INTO screen_contexts (frame_id, application, activity_summary, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (frame_id, application, activity_summary, now),
            )
        return cur.lastrowid  # type: ignore

    # ------------------------------------------------------------------
    # entities table
    # ------------------------------------------------------------------

    def insert_entity(self, frame_id: int, entity_type: str, payload: dict) -> int:
        conn = self._conn()
        now = datetime.now(timezone.utc).isoformat()
        with conn:
            cur = conn.execute(
                """
                INSERT INTO entities (frame_id, entity_type, payload, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (frame_id, entity_type, json.dumps(payload), now),
            )
        return cur.lastrowid  # type: ignore

    def get_entities_by_type(self, entity_type: str, limit: int = 100) -> list[dict]:
        """Return all stored entities of a given type, newest first."""
        conn = self._conn()
        rows = conn.execute(
            """
            SELECT e.id, e.frame_id, e.entity_type, e.payload, e.created_at,
                   f.captured_at, f.image_path
            FROM entities e
            JOIN frames f ON f.id = e.frame_id
            WHERE e.entity_type = ?
            ORDER BY e.created_at DESC LIMIT ?
            """,
            (entity_type, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def search_entities(self, query: str, limit: int = 50) -> list[dict]:
        """Full-text search across entity payloads using SQLite LIKE."""
        conn = self._conn()
        rows = conn.execute(
            """
            SELECT e.*, f.captured_at, f.image_path
            FROM entities e
            JOIN frames f ON f.id = e.frame_id
            WHERE e.payload LIKE ?
            ORDER BY e.created_at DESC LIMIT ?
            """,
            (f"%{query}%", limit),
        ).fetchall()
        return [dict(r) for r in rows]

    # ==================================================================
    # Stage 4 Required Repository/Database Functions
    # ==================================================================

    def save_capture(
        self,
        timestamp: datetime,
        image_path: str,
        width: int,
        height: int,
        application: Optional[str] = None,
        window_title: Optional[str] = None,
        diff_score: float = 0.0,
        capture_reason: str = "visual_change",
    ) -> int:
        """
        Saves a capture record. Avoids storing duplicate capture records
        by checking if the image_path already exists.
        """
        conn = self._conn()
        row = conn.execute("SELECT id FROM frames WHERE image_path = ?", (image_path,)).fetchone()
        if row:
            return row["id"]

        return self.insert_frame(
            captured_at=timestamp,
            image_path=image_path,
            width=width,
            height=height,
            application=application,
            window_title=window_title,
            diff_score=diff_score,
            capture_reason=capture_reason,
        )

    def save_raw_text(
        self,
        frame_id: int,
        raw_text: str,
        confidence: Optional[float] = None,
        ocr_engine: str = "unknown",
        blocks_json: str = "[]",
    ) -> int:
        """
        Saves raw OCR text. Avoids duplicates by updating existing raw text
        record if it exists for the frame_id.
        """
        conn = self._conn()
        row = conn.execute("SELECT id FROM raw_text_records WHERE frame_id = ?", (frame_id,)).fetchone()
        if row:
            with conn:
                conn.execute(
                    """
                    UPDATE raw_text_records
                    SET raw_text = ?, confidence = ?, ocr_engine = ?, blocks_json = ?
                    WHERE id = ?
                    """,
                    (raw_text, confidence, ocr_engine, blocks_json, row["id"]),
                )
            self.update_frame_ocr(frame_id, raw_text)
            return row["id"]

        from ocr.models import RawTextRecord
        rec = RawTextRecord(
            frame_id=frame_id,
            timestamp=datetime.now(timezone.utc),  # Placeholder, will be persisted in table
            image_path="",  # Fallback
            raw_text=raw_text,
            confidence=confidence,
            ocr_engine=ocr_engine,
            blocks_json=blocks_json,
            char_count=len(raw_text),
            is_empty=not bool(raw_text.strip()),
        )
        # We query the frame to populate correct image_path and timestamp if available
        frame = conn.execute("SELECT captured_at, image_path FROM frames WHERE id = ?", (frame_id,)).fetchone()
        if frame:
            rec.timestamp = datetime.fromisoformat(frame["captured_at"])
            rec.image_path = frame["image_path"]

        return self.insert_raw_text_record(rec)

    def save_metadata(self, metadata: "StructuredMetadata") -> None:
        """
        Save all entities inside StructuredMetadata to the entities table,
        avoiding duplicate insertions across frames by merging and updating.
        """
        def _normalize_str(text: Optional[str]) -> str:
            if not text:
                return ""
            t = re.sub(r"^https?://(www\.)?", "", str(text).strip(), flags=re.IGNORECASE).rstrip("/")
            return re.sub(r"[^\w\s-]", "", t).strip().lower()

        def save_entity(frame_id: int, entity_type: str, payload: dict, unique_keys: list[str]):
            conn = self._conn()
            # Check recent entities of this type across all frames
            rows = conn.execute(
                "SELECT id, payload, frame_id FROM entities WHERE entity_type = ? ORDER BY id DESC LIMIT 100",
                (entity_type,),
            ).fetchall()

            for r in rows:
                p = json.loads(r["payload"]) if isinstance(r["payload"], str) else r["payload"]

                # 1. Deduplicate Meetings
                if entity_type == "meeting":
                    m_link_exist = _normalize_str(p.get("meeting_link"))
                    m_link_new = _normalize_str(payload.get("meeting_link"))
                    title_exist = _normalize_str(p.get("title"))
                    title_new = _normalize_str(payload.get("title"))

                    is_dup = False
                    if m_link_exist and m_link_new and m_link_exist == m_link_new:
                        is_dup = True
                    elif title_exist and title_new and (title_exist == title_new or title_exist in title_new or title_new in title_exist):
                        is_dup = True

                    if is_dup:
                        merged = dict(p)
                        merged.update({k: v for k, v in payload.items() if v and not p.get(k)})
                        for list_key in ["participants", "emails", "discussion_points", "action_items", "source_frame_ids", "source_timestamps"]:
                            existing_list = p.get(list_key) or []
                            new_list = payload.get(list_key) or []
                            merged[list_key] = list(dict.fromkeys(existing_list + new_list))
                        if payload.get("meeting_link"):
                            merged["meeting_link"] = payload["meeting_link"]
                        with conn:
                            conn.execute("UPDATE entities SET payload = ?, frame_id = ? WHERE id = ?", (json.dumps(merged), frame_id, r["id"]))
                        return

                # 2. Deduplicate URLs
                elif entity_type == "url_reference":
                    url_exist = _normalize_str(p.get("url"))
                    url_new = _normalize_str(payload.get("url"))
                    if url_exist and url_new and url_exist == url_new:
                        merged = dict(p)
                        if payload.get("title") and payload["title"] != payload.get("url"):
                            merged["title"] = payload["title"]
                        with conn:
                            conn.execute("UPDATE entities SET payload = ?, frame_id = ? WHERE id = ?", (json.dumps(merged), frame_id, r["id"]))
                        return

                # 3. Deduplicate General Entities (Files, People, Orgs, Projects, Appointments)
                else:
                    match = True
                    for k in unique_keys:
                        v1 = _normalize_str(p.get(k))
                        v2 = _normalize_str(payload.get(k))
                        if not v1 or not v2 or v1 != v2:
                            match = False
                            break
                    if match:
                        merged = dict(p)
                        merged.update({k: v for k, v in payload.items() if v})
                        with conn:
                            conn.execute("UPDATE entities SET payload = ?, frame_id = ? WHERE id = ?", (json.dumps(merged), frame_id, r["id"]))
                        return

            self.insert_entity(frame_id, entity_type, payload)

        for m in metadata.meetings:
            fid = m.source_frame_ids[0] if m.source_frame_ids else 0
            save_entity(fid, "meeting", m.model_dump(), ["title"])
        for f in metadata.files:
            fid = f.source_frame_ids[0] if f.source_frame_ids else 0
            save_entity(fid, "file_activity", f.model_dump(), ["file_name", "document_title"])
        for appt in metadata.appointments:
            fid = appt.source_frame_ids[0] if appt.source_frame_ids else 0
            save_entity(fid, "appointment", appt.model_dump(), ["title"])
        for p in metadata.people:
            fid = p.source_frame_ids[0] if p.source_frame_ids else 0
            save_entity(fid, "person", p.model_dump(), ["name"])
        for org in metadata.organizations:
            fid = org.source_frame_ids[0] if org.source_frame_ids else 0
            save_entity(fid, "organization", org.model_dump(), ["name"])
        for proj in metadata.projects:
            fid = proj.source_frame_ids[0] if proj.source_frame_ids else 0
            save_entity(fid, "project", proj.model_dump(), ["name"])
        for u in metadata.urls:
            fid = u.source_frame_ids[0] if u.source_frame_ids else 0
            save_entity(fid, "url_reference", u.model_dump(), ["url"])

    def save_fact_evidence(self, evidence: "FactEvidence") -> int:
        """
        Saves a metadata verification fact evidence record.
        Avoids duplicates by updating if the fact_id already exists.
        """
        conn = self._conn()
        row = conn.execute("SELECT id FROM fact_evidences WHERE fact_id = ?", (evidence.fact_id,)).fetchone()
        if row:
            with conn:
                conn.execute(
                    """
                    UPDATE fact_evidences
                    SET fact_type = ?, fact = ?, source_frame = ?, source_timestamp = ?,
                        evidence_text = ?, verification_status = ?
                    WHERE id = ?
                    """,
                    (
                        evidence.fact_type,
                        json.dumps(evidence.fact),
                        evidence.source_frame,
                        evidence.source_timestamp,
                        evidence.evidence_text,
                        evidence.verification_status,
                        row["id"],
                    ),
                )
            return row["id"]

        now = datetime.now(timezone.utc).isoformat()
        with conn:
            cur = conn.execute(
                """
                INSERT INTO fact_evidences
                    (fact_id, fact_type, fact, source_frame, source_timestamp,
                     evidence_text, verification_status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    evidence.fact_id,
                    evidence.fact_type,
                    json.dumps(evidence.fact),
                    evidence.source_frame,
                    evidence.source_timestamp,
                    evidence.evidence_text,
                    evidence.verification_status,
                    now,
                ),
            )
        return cur.lastrowid  # type: ignore

    def get_session(self) -> sqlite3.Connection:
        """Returns the thread's active database connection session."""
        return self._conn()

    def search_text(self, query: str) -> list[dict]:
        """Searches raw text in raw text records."""
        return self.search_raw_text(query)

    def get_metadata(self, limit: int = 50, entity_type: Optional[str] = None) -> list[dict]:
        """Retrieves metadata entities. Filters by entity_type if provided."""
        conn = self._conn()
        if entity_type:
            rows = conn.execute(
                "SELECT * FROM entities WHERE entity_type = ? ORDER BY created_at DESC LIMIT ?",
                (entity_type, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM entities ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_captures_by_timestamp(self, start: datetime, end: datetime) -> list[dict]:
        """Query captured frames within a timestamp range."""
        conn = self._conn()
        rows = conn.execute(
            """
            SELECT * FROM frames
            WHERE captured_at BETWEEN ? AND ?
            ORDER BY captured_at DESC
            """,
            (start.isoformat(), end.isoformat()),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_captures_by_application(self, app_name: str) -> list[dict]:
        """Query captured frames for a foreground application."""
        conn = self._conn()
        rows = conn.execute(
            """
            SELECT * FROM frames
            WHERE application = ?
            ORDER BY captured_at DESC
            """,
            (app_name,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_records_by_time(self, start: datetime, end: datetime) -> list[dict]:
        """Query raw text records within a timestamp range."""
        conn = self._conn()
        rows = conn.execute(
            """
            SELECT * FROM raw_text_records
            WHERE timestamp BETWEEN ? AND ?
            ORDER BY timestamp DESC
            """,
            (start.isoformat(), end.isoformat()),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_records_by_application(self, application: str, limit: int = 100) -> list[dict]:
        """Query raw text records for a given application."""
        conn = self._conn()
        rows = conn.execute(
            """
            SELECT * FROM raw_text_records
            WHERE application = ?
            ORDER BY timestamp DESC LIMIT ?
            """,
            (application, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_record(self, record_id: int) -> Optional[dict]:
        """Fetch a specific raw text record by its primary key ID."""
        conn = self._conn()
        row = conn.execute(
            "SELECT * FROM raw_text_records WHERE id = ? LIMIT 1",
            (record_id,),
        ).fetchone()
        return dict(row) if row else None

    def get_source_frames(self, record_id: int) -> list[dict]:
        """Retrieve the source frame(s) corresponding to a raw text record."""
        rec = self.get_record(record_id)
        if not rec:
            return []
        conn = self._conn()
        frame_id = rec.get("frame_id")
        if not frame_id:
            return []
        rows = conn.execute("SELECT * FROM frames WHERE id = ?", (frame_id,)).fetchall()
        return [dict(r) for r in rows]

    def answer_query(self, question: str) -> dict[str, Any]:
        """
        Answers natural language queries against extracted metadata and raw text.
        Supports demo questions like:
        - 'What happened between 10 AM and 12 PM?'
        - 'Which files did I work on?'
        - 'Which meetings did I attend?'
        - 'Who appeared repeatedly today?'
        - 'What deadlines were visible?'
        - 'What applications did I use?'
        - 'Show the source frame for this extracted fact.'
        """
        q = question.lower()
        conn = self._conn()

        if "file" in q or "work on" in q or "code" in q:
            entities = self.get_entities_by_type("file_activity")
            results = []
            for e in entities:
                p = json.loads(e["payload"]) if isinstance(e["payload"], str) else e["payload"]
                name = p.get("file_name") or p.get("document_title") or "Unnamed file"
                results.append({
                    "entity": name,
                    "application": p.get("application"),
                    "duration": p.get("estimated_duration"),
                    "source_frame": p.get("source_frame_ids"),
                })
            return {"query": question, "category": "files", "count": len(results), "results": results}

        elif "meeting" in q or "call" in q or "sync" in q or "attend" in q:
            entities = self.get_entities_by_type("meeting")
            results = []
            for e in entities:
                p = json.loads(e["payload"]) if isinstance(e["payload"], str) else e["payload"]
                results.append({
                    "title": p.get("title"),
                    "platform": p.get("platform"),
                    "participants": p.get("participants"),
                    "discussion_points": p.get("discussion_points"),
                    "action_items": p.get("action_items"),
                    "source_frame": p.get("source_frame_ids"),
                })
            return {"query": question, "category": "meetings", "count": len(results), "results": results}

        elif "who" in q or "people" in q or "person" in q or "appear" in q:
            entities = self.get_entities_by_type("person")
            results = []
            for e in entities:
                p = json.loads(e["payload"]) if isinstance(e["payload"], str) else e["payload"]
                results.append({
                    "name": p.get("name"),
                    "email": p.get("email"),
                    "organization": p.get("organization"),
                    "source_frame": p.get("source_frame_ids"),
                })
            return {"query": question, "category": "people", "count": len(results), "results": results}

        elif "deadline" in q or "appointment" in q or "reminder" in q or "due" in q:
            entities = self.get_entities_by_type("appointment")
            results = []
            for e in entities:
                p = json.loads(e["payload"]) if isinstance(e["payload"], str) else e["payload"]
                results.append({
                    "title": p.get("title"),
                    "time": p.get("time"),
                    "deadline": p.get("deadline"),
                    "reminder": p.get("reminder"),
                    "source_frame": p.get("source_frame_ids"),
                })
            return {"query": question, "category": "deadlines", "count": len(results), "results": results}

        elif "application" in q or "apps" in q:
            rows = conn.execute("SELECT DISTINCT application, COUNT(*) as frame_count FROM frames WHERE application IS NOT NULL GROUP BY application").fetchall()
            results = [{"application": r["application"], "frame_count": r["frame_count"]} for r in rows]
            return {"query": question, "category": "applications", "count": len(results), "results": results}

        elif "source frame" in q or "evidence" in q:
            evs = conn.execute("SELECT * FROM fact_evidences ORDER BY id DESC LIMIT 20").fetchall()
            results = []
            for ev in evs:
                f_data = json.loads(ev["fact"]) if isinstance(ev["fact"], str) else ev["fact"]
                results.append({
                    "fact_id": ev["fact_id"],
                    "fact_type": ev["fact_type"],
                    "source_frame": ev["source_frame"],
                    "evidence_text": ev["evidence_text"],
                    "verification_status": ev["verification_status"],
                })
            return {"query": question, "category": "fact_evidences", "count": len(results), "results": results}

        else:
            # Default: general activity summary
            frames = self.get_capture_records(limit=20)
            return {"query": question, "category": "activity_timeline", "count": len(frames), "results": frames}

