from __future__ import annotations

import logging
import queue
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional, Sequence

# ── Make package importable when run directly ──────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from capture.models import CaptureRecord
from capture.screen_recorder import ScreenRecorder
from config.settings import settings
from metadata.extractor import MetadataExtractor
from metadata.llm_client import LLMClient
from metadata.schemas import StructuredMetadata, FactEvidence
from metadata.verifier import MetadataVerifier
from ocr.models import RawTextRecord
from ocr.pipeline import OCRPipeline
from processing.models import MergedTextRecord
from processing.privacy_filter import PrivacyFilter
from processing.text_processor import TextProcessor, TextProcessorConfig
from storage.db import Database
from storage.file_manager import FileManager

logger = logging.getLogger(__name__)


class PipelineCoordinator:
    """
    Coordinates and executes the complete multi-stage pipeline.

    Args:
        db:             Database instance for persistence.
        file_manager:   FileManager for image storage.
        data_dir:       Root directory for file storage.
        on_status:      Optional callback(stage: str, message: str) for UI notifications.
    """

    def __init__(
        self,
        db: Database,
        file_manager: FileManager,
        data_dir: Optional[Path] = None,
        on_status: Optional[Callable[[str, str], None]] = None,
    ) -> None:
        self._db = db
        self._file_manager = file_manager
        self._data_dir = data_dir or settings.data_dir
        self._on_status = on_status

        # Internal queues between stages
        self._capture_queue: queue.Queue[CaptureRecord] = queue.Queue(maxsize=settings.max_queue_size)
        self._ocr_queue: queue.Queue[RawTextRecord] = queue.Queue(maxsize=settings.max_queue_size)

        # Build pipeline components
        self._ocr_pipeline = OCRPipeline.build(
            db=self._db,
            data_dir=self._data_dir,
            output_queue=self._ocr_queue,
        )

        self._text_processor = TextProcessor(
            config=TextProcessorConfig(
                clean_artifacts=True,
                filter_ui_chrome=True,
                ui_chrome_window=settings.text_ui_chrome_window,
                ui_chrome_min_repeats=settings.text_ui_chrome_min_repeats,
                deduplicate=True,
                similarity_threshold=settings.text_similarity_threshold,
                min_chars_to_compare=settings.text_min_chars_to_compare,
                merge_groups=True,
            )
        )

        # LLM Extractor (optional if API key is not configured)
        self._llm_client: Optional[LLMClient] = None
        if settings.llm_api_key:
            try:
                self._llm_client = LLMClient(
                    api_key=settings.llm_api_key,
                    model=settings.llm_model,
                    base_url=settings.llm_base_url,
                    max_retries=settings.llm_max_retries,
                    timeout=settings.llm_timeout_seconds,
                )
                logger.info("[LLM_EXTRACT] LLM client initialised with model %s.", settings.llm_model)
            except Exception as exc:  # noqa: BLE001
                logger.warning("[LLM_EXTRACT] Could not initialise LLM client: %s", exc)

        self._metadata_extractor = MetadataExtractor(
            llm_client=self._llm_client,
            privacy_filter=PrivacyFilter(
                redact_card_numbers=settings.privacy_redact_card_numbers,
                redact_ssn=settings.privacy_redact_ssn,
                redact_api_keys=settings.privacy_redact_api_keys,
                redact_emails=settings.privacy_redact_emails,
            ),
        )

        self._verifier = MetadataVerifier()

        # Screen Recorder (Stage 1)
        self._recorder = ScreenRecorder(
            db=self._db,
            file_manager=self._file_manager,
            output_queue=self._capture_queue,
            monitor_index=settings.monitor_index,
            interval_seconds=settings.capture_interval_seconds,
            change_threshold=settings.change_threshold,
            max_capture_interval=settings.max_capture_interval_seconds,
            idle_threshold_seconds=settings.idle_threshold_seconds,
        )

        # Worker threads & control flags
        self._stop_event = threading.Event()
        self._workers: list[threading.Thread] = []
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Lifecycle Controls
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the screen recorder and background processing workers."""
        with self._lock:
            if self._workers and any(w.is_alive() for w in self._workers):
                logger.warning("PipelineCoordinator already running.")
                return

            self._stop_event.clear()

            # Start Stage 1 Screen Recorder
            logger.info("[CAPTURE] Starting ScreenRecorder...")
            self._recorder.start()

            # Start Stage 2 (OCR Worker) & Stage 3 (Processing/Extraction Worker)
            ocr_worker = threading.Thread(
                target=self._ocr_worker_loop,
                daemon=True,
                name="Pipeline-OCR-Worker",
            )
            extraction_worker = threading.Thread(
                target=self._extraction_worker_loop,
                daemon=True,
                name="Pipeline-Extraction-Worker",
            )

            self._workers = [ocr_worker, extraction_worker]
            for w in self._workers:
                w.start()

            logger.info("PipelineCoordinator: all background workers active.")
            if self._on_status:
                self._on_status("PIPELINE", "All pipeline stages running")

    def stop(self) -> None:
        """Stop all pipeline workers gracefully."""
        with self._lock:
            logger.info("PipelineCoordinator stopping...")
            self._stop_event.set()

            # Stop recorder first
            self._recorder.stop()

            # Join worker threads
            for w in self._workers:
                if w.is_alive():
                    w.join(timeout=5)

            self._workers.clear()
            logger.info("PipelineCoordinator stopped cleanly.")
            if self._on_status:
                self._on_status("PIPELINE", "Pipeline stopped")

    def pause(self) -> None:
        """Pause capture without stopping downstream processing."""
        self._recorder.pause()
        if self._on_status:
            self._on_status("PIPELINE", "Pipeline paused")

    def resume(self) -> None:
        """Resume capture."""
        self._recorder.resume()
        if self._on_status:
            self._on_status("PIPELINE", "Pipeline resumed")

    @property
    def is_running(self) -> bool:
        return self._recorder.is_running

    @property
    def is_paused(self) -> bool:
        return self._recorder.is_paused

    @property
    def recorder(self) -> ScreenRecorder:
        return self._recorder

    # ------------------------------------------------------------------
    # Worker 1: Stage 2 OCR Loop
    # ------------------------------------------------------------------

    def _ocr_worker_loop(self) -> None:
        """Pulls CaptureRecords from queue (or polls DB) and executes OCR."""
        logger.info("[OCR] OCR worker thread started.")
        while not self._stop_event.is_set():
            record: Optional[CaptureRecord] = None
            try:
                record = self._capture_queue.get(timeout=1.0)
            except queue.Empty:
                pass

            if record is not None:
                self._process_single_capture_ocr(record)
                self._capture_queue.task_done()
            else:
                # Poll DB for any historical pending OCR frames
                self._poll_pending_ocr_frames()

    def _process_single_capture_ocr(self, record: CaptureRecord) -> Optional[RawTextRecord]:
        """Runs OCR for one frame with error isolation."""
        try:
            logger.info("[OCR] Processing frame %s (app: %s)", record.id, record.application)
            raw_rec = self._ocr_pipeline.process(record)
            if self._on_status:
                self._on_status("OCR", f"Processed frame {record.id}: {raw_rec.char_count} chars")
            return raw_rec
        except Exception as exc:  # noqa: BLE001
            logger.error("[OCR] Unexpected OCR error for frame %s: %s", record.id, exc, exc_info=True)
            return None

    def _poll_pending_ocr_frames(self) -> None:
        """Process un-processed frames in SQLite."""
        try:
            pending_rows = self._db.get_pending_ocr_frames(limit=3)
            for row in pending_rows:
                if self._stop_event.is_set():
                    break
                rec = CaptureRecord(
                    id=row["id"],
                    timestamp=datetime.fromisoformat(row["captured_at"]),
                    image_path=row["image_path"],
                    application=row["application"],
                    window_title=row["window_title"],
                    monitor=row["monitor"],
                    width=row["width"] or 1920,
                    height=row["height"] or 1080,
                    capture_reason=row["capture_reason"],
                    diff_score=row["diff_score"] or 0.0,
                )
                self._process_single_capture_ocr(rec)
        except Exception as exc:  # noqa: BLE001
            logger.debug("[OCR] Pending frame polling skipped: %s", exc)

    # ------------------------------------------------------------------
    # Worker 2: Stage 2.5 Text Processing -> Stage 3/3.5 LLM & Verification
    # ------------------------------------------------------------------

    def _extraction_worker_loop(self) -> None:
        """Consumes OCR results, merges text, extracts metadata, verifies facts, and saves."""
        logger.info("[TEXT_PROC/LLM] Extraction & verification worker started.")
        batch: list[RawTextRecord] = []
        last_flush = time.monotonic()

        while not self._stop_event.is_set():
            try:
                raw_record = self._ocr_queue.get(timeout=1.0)
                if not raw_record.is_empty and raw_record.raw_text.strip():
                    batch.append(raw_record)
                self._ocr_queue.task_done()
            except queue.Empty:
                pass

            # Flush batch if batch size >= 3 or time elapsed >= 10 seconds
            time_since_flush = time.monotonic() - last_flush
            if batch and (len(batch) >= 3 or time_since_flush >= 10.0):
                self._process_text_batch(batch)
                batch = []
                last_flush = time.monotonic()

            # Poll for any pending LLM frames in DB
            if not batch and time_since_flush >= 15.0:
                self._poll_pending_llm_frames()
                last_flush = time.monotonic()

        # Final flush on shutdown
        if batch:
            self._process_text_batch(batch)

    def _process_text_batch(self, batch: list[RawTextRecord]) -> None:
        """Runs TextProcessor -> MetadataExtractor -> MetadataVerifier -> SQLite."""
        if not batch:
            return

        frame_ids = [r.frame_id for r in batch if r.frame_id]
        logger.info("[TEXT_PROC] Cleaning & deduplicating batch of %d frames: %s", len(batch), frame_ids)

        try:
            # ── Stage 2.5: Text Cleaning, UI Filtering, Deduplication & Merging ──
            merged_rec: MergedTextRecord = self._text_processor.process(batch)
            try:
                merged_id = self._db.insert_merged_text_record(merged_rec)
                merged_rec.id = merged_id
                logger.info(
                    "[TEXT_PROC] Saved MergedTextRecord #%s (deduped=%s, chars=%d)",
                    merged_id, merged_rec.is_deduplicated, merged_rec.char_count,
                )
            except Exception as db_exc:  # noqa: BLE001
                logger.warning("[TEXT_PROC] DB error saving merged record: %s", db_exc)

            # ── Stage 3: Metadata Extraction (LLM or Dynamic Heuristic) ──
            if not self._metadata_extractor:
                logger.debug("[LLM_EXTRACT] Metadata extractor not configured. Skipping extraction.")
                return

            logger.info("[LLM_EXTRACT] Requesting metadata extraction for %d contributing frames...", len(frame_ids))
            metadata: Optional[StructuredMetadata] = self._metadata_extractor.extract(merged_rec)

            if not metadata:
                logger.warning("[LLM_EXTRACT] Extraction produced no structured metadata for frames: %s", frame_ids)
                return

            # ── Stage 3.5: Fact Verification ──
            raw_text_map: dict[int, str] = {r.frame_id: r.raw_text for r in batch if r.frame_id}
            timestamps_map: dict[int, str] = {
                r.frame_id: r.timestamp.isoformat() for r in batch if r.frame_id
            }

            logger.info("[VERIFY] Verifying extracted facts against original raw text...")
            verified_metadata, evidences = self._verifier.verify(
                metadata=metadata,
                raw_text_map=raw_text_map,
                timestamps_map=timestamps_map,
            )

            # ── Stage 4: SQLite Persistence ──
            self._db.save_metadata(verified_metadata)
            logger.info(
                "[STORAGE] Saved entities (meetings=%d, files=%d, appts=%d, people=%d, projects=%d)",
                len(verified_metadata.meetings),
                len(verified_metadata.files),
                len(verified_metadata.appointments),
                len(verified_metadata.people),
                len(verified_metadata.projects),
            )

            for ev in evidences:
                self._db.save_fact_evidence(ev)

            # Mark all contributing frames as LLM processed
            for fid in frame_ids:
                self._db.mark_frame_llm_done(fid)

            logger.info("[STORAGE] All %d frames marked llm_processed=1.", len(frame_ids))

            if self._on_status:
                self._on_status("EXTRACTION", f"Extracted & verified metadata from {len(frame_ids)} frames")

        except Exception as exc:  # noqa: BLE001
            logger.error("[PIPELINE] Error in extraction/verification stage: %s", exc, exc_info=True)

    def _poll_pending_llm_frames(self) -> None:
        """Pulls unprocessed OCR frames from DB and runs extraction."""
        if not self._metadata_extractor or not self._llm_client:
            return

        try:
            pending_rows = self._db.get_pending_llm_frames(limit=5)
            if not pending_rows:
                return

            batch: list[RawTextRecord] = []
            for row in pending_rows:
                raw_rec = RawTextRecord(
                    frame_id=row["id"],
                    timestamp=datetime.fromisoformat(row["captured_at"]),
                    image_path=row["image_path"],
                    application=row["application"],
                    window_title=row["window_title"],
                    raw_text=row["raw_text"] or "",
                    char_count=len(row["raw_text"] or ""),
                    is_empty=not bool((row["raw_text"] or "").strip()),
                )
                batch.append(raw_rec)

            if batch:
                logger.info("[LLM_EXTRACT] Processing %d pending frames from DB.", len(batch))
                self._process_text_batch(batch)
        except Exception as exc:  # noqa: BLE001
            logger.debug("[LLM_EXTRACT] Pending LLM polling skipped: %s", exc)

    # ------------------------------------------------------------------
    # Manual / Standalone Processing API
    # ------------------------------------------------------------------

    def process_pending(self) -> int:
        """Synchronously process all pending OCR and LLM frames in the database."""
        self._poll_pending_ocr_frames()
        self._poll_pending_llm_frames()
        return 0
