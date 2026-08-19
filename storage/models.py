"""
models.py

Defines the SQLite database schema using plain SQL DDL.
We use Python's built-in sqlite3 module (no ORM) to keep dependencies
minimal and the schema transparent.

Tables:
    frames          — One row per accepted keyframe.
    ocr_blocks      — Individual text blocks from OCR (with bounding boxes).
    screen_contexts — LLM-extracted top-level summary per frame.
    entities        — Denormalised entity table for all extracted entities.
                      Each row stores the entity type, JSON payload, and frame_id.

Design decisions:
- Storing entities as JSON blobs in a single `entities` table avoids
  schema churn as entity types evolve. Full-text search on `payload`
  is feasible with SQLite's JSON functions.
- `frames.image_path` stores relative paths (relative to the data dir)
  so the database is portable between machines.
- All timestamps are stored as ISO-8601 UTC strings.
"""

SCHEMA_SQL = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- -----------------------------------------------------------------------
-- frames: one row per saved keyframe
-- -----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS frames (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    captured_at     TEXT    NOT NULL,           -- ISO-8601 UTC
    image_path      TEXT    NOT NULL,           -- Relative path to WebP image
    image_hash      TEXT,                       -- perceptual hash for dedup
    width           INTEGER,
    height          INTEGER,
    application     TEXT,                       -- Foreground process name at capture time
    window_title    TEXT,                       -- Foreground window title at capture time
    monitor         INTEGER NOT NULL DEFAULT 1, -- MSS monitor index (1 = primary)
    capture_reason  TEXT    NOT NULL DEFAULT 'visual_change', -- CaptureReason value
    diff_score      REAL    NOT NULL DEFAULT 0.0,             -- Normalised pixel diff [0-1]
    raw_text        TEXT,                       -- Full OCR text dump
    ocr_processed   INTEGER NOT NULL DEFAULT 0, -- 0 = pending, 1 = done
    llm_processed   INTEGER NOT NULL DEFAULT 0  -- 0 = pending, 1 = done
);

-- -----------------------------------------------------------------------
-- ocr_blocks: individual text regions from PaddleOCR
-- -----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS ocr_blocks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    frame_id    INTEGER NOT NULL REFERENCES frames(id) ON DELETE CASCADE,
    text        TEXT    NOT NULL,
    confidence  REAL    NOT NULL,
    x1          INTEGER, y1 INTEGER,
    x2          INTEGER, y2 INTEGER
);

-- -----------------------------------------------------------------------
-- raw_text_records: Stage 2 OCR output — one row per processed frame
-- -----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS raw_text_records (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    frame_id     INTEGER NOT NULL REFERENCES frames(id) ON DELETE CASCADE,
    timestamp    TEXT    NOT NULL,           -- ISO-8601 UTC (copied from frame)
    image_path   TEXT    NOT NULL,           -- Relative path (provenance)
    application  TEXT,                       -- Foreground app at capture time
    window_title TEXT,                       -- Foreground window title
    raw_text     TEXT    NOT NULL DEFAULT '', -- Normalised OCR text
    confidence   REAL,                       -- Mean block confidence [0-1]; NULL if unknown
    ocr_engine   TEXT    NOT NULL DEFAULT 'unknown',
    blocks_json  TEXT    NOT NULL DEFAULT '[]', -- JSON array of TextBlock dicts
    char_count   INTEGER NOT NULL DEFAULT 0,
    is_empty     INTEGER NOT NULL DEFAULT 1,  -- 1 = no text detected
    ocr_error    TEXT,                        -- Non-NULL on OCR failure
    created_at   TEXT    NOT NULL             -- ISO-8601 UTC
);

-- -----------------------------------------------------------------------
-- merged_text_records: Stage 2.5 — deduplicated, merged frame text
-- -----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS merged_text_records (
    id                       INTEGER PRIMARY KEY AUTOINCREMENT,
    contributing_frame_ids   TEXT    NOT NULL,   -- JSON array of frame IDs
    contributing_timestamps  TEXT    NOT NULL,   -- JSON array of ISO timestamps
    contributing_image_paths TEXT    NOT NULL,   -- JSON array of relative paths
    first_timestamp          TEXT    NOT NULL,   -- ISO-8601 UTC
    last_timestamp           TEXT    NOT NULL,   -- ISO-8601 UTC
    application              TEXT,
    window_title             TEXT,
    merged_text              TEXT    NOT NULL DEFAULT '',
    char_count               INTEGER NOT NULL DEFAULT 0,
    is_empty                 INTEGER NOT NULL DEFAULT 1,
    is_deduplicated          INTEGER NOT NULL DEFAULT 0,
    frame_count              INTEGER NOT NULL DEFAULT 0,
    similarity_scores        TEXT    NOT NULL DEFAULT '{}', -- JSON dict
    ocr_engines              TEXT    NOT NULL DEFAULT '[]', -- JSON list
    created_at               TEXT    NOT NULL
);

-- -----------------------------------------------------------------------
-- screen_contexts: LLM top-level summary per frame
-- -----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS screen_contexts (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    frame_id         INTEGER NOT NULL REFERENCES frames(id) ON DELETE CASCADE,
    application      TEXT,
    activity_summary TEXT,
    created_at       TEXT NOT NULL               -- ISO-8601 UTC
);

-- -----------------------------------------------------------------------
-- entities: all extracted entities in a single denormalised table
-- -----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS entities (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    frame_id    INTEGER NOT NULL REFERENCES frames(id) ON DELETE CASCADE,
    entity_type TEXT    NOT NULL,  -- 'person', 'meeting', 'url', etc.
    payload     TEXT    NOT NULL,  -- JSON blob of the Pydantic model
    created_at  TEXT    NOT NULL   -- ISO-8601 UTC
);

-- -----------------------------------------------------------------------
-- fact_evidences: Stage 3.5 — output from the MetadataVerifier
-- -----------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS fact_evidences (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    fact_id             TEXT    NOT NULL,
    fact_type           TEXT    NOT NULL,
    fact                TEXT    NOT NULL,           -- JSON representation
    source_frame        INTEGER REFERENCES frames(id) ON DELETE SET NULL,
    source_timestamp    TEXT,                       -- ISO-8601 UTC
    evidence_text       TEXT,
    verification_status TEXT    NOT NULL,           -- verified, partially_supported, unsupported
    created_at          TEXT    NOT NULL            -- ISO-8601 UTC
);

-- Indices for common queries
CREATE INDEX IF NOT EXISTS idx_frames_captured_at        ON frames(captured_at);
CREATE INDEX IF NOT EXISTS idx_frames_application        ON frames(application);
CREATE INDEX IF NOT EXISTS idx_frames_reason             ON frames(capture_reason);
CREATE INDEX IF NOT EXISTS idx_ocr_blocks_frame_id       ON ocr_blocks(frame_id);
CREATE INDEX IF NOT EXISTS idx_raw_text_frame_id         ON raw_text_records(frame_id);
CREATE INDEX IF NOT EXISTS idx_raw_text_application      ON raw_text_records(application);
CREATE INDEX IF NOT EXISTS idx_merged_first_timestamp    ON merged_text_records(first_timestamp);
CREATE INDEX IF NOT EXISTS idx_merged_last_timestamp     ON merged_text_records(last_timestamp);
CREATE INDEX IF NOT EXISTS idx_merged_application        ON merged_text_records(application);
CREATE INDEX IF NOT EXISTS idx_entities_frame_id         ON entities(frame_id);
CREATE INDEX IF NOT EXISTS idx_entities_type             ON entities(entity_type);
CREATE INDEX IF NOT EXISTS idx_fact_evidences_status     ON fact_evidences(verification_status);
"""

