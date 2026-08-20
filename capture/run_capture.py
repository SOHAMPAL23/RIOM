"""
capture/run_capture.py

Standalone CLI entry point for Stage 1: continuous screen capture.

Usage
-----
    # From the ambient_screen/ directory:
    python -m capture.run_capture

    # Or via the project root:
    python ambient_screen/capture/run_capture.py

    # With custom settings:
    AMBIENT_CAPTURE_INTERVAL_SECONDS=10 python -m capture.run_capture
    AMBIENT_CHANGE_THRESHOLD=0.05 python -m capture.run_capture

The script runs until interrupted with Ctrl-C.
It prints a live summary line every 30 seconds so you can confirm it
is working without checking the log file.

Exit codes
----------
0  Clean exit (Ctrl-C).
1  Fatal startup error (e.g. missing data directory permissions).
"""
from __future__ import annotations

import signal
import sys
import threading
import time
from pathlib import Path

# ── Make the package importable when run as a script ────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from capture.logging_setup import configure_logging
from capture.screen_recorder import ScreenRecorder
from config.settings import settings
from storage.db import Database
from storage.file_manager import FileManager


def main() -> None:
    # ── Logging ──────────────────────────────────────────────────────────────
    configure_logging(
        log_level=settings.log_level,
        log_file=settings.log_file,
    )

    import logging
    logger = logging.getLogger("run_capture")

    # ── Bootstrap storage ────────────────────────────────────────────────────
    settings.data_dir.mkdir(parents=True, exist_ok=True)

    try:
        db = Database(db_path=settings.db_path)
        file_manager = FileManager(
            data_dir=settings.data_dir,
            webp_quality=settings.webp_quality,
        )
    except Exception as exc:
        logger.critical("Failed to initialise storage: %s", exc)
        sys.exit(1)

    # ── Capture recorder ─────────────────────────────────────────────────────
    recorder = ScreenRecorder(
        db=db,
        file_manager=file_manager,
        monitor_index=settings.monitor_index,
        interval_seconds=settings.capture_interval_seconds,
        change_threshold=settings.change_threshold,
    )

    # ── Graceful shutdown on Ctrl-C / SIGTERM ───────────────────────────────
    stop_flag = threading.Event()

    def _shutdown(signum, frame) -> None:  # noqa: ANN001
        logger.info("Shutdown signal received (%s). Stopping…", signum)
        stop_flag.set()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # ── Start ─────────────────────────────────────────────────────────────────
    logger.info(
        "Starting capture — data_dir=%s  interval=%.1fs  monitor=%d",
        settings.data_dir,
        settings.capture_interval_seconds,
        settings.monitor_index,
    )
    recorder.start()

    print(
        f"\n  Ambient Screen Capture running.\n"
        f"  Data directory : {settings.data_dir}\n"
        f"  Log file       : {settings.log_file}\n"
        f"  Interval       : {settings.capture_interval_seconds}s\n"
        f"  Press Ctrl-C to stop.\n"
    )

    # ── Status loop ───────────────────────────────────────────────────────────
    STATUS_INTERVAL = 30  # seconds
    try:
        while not stop_flag.is_set():
            stop_flag.wait(timeout=STATUS_INTERVAL)
            if not stop_flag.is_set():
                disk_mb = file_manager.disk_usage_bytes() / (1024 * 1024)
                logger.info(
                    "Status — saved=%d  skipped=%d  errors=%d  disk=%.1f MB",
                    recorder.frames_captured,
                    recorder.frames_skipped,
                    recorder.frames_errored,
                    disk_mb,
                )
    finally:
        recorder.stop()
        db.close()
        print(
            f"\n  Session complete.\n"
            f"  Frames saved   : {recorder.frames_captured}\n"
            f"  Frames skipped : {recorder.frames_skipped}\n"
            f"  Frames errored : {recorder.frames_errored}\n"
        )


if __name__ == "__main__":
    main()
