"""
main.py

Entry point for the RIOM Ambient Screen Understanding system.

Usage:
------
    # Run the desktop application with dashboard & full background pipeline:
    python ambient_screen/main.py

    # Run the simulated work session and launch the dashboard:
    python ambient_screen/main.py --simulate

    # Run batch processing for all pending OCR and LLM frames in SQLite:
    python ambient_screen/main.py --process-pending
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure the ambient_screen package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.settings import settings
from storage.db import Database
from storage.file_manager import FileManager


def main() -> None:
    parser = argparse.ArgumentParser(description="RIOM Ambient Screen Understanding System")
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Run a simulated working session to populate the pipeline and open the dashboard.",
    )
    parser.add_argument(
        "--process-pending",
        action="store_true",
        help="Batch process all pending historical OCR and LLM frames in the database.",
    )
    args = parser.parse_args()

    # Ensure storage paths exist
    settings.data_dir.mkdir(parents=True, exist_ok=True)

    if args.process_pending:
        from processing.pipeline_coordinator import PipelineCoordinator
        db = Database(db_path=settings.db_path)
        fm = FileManager(data_dir=settings.data_dir, webp_quality=settings.webp_quality)
        coord = PipelineCoordinator(db=db, file_manager=fm, data_dir=settings.data_dir)
        coord.process_pending()
        print("Batch processing of pending frames complete.")
        return

    if args.simulate:
        from capture.simulation import run_simulation
        db = Database(db_path=settings.db_path)
        fm = FileManager(data_dir=settings.data_dir, webp_quality=settings.webp_quality)
        run_simulation(db=db, file_manager=fm, data_dir=settings.data_dir)

    # Launch PySide6 UI
    from ui.main_window import run_app
    run_app()


if __name__ == "__main__":
    main()
