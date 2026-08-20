"""
main.py

Entry point for the RIOM Ambient Screen Understanding system.

Usage:
------
    # Run the desktop application with dashboard & full background pipeline:
    python main.py

    # Run Stage 2 standalone (Screen -> OCR -> Raw Text -> Clean/Dedup):
    python main.py --stage2

    # Run Stage 3 standalone (Raw Text -> Context Grouping -> LLM Metadata -> Verification):
    python main.py --stage3

    # Query the knowledge database:
    python main.py --query "Which meetings did I attend?"
    python main.py --query "Which files did I work on?"

    # Run the simulated work session and launch the dashboard:
    python main.py --simulate

    # Run batch processing for all pending OCR and LLM frames in SQLite:
    python main.py --process-pending
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure the root package is importable
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
        "--stage2",
        action="store_true",
        help="Run Stage 2 standalone: process pending screen captures through OCR, image preprocessing, cleaning, and deduplication.",
    )
    parser.add_argument(
        "--stage3",
        action="store_true",
        help="Run Stage 3 standalone: extract structured metadata from raw text records with deterministic evidence verification.",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Run a query against extracted metadata (e.g. --query 'Which meetings did I attend?').",
    )
    parser.add_argument(
        "--process-pending",
        action="store_true",
        help="Batch process all pending historical OCR and LLM frames in the database.",
    )
    parser.add_argument(
        "--tray",
        "--background",
        dest="tray",
        action="store_true",
        help="Launch application unobtrusively minimized directly in the system tray.",
    )
    parser.add_argument(
        "--video",
        action="store_true",
        help="Enable continuous screen video recording in addition to smart keyframe stills.",
    )
    parser.add_argument(
        "--no-ui",
        action="store_true",
        help="Run headless without launching the Qt desktop UI.",
    )
    args = parser.parse_args()

    # Ensure storage paths exist
    settings.data_dir.mkdir(parents=True, exist_ok=True)

    if args.video:
        settings.enable_video_recording = True

    db = Database(db_path=settings.db_path)
    fm = FileManager(data_dir=settings.data_dir, webp_quality=settings.webp_quality)

    if args.query:
        ans = db.answer_query(args.query)
        print("\n=== RIOM Ambient Screen Knowledge Query ===")
        print(f"Question: {ans.get('query')}")
        print(f"Category: {ans.get('category')} (Found {ans.get('count')} results)")
        print(json.dumps(ans.get("results"), indent=2, default=str))
        return

    if args.stage2:
        from processing.pipeline_coordinator import PipelineCoordinator
        coord = PipelineCoordinator(db=db, file_manager=fm, data_dir=settings.data_dir)
        print("[Stage 2] Running OCR & Raw Text processing on pending frames...")
        coord._poll_pending_ocr_frames()
        print("[Stage 2] Complete.")
        if args.no_ui:
            return

    if args.stage3:
        from processing.pipeline_coordinator import PipelineCoordinator
        coord = PipelineCoordinator(db=db, file_manager=fm, data_dir=settings.data_dir)
        print("[Stage 3] Running Metadata Extraction & Evidence Verification on pending raw text...")
        coord._poll_pending_llm_frames()
        print("[Stage 3] Complete.")
        if args.no_ui:
            return

    if args.process_pending:
        from processing.pipeline_coordinator import PipelineCoordinator
        coord = PipelineCoordinator(db=db, file_manager=fm, data_dir=settings.data_dir)
        coord.process_pending()
        print("Batch processing of pending frames complete.")
        if args.no_ui:
            return

    if args.simulate:
        from capture.simulation import run_simulation
        print("[Simulation] Running simulated work session with smart capture, OCR, and metadata extraction...")
        run_simulation(db=db, file_manager=fm, data_dir=settings.data_dir)
        print("[Simulation] Complete.")
        if args.no_ui:
            return

    if args.no_ui:
        return

    # Launch PySide6 UI
    from ui.main_window import run_app
    run_app(minimized_to_tray=args.tray)


if __name__ == "__main__":
    main()
