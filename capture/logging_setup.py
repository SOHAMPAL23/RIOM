"""
capture/logging_setup.py

Configures the application-wide Python logging system.

Outputs:
- Console (stderr): human-readable, coloured if colorlog is available.
- Rotating file (ambient.log): JSON-free, plain text, 10 MB per file,
  5 files retained — sufficient for a full working day.

Call configure_logging() once at application startup (main.py or
run_capture.py).  All modules use standard logging.getLogger(__name__)
and automatically inherit this configuration.
"""
from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path


_FMT = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"


def configure_logging(log_level: str = "INFO", log_file: Path | None = None) -> None:
    """
    Set up console + optional rotating-file logging.

    Args:
        log_level: "DEBUG", "INFO", "WARNING", or "ERROR".
        log_file:  Path to the log file.  If None, only console output.
    """
    level = getattr(logging, log_level.upper(), logging.INFO)

    handlers: list[logging.Handler] = []

    # ── Console handler ──────────────────────────────────────────────
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(level)
    console.setFormatter(_build_formatter(use_colour=sys.stderr.isatty()))
    handlers.append(console)

    # ── Rotating file handler ────────────────────────────────────────
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            filename=str(log_file),
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)  # Always log everything to file
        file_handler.setFormatter(logging.Formatter(_FMT, datefmt=_DATE_FMT))
        handlers.append(file_handler)

    # ── Root logger ──────────────────────────────────────────────────
    logging.basicConfig(level=level, handlers=handlers, force=True)

    # Silence noisy third-party loggers
    for noisy in ("urllib3", "PIL", "matplotlib", "paddle", "ppocr"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logging.getLogger(__name__).debug("Logging configured: level=%s  file=%s", log_level, log_file)


def _build_formatter(use_colour: bool) -> logging.Formatter:
    """Return a coloured formatter if colorlog is available, else plain."""
    if use_colour:
        try:
            import colorlog  # type: ignore
            return colorlog.ColoredFormatter(
                "%(log_color)s" + _FMT,
                datefmt=_DATE_FMT,
                log_colors={
                    "DEBUG":    "cyan",
                    "INFO":     "green",
                    "WARNING":  "yellow",
                    "ERROR":    "red",
                    "CRITICAL": "bold_red",
                },
            )
        except ImportError:
            pass
    return logging.Formatter(_FMT, datefmt=_DATE_FMT)
