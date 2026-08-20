"""
ui/main_window.py

RIOM — Enterprise Work Memory Dashboard
Executive-grade interface for on-device ambient screen intelligence,
unobtrusive background / system tray operation, and knowledge extraction.

Features:
---------
- System Tray Integration: Runs unobtrusively in the background, minimizes to tray,
  context menu for start/stop, snapshot, video mode, and simulation.
- Desktop Vision & Snapshot Linking: Real-time keyframe preview with click-to-open,
  manual snapshot trigger, storage folder browsing, and continuous video mode toggle.
- Extracted Work Memory Knowledge Graph:
  1. Files & Workspaces (file name, path, doc title, app, duration, open actions)
  2. Video Conferences & Calls (Google Meet, Zoom, Teams, discussion points, action items, links)
  3. Planned Appointments & Deadlines (dates, times, deadlines, reminders)
  4. Captured Web Resources (browser URLs, document tabs, clickable hyperlinks)
  5. Collaborators & Projects (interactive entity chips)
  6. Truth Grounding Audit Trail (deterministic screen verification & provenance)
- Real-time OCR Text Stream and chronological Activity Timeline.
"""
from __future__ import annotations

import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from PySide6.QtCore import (
    Qt, QTimer, QThread, Signal, QObject, QUrl,
)
from PySide6.QtGui import (
    QFont, QPixmap, QColor, QPainter, QBrush, QPen, QIcon, QDesktopServices, QAction,
)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QScrollArea, QFrame, QSplitter,
    QSizePolicy, QTextEdit, QSystemTrayIcon, QMenu, QDialog,
)

# ---------------------------------------------------------------------------
# Pipeline imports & safe fallback
# ---------------------------------------------------------------------------
try:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from config.settings import settings
    from storage.db import Database
    from storage.file_manager import FileManager
    from capture.logging_setup import configure_logging
    from processing.pipeline_coordinator import PipelineCoordinator
    from capture.simulation import run_simulation
    from metadata.schemas import Meeting
    from processing.meeting_notes import MeetingNotesGenerator, sanitize_filename
    _PIPELINE_AVAILABLE = True
except Exception:  # noqa: BLE001
    _PIPELINE_AVAILABLE = False
    settings = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# ===========================================================================
# Enterprise Design Tokens & Palette
# ===========================================================================
C_BG            = "#080c14"      # Deep obsidian canvas
C_SURFACE       = "#0f1623"      # Primary structural panel
C_SURFACE_HOVER = "#172133"      # Interactive highlight
C_SURFACE_ELEV  = "#151e2e"      # Elevated container
C_BORDER        = "#1e2b40"      # Subtle separator
C_BORDER_LIGHT  = "#2c3d59"      # Active / focused boundary
C_CYAN          = "#38bdf8"      # Primary accent (electric sky)
C_EMERALD       = "#10b981"      # Success / Verified state
C_EMERALD_BG    = "#042f24"      # Emerald container
C_AMBER         = "#f59e0b"      # Inferred / Warning state
C_AMBER_BG      = "#3b1e06"      # Amber container
C_ROSE          = "#f43f5e"      # Discarded / Error state
C_ROSE_BG       = "#3c0a18"      # Rose container
C_VIOLET        = "#a855f7"      # Simulation / Metadata accent
C_VIOLET_BG     = "#2c0b4a"      # Violet container
C_TEXT          = "#f8fafc"      # Primary high-contrast text
C_TEXT_MUTED    = "#94a3b8"      # Secondary metadata text
C_TEXT_FAINT    = "#475569"      # Subtle placeholder / timestamp text

FONT_MONO = "JetBrains Mono, Consolas, Courier New, monospace"
FONT_UI   = "Segoe UI, -apple-system, BlinkMacSystemFont, Roboto, sans-serif"


# ===========================================================================
# Worker — manages background PipelineCoordinator on dedicated QThread
# ===========================================================================
class PipelineWorker(QObject):
    """Executes PipelineCoordinator lifecycle on a separate background thread."""

    started       = Signal()
    stopped       = Signal()
    error         = Signal(str)
    status_update = Signal(str, str)
    sim_finished  = Signal(int)
    snapshot_done = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self._coordinator: Optional[PipelineCoordinator] = None

    def start_pipeline(self) -> None:
        if not _PIPELINE_AVAILABLE:
            self.error.emit("Pipeline dependencies not available.")
            return
        try:
            configure_logging(log_level=settings.log_level, log_file=settings.log_file)
            settings.data_dir.mkdir(parents=True, exist_ok=True)
            db = Database(db_path=settings.db_path)
            fm = FileManager(data_dir=settings.data_dir, webp_quality=settings.webp_quality)

            def _on_status_cb(stage: str, msg: str) -> None:
                self.status_update.emit(stage, msg)

            self._coordinator = PipelineCoordinator(
                db=db,
                file_manager=fm,
                data_dir=settings.data_dir,
                on_status=_on_status_cb,
            )
            self._coordinator.start()
            self.started.emit()
        except Exception as exc:  # noqa: BLE001
            self.error.emit(str(exc))

    def stop_pipeline(self) -> None:
        if self._coordinator:
            self._coordinator.stop()
            self._coordinator = None
        self.stopped.emit()

    def toggle_pause(self) -> bool:
        if not self._coordinator:
            return False
        if self._coordinator.is_paused:
            self._coordinator.resume()
            return False
        else:
            self._coordinator.pause()
            return True

    def force_snapshot(self) -> None:
        if self._coordinator:
            self._coordinator.force_capture()
            self.snapshot_done.emit("Manual snapshot triggered")

    def toggle_video(self) -> bool:
        if self._coordinator:
            return self._coordinator.toggle_video_recording()
        return False

    def run_demo_simulation(self) -> None:
        try:
            settings.data_dir.mkdir(parents=True, exist_ok=True)
            db = Database(db_path=settings.db_path)
            fm = FileManager(data_dir=settings.data_dir, webp_quality=settings.webp_quality)
            count = run_simulation(db=db, file_manager=fm, data_dir=settings.data_dir)
            self.sim_finished.emit(count)
        except Exception as exc:  # noqa: BLE001
            self.error.emit(f"Simulation error: {exc}")

    @property
    def coordinator(self) -> Optional[PipelineCoordinator]:
        return self._coordinator


# ===========================================================================
# Reusable UI Custom Components & Helpers
# ===========================================================================

def _label(text: str, size: int = 12, color: str = C_TEXT,
           bold: bool = False, mono: bool = False) -> QLabel:
    lbl = QLabel(text)
    font_family = FONT_MONO if mono else FONT_UI
    weight = QFont.Weight.Bold if bold else QFont.Weight.Normal
    lbl.setFont(QFont(font_family, size, weight))
    lbl.setStyleSheet(f"color: {color}; background: transparent; border: none;")
    return lbl


def _card(title: str, subtitle: Optional[str] = None, badge: str = "SYS") -> tuple[QFrame, QVBoxLayout]:
    """Generates an executive-grade rounded card container with structured header."""
    card = QFrame()
    card.setStyleSheet(f"""
        QFrame {{
            background: {C_SURFACE};
            border: 1px solid {C_BORDER};
            border-radius: 10px;
        }}
    """)
    card_layout = QVBoxLayout(card)
    card_layout.setContentsMargins(16, 14, 16, 16)
    card_layout.setSpacing(10)

    # Header Row
    header_w = QWidget()
    header_w.setStyleSheet("background: transparent; border: none;")
    hh = QHBoxLayout(header_w)
    hh.setContentsMargins(0, 0, 0, 0)
    hh.setSpacing(8)

    # Category Tag Badge
    badge_lbl = _label(badge.upper(), size=9, color=C_CYAN, bold=True, mono=True)
    badge_lbl.setStyleSheet(f"""
        background: {C_CYAN}15;
        border: 1px solid {C_CYAN}40;
        border-radius: 3px;
        padding: 2px 6px;
        font-weight: 700;
        letter-spacing: 0.06em;
    """)
    hh.addWidget(badge_lbl)

    title_lbl = _label(title, size=12, color=C_TEXT, bold=True)
    title_lbl.setStyleSheet("letter-spacing: 0.02em; font-weight: 700;")
    hh.addWidget(title_lbl)

    if subtitle:
        sub_lbl = _label(f"|  {subtitle}", size=11, color=C_TEXT_MUTED)
        hh.addWidget(sub_lbl)

    hh.addStretch()
    card_layout.addWidget(header_w)

    content_area = QVBoxLayout()
    content_area.setContentsMargins(0, 2, 0, 0)
    content_area.setSpacing(8)
    card_layout.addLayout(content_area)

    return card, content_area


class StatusPill(QWidget):
    """Sleek minimalist status indicator for live capture state."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFixedHeight(28)
        self._state = "idle"

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 0, 12, 0)
        layout.setSpacing(8)

        self._dot = QLabel("●")
        self._dot.setFont(QFont(FONT_UI, 9, QFont.Weight.Bold))
        self._dot.setStyleSheet(f"color: {C_TEXT_FAINT}; background: transparent; border: none;")

        self._text = _label("IDLE", size=10, color=C_TEXT_MUTED, bold=True, mono=True)
        layout.addWidget(self._dot)
        layout.addWidget(self._text)

        self._set_style(C_TEXT_FAINT, C_SURFACE_ELEV, "STANDBY")

    def set_state(self, state: str, detail: str = "") -> None:
        self._state = state
        if state == "active":
            label = "RECORDING LIVE" if not detail else detail
            self._set_style(C_EMERALD, C_EMERALD_BG, label)
        elif state == "paused":
            self._set_style(C_AMBER, C_AMBER_BG, "PAUSED")
        elif state == "error":
            self._set_style(C_ROSE, C_ROSE_BG, f"ERROR: {detail[:25] if detail else 'FAILED'}")
        else:
            self._set_style(C_TEXT_FAINT, C_SURFACE_ELEV, "STANDBY")

    def _set_style(self, fg: str, bg: str, text: str) -> None:
        self.setStyleSheet(f"""
            QWidget {{
                background: {bg};
                border: 1px solid {fg}40;
                border-radius: 14px;
            }}
        """)
        self._dot.setStyleSheet(f"color: {fg}; background: transparent; border: none;")
        self._text.setText(text.upper())
        self._text.setStyleSheet(f"color: {fg}; background: transparent; border: none; font-weight: 700; letter-spacing: 0.05em;")


class TimelineWidget(QWidget):
    """Chronological timeline of active application switches."""

    DEMO_ENTRIES = [
        ("10:30", "Google Meet — Sprint Planning", "MEET", "#38bdf8"),
        ("11:15", "VS Code — project.py",          "CODE", "#10b981"),
        ("12:00", "Gmail — Client Q3 Sync",        "MAIL", "#f59e0b"),
        ("13:30", "Docs — Project Brief Architecture", "DOCS",  "#a855f7"),
    ]

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 2, 0, 2)
        self._layout.setSpacing(6)
        self._entries = list(self.DEMO_ENTRIES)
        self._render()

    def _render(self) -> None:
        while self._layout.count():
            item = self._layout.takeAt(0)
            if item and item.widget():
                w = item.widget()
                w.setParent(None)
                w.deleteLater()

        if not self._entries:
            empty = _label("No activity recorded yet.", size=11, color=C_TEXT_MUTED)
            self._layout.addWidget(empty)
            return

        for time_str, label, kind, color in self._entries:
            row = QWidget()
            row.setStyleSheet(f"""
                QWidget {{
                    background: {C_SURFACE_ELEV};
                    border: 1px solid {C_BORDER};
                    border-radius: 6px;
                }}
            """)
            h = QHBoxLayout(row)
            h.setContentsMargins(10, 6, 10, 6)
            h.setSpacing(8)

            tag_lbl = _label(kind.upper(), size=9, color=color, bold=True, mono=True)
            tag_lbl.setStyleSheet(f"""
                background: {color}15;
                color: {color};
                border: 1px solid {color}40;
                border-radius: 3px;
                padding: 1px 5px;
                font-weight: 700;
            """)
            h.addWidget(tag_lbl)

            lbl = _label(label, size=11, color=C_TEXT, bold=True)
            lbl.setStyleSheet("background: transparent; border: none;")
            lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
            h.addWidget(lbl)

            t = _label(time_str, size=10, color=C_TEXT_MUTED, mono=True)
            h.addWidget(t)

            self._layout.addWidget(row)

    def update_entries(self, entries: list[tuple[str, str, str, str]]) -> None:
        self._entries = entries
        self._render()


class EntityChip(QLabel):
    """Refined typographic pill badge for extracted people, organizations, projects."""

    _TYPE_CONFIG = {
        "person":       (C_CYAN,    "#062438", "PERSON"),
        "project":      (C_EMERALD, "#042f24", "PROJECT"),
        "organization": (C_AMBER,   "#3b1e06", "ORG"),
        "url_reference":(C_VIOLET,  "#2c0b4a", "URL"),
        "meeting":      (C_CYAN,    "#062438", "MEET"),
        "appointment":  (C_AMBER,   "#3b1e06", "EVENT"),
    }

    def __init__(self, text: str, entity_type: str = "person", tooltip: str = "") -> None:
        fg, bg, tag = self._TYPE_CONFIG.get(entity_type, (C_TEXT_MUTED, C_SURFACE_ELEV, "ENTITY"))
        super().__init__(f"{tag}: {text}")
        if tooltip:
            self.setToolTip(tooltip)
        self.setStyleSheet(f"""
            QLabel {{
                background: {bg};
                color: {fg};
                border: 1px solid {fg}50;
                border-radius: 4px;
                padding: 3px 8px;
                font-size: 10px;
                font-weight: 600;
                font-family: {FONT_MONO};
                letter-spacing: 0.02em;
            }}
        """)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)


class VerificationBadge(QLabel):
    """Audit status badge explaining truth grounding."""

    _STATUS = {
        "verified":             ("VERIFIED ON SCREEN",     C_EMERALD, C_EMERALD_BG, "Verified 100% directly from screen OCR text"),
        "partially_supported":  ("INFERRED CONTEXT",       C_AMBER,   C_AMBER_BG,   "Primary topic verified; contextual detail inferred"),
        "unsupported":          ("REJECTED / UNGROUNDED",  C_ROSE,    C_ROSE_BG,    "Hallucination / ungrounded inference rejected"),
    }

    def __init__(self, status: str) -> None:
        label, fg, bg, tooltip = self._STATUS.get(
            status, ("UNVERIFIED", C_TEXT_MUTED, C_SURFACE_ELEV, "Pending screen text verification")
        )
        super().__init__(label)
        self.setToolTip(tooltip)
        self.setStyleSheet(f"""
            QLabel {{
                background: {bg};
                color: {fg};
                border: 1px solid {fg}50;
                border-radius: 4px;
                padding: 2px 7px;
                font-size: 9px;
                font-weight: 700;
                font-family: {FONT_MONO};
                letter-spacing: 0.05em;
            }}
        """)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)


# ===========================================================================
# Main Application Window (Unified Ambient Intelligence Dashboard)
# ===========================================================================
class MainWindow(QMainWindow):
    """RIOM Work Memory Dashboard — Unified Single-Page Interface with System Tray."""

    _start_requested = Signal()
    _stop_requested  = Signal()
    _demo_requested  = Signal()
    _force_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("RIOM — Enterprise Work Memory Dashboard")
        self.setMinimumSize(1180, 800)
        self.resize(1360, 880)

        # Pipeline state
        self._is_capturing = False
        self._is_paused    = False
        self._video_mode   = False
        self._db: Optional[Database] = None
        self._latest_image_path: Optional[Path] = None

        # Worker thread
        self._worker_thread = QThread()
        self._worker = PipelineWorker()
        self._worker.moveToThread(self._worker_thread)
        self._worker.started.connect(self._on_capture_started)
        self._worker.stopped.connect(self._on_capture_stopped)
        self._worker.error.connect(self._on_capture_error)
        self._worker.status_update.connect(self._on_status_update)
        self._worker.sim_finished.connect(self._on_simulation_finished)
        self._worker.snapshot_done.connect(self._on_snapshot_done)

        self._start_requested.connect(self._worker.start_pipeline)
        self._stop_requested.connect(self._worker.stop_pipeline)
        self._demo_requested.connect(self._worker.run_demo_simulation)
        self._force_requested.connect(self._worker.force_snapshot)
        self._worker_thread.start()

        self._setup_ui()
        self._setup_tray_icon()
        self._apply_styles()

        # Refresh timer — polls DB every 3 s
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setInterval(3000)
        self._refresh_timer.timeout.connect(self._refresh_data)
        self._refresh_timer.start()

        # Blink timer for REC indicator
        self._blink_timer = QTimer(self)
        self._blink_timer.setInterval(800)
        self._blink_timer.timeout.connect(self._blink_rec)
        self._blink_on = True

        # Initial data load if DB exists
        if _PIPELINE_AVAILABLE:
            try:
                self._db = Database(db_path=settings.db_path)
                self._refresh_data()
            except Exception:  # noqa: BLE001
                pass

        # Auto-start continuous live screen capture and OCR on launch
        if _PIPELINE_AVAILABLE:
            QTimer.singleShot(250, self._on_start_clicked)

    # ------------------------------------------------------------------
    # System Tray Icon Support
    # ------------------------------------------------------------------

    def _setup_tray_icon(self) -> None:
        """Configures unobtrusive system tray integration."""
        self._tray_icon = QSystemTrayIcon(self)

        # Generate sleek procedural icon
        pix = QPixmap(32, 32)
        pix.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pix)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(QBrush(QColor(C_SURFACE)))
        painter.setPen(QPen(QColor(C_CYAN), 2))
        painter.drawRoundedRect(2, 2, 28, 28, 6, 6)
        painter.setBrush(QBrush(QColor(C_EMERALD)))
        painter.drawEllipse(12, 12, 8, 8)
        painter.end()

        self._tray_icon.setIcon(QIcon(pix))
        self._tray_icon.setToolTip("RIOM — Ambient Work Memory")

        # Tray Context Menu
        tray_menu = QMenu()
        tray_menu.setStyleSheet(f"""
            QMenu {{
                background: {C_SURFACE};
                color: {C_TEXT};
                border: 1px solid {C_BORDER};
                padding: 6px;
                font-family: {FONT_UI};
                font-size: 11px;
            }}
            QMenu::item {{
                padding: 6px 20px;
                border-radius: 4px;
            }}
            QMenu::item:selected {{
                background: {C_CYAN}25;
                color: {C_CYAN};
            }}
        """)

        act_show = QAction("Open Dashboard", self)
        act_show.triggered.connect(self._show_from_tray)
        tray_menu.addAction(act_show)

        tray_menu.addSeparator()

        self._act_tray_start = QAction("Start Live Capture", self)
        self._act_tray_start.triggered.connect(self._on_start_clicked)
        tray_menu.addAction(self._act_tray_start)

        self._act_tray_pause = QAction("Pause Capture", self)
        self._act_tray_pause.triggered.connect(self._on_pause_clicked)
        tray_menu.addAction(self._act_tray_pause)

        self._act_tray_stop = QAction("Stop Capture", self)
        self._act_tray_stop.triggered.connect(self._on_stop_clicked)
        tray_menu.addAction(self._act_tray_stop)

        tray_menu.addSeparator()

        act_snapshot = QAction("Take Snapshot Now", self)
        act_snapshot.triggered.connect(self._on_snapshot_clicked)
        tray_menu.addAction(act_snapshot)

        act_sim = QAction("Simulate Workday Session", self)
        act_sim.triggered.connect(self._on_demo_clicked)
        tray_menu.addAction(act_sim)

        act_folder = QAction("Open Captures Folder", self)
        act_folder.triggered.connect(self._on_open_storage_clicked)
        tray_menu.addAction(act_folder)

        act_notes = QAction("Open Meeting Notes Folder", self)
        act_notes.triggered.connect(self._on_open_meeting_notes_clicked)
        tray_menu.addAction(act_notes)

        tray_menu.addSeparator()

        act_quit = QAction("Quit RIOM", self)
        act_quit.triggered.connect(self._on_quit_app)
        tray_menu.addAction(act_quit)

        self._tray_icon.setContextMenu(tray_menu)
        self._tray_icon.activated.connect(self._on_tray_activated)
        self._tray_icon.show()

    def _on_tray_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        if reason in (QSystemTrayIcon.ActivationReason.Trigger, QSystemTrayIcon.ActivationReason.DoubleClick):
            self._show_from_tray()

    def _show_from_tray(self) -> None:
        self.showNormal()
        self.activateWindow()
        self.raise_()

    def closeEvent(self, event) -> None:  # noqa: N802
        """Intercept close to run unobtrusively in the tray."""
        if settings and settings.minimize_to_tray and self._tray_icon.isVisible():
            event.ignore()
            self.hide()
            self._tray_icon.showMessage(
                "RIOM Ambient Memory",
                "RIOM is running unobtrusively in the background.",
                QSystemTrayIcon.MessageIcon.Information,
                2000,
            )
        else:
            self._on_quit_app()
            event.accept()

    def _on_quit_app(self) -> None:
        """Clean shutdown of background threads and app exit."""
        logger.info("Exiting RIOM application...")
        self._worker.stop_pipeline()
        self._worker_thread.quit()
        self._worker_thread.wait(3000)
        QApplication.quit()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _setup_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        vbox = QVBoxLayout(root)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(0)

        # Header Bar
        vbox.addWidget(self._build_header())

        # Main 2-Column Splitter Body
        vbox.addWidget(self._build_body(), stretch=1)

    # ── Header ──────────────────────────────────────────────────────────

    def _build_header(self) -> QWidget:
        hdr = QWidget()
        hdr.setObjectName("header")
        hdr.setFixedHeight(64)
        h = QHBoxLayout(hdr)
        h.setContentsMargins(24, 0, 24, 0)
        h.setSpacing(12)

        # Brand Badge
        title = _label("RIOM", size=17, bold=True, color=C_TEXT)
        title.setStyleSheet("letter-spacing: 0.12em; font-weight: 900;")
        tag = _label("AMBIENT WORK MEMORY", size=9, color=C_CYAN, bold=True, mono=True)
        tag.setStyleSheet(f"background: {C_CYAN}15; border: 1px solid {C_CYAN}40; border-radius: 4px; padding: 2px 7px; letter-spacing: 0.08em;")

        h.addWidget(title)
        h.addWidget(tag)
        h.addStretch()

        # Status Pill Indicator
        self._status_pill = StatusPill()
        h.addWidget(self._status_pill)

        sep = QLabel("|")
        sep.setStyleSheet(f"color: {C_BORDER_LIGHT}; font-size: 13px; background: transparent; border: none;")
        h.addWidget(sep)

        # Action Buttons
        self._btn_snapshot = self._ctrl_button("Take Snapshot", C_CYAN, width=115)
        self._btn_snapshot.setToolTip("Force an immediate screen capture snapshot")
        self._btn_snapshot.clicked.connect(self._on_snapshot_clicked)
        h.addWidget(self._btn_snapshot)

        self._btn_video_toggle = self._ctrl_button("Video: OFF", C_TEXT_MUTED, width=100)
        self._btn_video_toggle.setToolTip("Toggle continuous screen video recording option")
        self._btn_video_toggle.clicked.connect(self._on_toggle_video_clicked)
        h.addWidget(self._btn_video_toggle)

        self._btn_demo = self._ctrl_button("Simulate Session", C_VIOLET, width=130)
        self._btn_demo.setToolTip("Load multi-app workday session (Conferences, Codebases, Documents)")
        self._btn_demo.clicked.connect(self._on_demo_clicked)
        h.addWidget(self._btn_demo)

        self._btn_clear = self._ctrl_button("Reset Memory", C_TEXT_MUTED, width=105)
        self._btn_clear.setToolTip("Clear session memory and cached records")
        self._btn_clear.clicked.connect(self._on_clear_clicked)
        h.addWidget(self._btn_clear)

        self._btn_start = self._ctrl_button("Start Capture", C_EMERALD, width=105)
        self._btn_pause = self._ctrl_button("Pause", C_AMBER, width=80)
        self._btn_stop  = self._ctrl_button("Stop", C_ROSE, width=75)
        self._btn_pause.setEnabled(False)
        self._btn_stop.setEnabled(False)
        self._btn_start.clicked.connect(self._on_start_clicked)
        self._btn_pause.clicked.connect(self._on_pause_clicked)
        self._btn_stop.clicked.connect(self._on_stop_clicked)

        for btn in (self._btn_start, self._btn_pause, self._btn_stop):
            h.addWidget(btn)

        return hdr

    def _ctrl_button(self, text: str, accent: str, width: int = 90) -> QPushButton:
        btn = QPushButton(text)
        btn.setFixedHeight(32)
        btn.setFixedWidth(width)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setStyleSheet(f"""
            QPushButton {{
                background: {C_SURFACE_ELEV};
                color: {accent};
                border: 1px solid {accent}40;
                border-radius: 6px;
                font-size: 11px;
                font-weight: 600;
                font-family: {FONT_UI};
                padding: 0 8px;
            }}
            QPushButton:hover {{
                background: {accent}20;
                border-color: {accent};
                color: {C_TEXT};
            }}
            QPushButton:pressed {{
                background: {accent}35;
            }}
            QPushButton:disabled {{
                color: {C_TEXT_FAINT};
                border-color: {C_BORDER};
                background: {C_SURFACE};
            }}
        """)
        return btn

    # ── Body: Two-Column Splitter ────────────────────────────────────────

    def _build_body(self) -> QWidget:
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(2)
        splitter.setStyleSheet(f"QSplitter::handle {{ background: {C_BORDER}; }}")

        # Left column — Vision & Activity Stream
        left = self._build_left_panel()
        splitter.addWidget(left)

        # Right column — Extracted Work Memory
        right = self._build_right_panel()
        splitter.addWidget(right)

        splitter.setSizes([540, 680])
        return splitter

    # ── Left Column: Vision & Activity Stream ────────────────────────────

    def _build_left_panel(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        container = QWidget()
        vbox = QVBoxLayout(container)
        vbox.setContentsMargins(20, 16, 12, 20)
        vbox.setSpacing(14)

        # 1. Desktop Vision Preview Card
        shot_card, shot_l = _card("Desktop Vision", "Active Display Monitor", "VISION")

        self._screenshot_label = QLabel()
        self._screenshot_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._screenshot_label.setFixedHeight(180)
        self._screenshot_label.setCursor(Qt.CursorShape.PointingHandCursor)
        self._screenshot_label.setToolTip("Click to view full-resolution snapshot or open in viewer")
        self._screenshot_label.setStyleSheet(f"""
            background: {C_SURFACE_ELEV};
            border: 1px solid {C_BORDER};
            border-radius: 6px;
            color: {C_TEXT_FAINT};
            font-size: 11px;
            font-family: {FONT_MONO};
        """)
        self._screenshot_label.setText("MONITORING STANDBY — NO ACTIVE SCREEN CAPTURE")
        self._screenshot_label.mousePressEvent = lambda e: self._on_snapshot_clicked()
        shot_l.addWidget(self._screenshot_label)

        # Meta & Quick Actions Bar
        meta_bar = QWidget()
        meta_bar.setStyleSheet(f"background: {C_SURFACE_ELEV}; border-radius: 4px; padding: 2px;")
        mh = QHBoxLayout(meta_bar)
        mh.setContentsMargins(8, 4, 8, 4)
        mh.setSpacing(6)

        self._shot_meta = _label("Active Window: —  |  Standing by...", size=10, color=C_TEXT_MUTED, mono=True)
        mh.addWidget(self._shot_meta)
        mh.addStretch()

        btn_open_folder = QPushButton("Open Folder")
        btn_open_folder.setFixedHeight(22)
        btn_open_folder.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_open_folder.setStyleSheet(f"""
            QPushButton {{
                background: {C_SURFACE};
                color: {C_CYAN};
                border: 1px solid {C_CYAN}40;
                border-radius: 3px;
                font-size: 9px;
                font-family: {FONT_MONO};
                padding: 0 6px;
            }}
            QPushButton:hover {{
                background: {C_CYAN}20;
            }}
        """)
        btn_open_folder.clicked.connect(self._on_open_storage_clicked)
        mh.addWidget(btn_open_folder)

        shot_l.addWidget(meta_bar)
        vbox.addWidget(shot_card)

        # 2. Live OCR Stream
        text_card, text_l = _card("OCR Text Stream", "On-Device Neural Recognition", "OCR")
        self._text_box = QTextEdit()
        self._text_box.setReadOnly(True)
        self._text_box.setFixedHeight(105)
        self._text_box.setPlaceholderText("Waiting for on-device OCR stream...")
        self._text_box.setStyleSheet(f"""
            QTextEdit {{
                background: {C_SURFACE_ELEV};
                color: {C_TEXT};
                border: 1px solid {C_BORDER};
                border-radius: 6px;
                padding: 8px;
                font-family: {FONT_MONO};
                font-size: 11px;
                line-height: 1.4;
            }}
        """)
        text_l.addWidget(self._text_box)
        vbox.addWidget(text_card)

        # 3. Workday Activity Timeline
        tl_card, tl_l = _card("Activity Timeline", "Application Context Switches", "TIMELINE")
        self._timeline = TimelineWidget()
        tl_l.addWidget(self._timeline)
        vbox.addWidget(tl_card)

        vbox.addStretch()
        scroll.setWidget(container)
        return scroll

    # ── Right Column: Extracted Work Memory ──────────────────────────────

    def _build_right_panel(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        container = QWidget()
        vbox = QVBoxLayout(container)
        vbox.setContentsMargins(12, 16, 20, 20)
        vbox.setSpacing(14)

        # 1. Active Files & Documents
        file_card, file_l = _card("Files & Workspaces", "Source Code & Workspace Documents", "FILES")
        self._files_widget = self._scrollable_list_widget("No workspace files detected yet.")
        file_l.addWidget(self._files_widget)
        vbox.addWidget(file_card)

        # 2. Detected Meetings & Calls
        meet_card, meet_l = _card("Conferences & Scheduled Meetings", "Google Meet, Zoom, Microsoft Teams", "MEET")
        self._meetings_widget = self._scrollable_list_widget("No active conferences or calls detected.")
        meet_l.addWidget(self._meetings_widget)
        vbox.addWidget(meet_card)

        # 3. Planned Appointments & Deadlines
        appt_card, appt_l = _card("Planned Appointments & Deadlines", "Dates, Deadlines, Reminders & Scheduled Events", "APPT")
        self._appointments_widget = self._scrollable_list_widget("No scheduled appointments or deadlines detected yet.")
        appt_l.addWidget(self._appointments_widget)
        vbox.addWidget(appt_card)

        # 4. Captured Links & Resources
        links_card, links_l = _card("Captured Web Resources", "Browser URLs, Cloud Documents & Issue Trackers", "LINKS")
        self._links_widget = self._scrollable_list_widget("No web links captured yet.")
        links_l.addWidget(self._links_widget)
        vbox.addWidget(links_card)

        # 5. People, Projects & Context
        ent_card, ent_l = _card("Collaborators & Projects", "Recognized Entities & Contextual Nodes", "GRAPH")
        self._entities_container_layout = ent_l
        self._entities_flow = QWidget()
        self._entities_flow.setStyleSheet("background: transparent; border: none;")
        self._entities_flow_inner = QVBoxLayout(self._entities_flow)
        self._entities_flow_inner.setContentsMargins(0, 0, 0, 0)
        self._entities_flow_inner.setSpacing(6)
        self._entities_flow_inner.addWidget(
            _label("No collaborators or projects recognized yet.", size=11, color=C_TEXT_FAINT)
        )
        ent_l.addWidget(self._entities_flow)
        vbox.addWidget(ent_card)

        # 6. Truth Grounding Audit Trail
        ver_card, ver_l = _card("Truth Grounding Audit", "Deterministic Screen Verification & Provenance", "AUDIT")
        self._verification_widget = self._scrollable_list_widget("No verified facts recorded yet.")
        ver_l.addWidget(self._verification_widget)
        vbox.addWidget(ver_card)

        vbox.addStretch()
        scroll.setWidget(container)
        return scroll

    def _scrollable_list_widget(self, placeholder_text: str = "No items recorded yet.") -> QWidget:
        w = QWidget()
        w.setStyleSheet("background: transparent; border: none;")
        w._layout = QVBoxLayout(w)  # type: ignore[attr-defined]
        w._layout.setContentsMargins(0, 0, 0, 0)
        w._layout.setSpacing(6)
        placeholder = _label(placeholder_text, size=11, color=C_TEXT_FAINT)
        w._layout.addWidget(placeholder)
        w._placeholder = placeholder  # type: ignore[attr-defined]
        return w

    # ------------------------------------------------------------------
    # Global Styling
    # ------------------------------------------------------------------

    def _apply_styles(self) -> None:
        self.setStyleSheet(f"""
            QMainWindow, QWidget {{
                background: {C_BG};
                color: {C_TEXT};
                font-family: {FONT_UI};
            }}

            #header {{
                background: {C_SURFACE};
                border-bottom: 1px solid {C_BORDER};
            }}

            QScrollArea {{
                background: {C_BG};
                border: none;
            }}

            QScrollBar:vertical {{
                background: {C_SURFACE};
                width: 6px;
                border-radius: 3px;
            }}
            QScrollBar::handle:vertical {{
                background: {C_BORDER_LIGHT};
                border-radius: 3px;
                min-height: 24px;
            }}
            QScrollBar::handle:vertical:hover {{
                background: {C_CYAN};
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}

            QToolTip {{
                background: {C_SURFACE_ELEV};
                color: {C_TEXT};
                border: 1px solid {C_BORDER_LIGHT};
                border-radius: 4px;
                padding: 6px 10px;
                font-family: {FONT_UI};
                font-size: 11px;
            }}
        """)

    # ------------------------------------------------------------------
    # Button handlers
    # ------------------------------------------------------------------

    def _on_start_clicked(self) -> None:
        self._btn_start.setEnabled(False)
        self._btn_pause.setEnabled(True)
        self._btn_stop.setEnabled(True)
        self._status_pill.set_state("active", "STARTING PIPELINE...")
        self._start_requested.emit()

    def _on_pause_clicked(self) -> None:
        now_paused = self._worker.toggle_pause()
        if now_paused:
            self._btn_pause.setText("Resume")
            self._status_pill.set_state("paused")
            self._blink_timer.stop()
        else:
            self._btn_pause.setText("Pause")
            self._status_pill.set_state("active")
            self._blink_timer.start()

    def _on_stop_clicked(self) -> None:
        self._btn_stop.setEnabled(False)
        self._btn_pause.setEnabled(False)
        self._btn_pause.setText("Pause")
        self._status_pill.set_state("idle")
        self._stop_requested.emit()

    def _on_snapshot_clicked(self) -> None:
        """Trigger an immediate manual screen capture snapshot."""
        if self._latest_image_path and self._latest_image_path.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._latest_image_path)))
        else:
            self._force_requested.emit()

    def _on_toggle_video_clicked(self) -> None:
        """Toggle continuous screen video recording option."""
        is_on = self._worker.toggle_video()
        self._video_mode = is_on
        if is_on:
            self._btn_video_toggle.setText("Video: ON")
            self._btn_video_toggle.setStyleSheet(f"""
                QPushButton {{
                    background: {C_ROSE_BG};
                    color: {C_ROSE};
                    border: 1px solid {C_ROSE};
                    border-radius: 6px;
                    font-size: 11px;
                    font-weight: 700;
                    font-family: {FONT_UI};
                }}
            """)
        else:
            self._btn_video_toggle.setText("Video: OFF")
            self._btn_video_toggle.setStyleSheet(f"""
                QPushButton {{
                    background: {C_SURFACE_ELEV};
                    color: {C_TEXT_MUTED};
                    border: 1px solid {C_BORDER};
                    border-radius: 6px;
                    font-size: 11px;
                    font-weight: 600;
                    font-family: {FONT_UI};
                }}
            """)

    def _on_open_storage_clicked(self) -> None:
        """Open the local storage directory in Explorer."""
        if settings:
            img_dir = settings.data_dir / "images"
            img_dir.mkdir(parents=True, exist_ok=True)
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(img_dir)))

    def _on_open_meeting_notes_clicked(self) -> None:
        """Open the ambient meeting notes folder in Explorer."""
        if settings:
            notes_dir = settings.meeting_notes_dir
            notes_dir.mkdir(parents=True, exist_ok=True)
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(notes_dir)))

    def _on_demo_clicked(self) -> None:
        self._btn_demo.setEnabled(False)
        self._status_pill.set_state("active", "LOADING SIMULATION...")
        self._demo_requested.emit()

    def _on_clear_clicked(self) -> None:
        """Clears all stored frames, OCR, and extracted entities."""
        if _PIPELINE_AVAILABLE and self._db:
            try:
                conn = self._db.get_session()
                with conn:
                    conn.execute("DELETE FROM fact_evidences")
                    conn.execute("DELETE FROM entities")
                    conn.execute("DELETE FROM merged_text_records")
                    conn.execute("DELETE FROM raw_text_records")
                    conn.execute("DELETE FROM frames")
                self._screenshot_label.clear()
                self._screenshot_label.setText("MONITORING STANDBY — NO ACTIVE SCREEN CAPTURE")
                self._shot_meta.setText("Active Window: —  |  Memory cleared")
                self._text_box.clear()
                self._timeline.update_entries([])
                self._refresh_data()
                self._status_pill.set_state("idle")
            except Exception as exc:  # noqa: BLE001
                self._status_pill.set_state("error", f"Clear failed: {exc}")

    # ------------------------------------------------------------------
    # Worker signals → UI updates
    # ------------------------------------------------------------------

    def _on_capture_started(self) -> None:
        self._is_capturing = True
        self._status_pill.set_state("active")
        self._blink_timer.start()
        if _PIPELINE_AVAILABLE and self._db is None:
            try:
                self._db = Database(db_path=settings.db_path)
            except Exception:  # noqa: BLE001
                pass

    def _on_capture_stopped(self) -> None:
        self._is_capturing = False
        self._is_paused    = False
        self._btn_start.setEnabled(True)
        self._btn_stop.setEnabled(False)
        self._btn_pause.setEnabled(False)
        self._blink_timer.stop()
        self._status_pill.set_state("idle")

    def _on_capture_error(self, msg: str) -> None:
        self._btn_start.setEnabled(True)
        self._btn_pause.setEnabled(False)
        self._btn_stop.setEnabled(False)
        self._btn_demo.setEnabled(True)
        self._blink_timer.stop()
        self._status_pill.set_state("error", msg)

    def _on_snapshot_done(self, msg: str) -> None:
        self._refresh_data()

    def _on_status_update(self, stage: str, msg: str) -> None:
        logger.debug("UI Status Update [%s]: %s", stage, msg)

    def _on_simulation_finished(self, count: int) -> None:
        self._btn_demo.setEnabled(True)
        self._status_pill.set_state("active", f"SIMULATION LOADED ({count} ITEMS)")
        if _PIPELINE_AVAILABLE and self._db is None:
            try:
                self._db = Database(db_path=settings.db_path)
            except Exception:  # noqa: BLE001
                pass
        self._refresh_data()

    def _blink_rec(self) -> None:
        self._blink_on = not self._blink_on
        if self._is_capturing and not self._is_paused:
            self._status_pill._dot.setStyleSheet(
                f"color: {C_EMERALD if self._blink_on else C_BORDER}; background: transparent; border: none;"
            )

    # ------------------------------------------------------------------
    # Data refresh — called every 3 s by QTimer
    # ------------------------------------------------------------------

    def _refresh_data(self) -> None:
        if self._db is None:
            return

        try:
            self._refresh_screenshot()
            self._refresh_text()
            self._refresh_meetings()
            self._refresh_files()
            self._refresh_appointments()
            self._refresh_links()
            self._refresh_entities()
            self._refresh_verification()
            self._refresh_timeline()
        except Exception:  # noqa: BLE001
            logger.exception("Error during dashboard data refresh")

    def _refresh_screenshot(self) -> None:
        rows = self._db.get_capture_records(limit=1)
        if not rows:
            return
        row = rows[0]
        img_rel = row.get("image_path", "")
        if not img_rel:
            return
        img_abs = settings.data_dir / img_rel
        self._latest_image_path = img_abs
        if img_abs.exists():
            pix = QPixmap(str(img_abs))
            if not pix.isNull():
                scaled = pix.scaled(
                    480, 180,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                self._screenshot_label.setPixmap(scaled)
                ts = row.get("captured_at", "")
                app = row.get("application", "Desktop") or "Desktop"
                title = row.get("window_title", "")
                title_short = f" — {title[:40]}..." if title else ""
                self._shot_meta.setText(f"Active Window: {app}{title_short}  |  {ts[11:19]}")

    def _refresh_text(self) -> None:
        rows = self._db.get_raw_text_records(limit=1)
        if not rows:
            return
        text = rows[0].get("raw_text", "") or ""
        if text:
            self._text_box.setPlainText(text[:2500])

    def _refresh_meetings(self) -> None:
        rows = self._db.get_entities_by_type("meeting", limit=15)
        _IGNORED = {"whatsapp", "brave", "chrome", "firefox", "edge", "netmirror", "riom", "dashboard", "open", "home"}
        valid_rows = []
        for r in rows:
            label = self._parse_entity_label(r, ["title", "platform"]).lower()
            if not any(ig in label for ig in _IGNORED) or "meet" in label or "zoom" in label or "teams" in label:
                if not any(ig == label.strip() for ig in _IGNORED):
                    valid_rows.append(r)

        layout = self._meetings_widget._layout  # type: ignore[attr-defined]
        while layout.count():
            item = layout.takeAt(0)
            if item and item.widget():
                w = item.widget()
                w.setParent(None)
                w.deleteLater()

        if not valid_rows:
            layout.addWidget(_label("No active conferences or calls detected.", size=11, color=C_TEXT_FAINT))
            return

        seen_keys: set[str] = set()
        for r in valid_rows:
            if len(seen_keys) >= 6:
                break
            payload = r.get("payload", "{}")
            try:
                d = json.loads(payload) if isinstance(payload, str) else payload
            except Exception:  # noqa: BLE001
                d = {}

            title = (d.get("title") or "Meeting").strip()
            platform = d.get("platform")
            meeting_link = d.get("meeting_link")
            emails = d.get("emails") or []
            participants = d.get("participants") or []
            disc_pts = d.get("discussion_points") or []
            action_items = d.get("action_items") or []

            # If meeting_link wasn't extracted directly, search payload
            if not meeting_link:
                raw_match = re.search(
                    r"(https?://(?:meet\.google\.com/[a-zA-Z0-9\-_]+|(?:[a-zA-Z0-9.\-_]+\.)?zoom\.us/(?:j|my)/[0-9a-zA-Z\-_?=&]+|teams\.microsoft\.com/[^\s\)\"\'<]+|teams\.live\.com/meet/[a-zA-Z0-9]+))",
                    str(payload),
                    re.IGNORECASE,
                )
                if not raw_match:
                    raw_match = re.search(r"\b(meet\.google\.com/[a-zA-Z0-9\-_]+|zoom\.us/j/[0-9]+)\b", str(payload), re.IGNORECASE)
                    if raw_match:
                        meeting_link = f"https://{raw_match.group(1)}"
                    else:
                        code_match = re.search(r"\b([a-z]{3}-[a-z]{4}-[a-z]{3})\b", f"{title} {payload}", re.IGNORECASE)
                        if code_match:
                            meeting_link = f"https://meet.google.com/{code_match.group(1).lower()}"
                else:
                    meeting_link = raw_match.group(1)

            # Robust deduplication key
            if meeting_link:
                key = re.sub(r"^https?://(www\.)?", "", meeting_link.strip(), flags=re.IGNORECASE).rstrip("/").lower()
            else:
                key = re.sub(r"[^\w\s-]", "", title.lower()).strip()

            if not key or key in seen_keys or len(key) < 3:
                continue
            seen_keys.add(key)

            # Build rich meeting card
            card = QWidget()
            card.setStyleSheet(f"""
                QWidget {{
                    background: {C_SURFACE_ELEV};
                    border: 1px solid {C_BORDER};
                    border-radius: 6px;
                }}
            """)
            cv = QVBoxLayout(card)
            cv.setContentsMargins(12, 10, 12, 10)
            cv.setSpacing(6)

            # Top Header Row: Platform Tag + Title + Timestamp
            top_h = QHBoxLayout()
            top_h.setContentsMargins(0, 0, 0, 0)
            top_h.setSpacing(8)

            p_str = (platform or "CONFERENCE").upper()
            p_tag = _label(p_str, size=9, color=C_CYAN, bold=True, mono=True)
            p_tag.setStyleSheet(f"background: {C_CYAN}15; border: 1px solid {C_CYAN}40; border-radius: 3px; padding: 2px 6px;")
            top_h.addWidget(p_tag)

            title_lbl = _label(title, size=11, color=C_TEXT, bold=True)
            title_lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
            top_h.addWidget(title_lbl)

            ts_str = r.get("captured_at") or r.get("created_at") or ""
            try:
                ts = datetime.fromisoformat(ts_str).astimezone()
                time_part = ts.strftime("%H:%M")
            except Exception:  # noqa: BLE001
                time_part = ""
            if time_part:
                time_lbl = _label(time_part, size=10, color=C_TEXT_MUTED, mono=True)
                top_h.addWidget(time_lbl)

            cv.addLayout(top_h)

            # Link Row (if meeting_link detected)
            if meeting_link:
                href = meeting_link if meeting_link.startswith(("http://", "https://")) else f"https://{meeting_link}"
                link_h = QHBoxLayout()
                link_h.setContentsMargins(10, 0, 0, 0)
                link_h.setSpacing(6)

                link_prefix = _label("URL:", size=10, color=C_TEXT_MUTED, bold=True, mono=True)
                link_lbl = _label(
                    f"<a href='{href}' style='color: {C_CYAN}; text-decoration: underline; font-weight: 600;'>{meeting_link}</a>",
                    size=10, color=C_CYAN, mono=True
                )
                link_lbl.setOpenExternalLinks(True)
                link_lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)

                link_h.addWidget(link_prefix)
                link_h.addWidget(link_lbl)
                link_h.addStretch()
                cv.addLayout(link_h)

            # Emails & Participants Row
            details_items = []
            if emails:
                details_items.append(f"Emails: {', '.join(emails[:3])}")
            if participants:
                details_items.append(f"Participants: {', '.join(participants[:3])}")

            if details_items:
                det_lbl = _label(" | ".join(details_items), size=10, color=C_TEXT_MUTED)
                det_lbl.setContentsMargins(10, 0, 0, 0)
                cv.addWidget(det_lbl)

            # Discussion Points & Action Items
            if disc_pts:
                pts_w = _label(f"• Discussion: {' | '.join(disc_pts[:2])}", size=10, color=C_TEXT_MUTED)
                pts_w.setContentsMargins(10, 0, 0, 0)
                cv.addWidget(pts_w)

            if action_items:
                act_w = _label(f"✓ Tasks: {' | '.join(action_items[:2])}", size=10, color=C_EMERALD)
                act_w.setContentsMargins(10, 0, 0, 0)
                cv.addWidget(act_w)

            # Meeting Notes File Actions Row
            actions_h = QHBoxLayout()
            actions_h.setContentsMargins(10, 4, 0, 0)
            actions_h.setSpacing(8)

            btn_open_notes = QPushButton("📝 Open Notes (.md)")
            btn_open_notes.setFixedHeight(22)
            btn_open_notes.setCursor(Qt.CursorShape.PointingHandCursor)
            btn_open_notes.setStyleSheet(f"""
                QPushButton {{
                    background: {C_CYAN}15;
                    color: {C_CYAN};
                    border: 1px solid {C_CYAN}40;
                    border-radius: 3px;
                    font-size: 9px;
                    font-weight: 700;
                    font-family: {FONT_UI};
                    padding: 0 8px;
                }}
                QPushButton:hover {{
                    background: {C_CYAN}30;
                }}
            """)

            def _make_open_notes_handler(m_title: str, m_payload: dict):
                def _handler():
                    notes_dir = settings.meeting_notes_dir if settings else Path.home() / ".ambient_screen" / "meeting_notes"
                    notes_dir.mkdir(parents=True, exist_ok=True)
                    date_prefix = datetime.now().strftime("%Y-%m-%d")
                    safe_title = sanitize_filename(m_title)
                    target_file = notes_dir / f"{date_prefix}_{safe_title}.md"
                    if not target_file.exists():
                        m_obj = Meeting.model_validate(m_payload)
                        gen = MeetingNotesGenerator(output_dir=notes_dir)
                        target_file = gen.save_meeting_notes(m_obj)
                    QDesktopServices.openUrl(QUrl.fromLocalFile(str(target_file.resolve())))
                return _handler

            btn_open_notes.clicked.connect(_make_open_notes_handler(title, d if isinstance(d, dict) else {}))
            actions_h.addWidget(btn_open_notes)

            btn_notes_folder = QPushButton("📂 Notes Folder")
            btn_notes_folder.setFixedHeight(22)
            btn_notes_folder.setCursor(Qt.CursorShape.PointingHandCursor)
            btn_notes_folder.setStyleSheet(f"""
                QPushButton {{
                    background: {C_SURFACE};
                    color: {C_TEXT_MUTED};
                    border: 1px solid {C_BORDER};
                    border-radius: 3px;
                    font-size: 9px;
                    font-family: {FONT_UI};
                    padding: 0 6px;
                }}
                QPushButton:hover {{
                    background: {C_SURFACE_HOVER};
                    color: {C_TEXT};
                }}
            """)
            btn_notes_folder.clicked.connect(self._on_open_meeting_notes_clicked)
            actions_h.addWidget(btn_notes_folder)
            actions_h.addStretch()

            cv.addLayout(actions_h)

            layout.addWidget(card)

    def _refresh_files(self) -> None:
        rows = self._db.get_entities_by_type("file_activity", limit=20)
        clean_rows = []
        seen_stems: list[str] = []
        for r in rows:
            fname = self._parse_entity_label(r, ["file_name", "document_title"]).strip()
            if not fname or len(fname) < 4 or fname.startswith(("_", "-", ".")) and not fname.startswith(".env"):
                continue
            if any(junk in fname for junk in ["wexbw", "Mter", "ffter", "rwxy"]):
                continue
            if any(s != fname and (s.endswith(fname) or fname in s) for s in seen_stems):
                continue
            seen_stems.append(fname)
            clean_rows.append(r)

        layout = self._files_widget._layout  # type: ignore[attr-defined]
        while layout.count():
            item = layout.takeAt(0)
            if item and item.widget():
                w = item.widget()
                w.setParent(None)
                w.deleteLater()

        if not clean_rows:
            layout.addWidget(_label("No workspace files detected yet.", size=11, color=C_TEXT_FAINT))
            return

        for r in clean_rows[:8]:
            payload = r.get("payload", "{}")
            try:
                d = json.loads(payload) if isinstance(payload, str) else payload
            except Exception:  # noqa: BLE001
                d = {}

            fname = d.get("file_name") or "file"
            doc_title = d.get("document_title") or ""
            fpath = d.get("file_path") or ""
            app_name = d.get("application") or ""
            dur = d.get("estimated_duration") or ""

            row_w = QWidget()
            row_w.setStyleSheet(f"""
                QWidget {{
                    background: {C_SURFACE_ELEV};
                    border: 1px solid {C_BORDER};
                    border-radius: 6px;
                }}
            """)
            rv = QVBoxLayout(row_w)
            rv.setContentsMargins(10, 8, 10, 8)
            rv.setSpacing(4)

            # Top row: format tag + filename + app/dur
            top_h = QHBoxLayout()
            top_h.setContentsMargins(0, 0, 0, 0)
            top_h.setSpacing(6)

            ext = fname.split(".")[-1].upper() if "." in fname else "FILE"
            tag_lbl = _label(ext[:5], size=9, color=C_EMERALD, bold=True, mono=True)
            tag_lbl.setStyleSheet(f"background: {C_EMERALD}15; border: 1px solid {C_EMERALD}40; border-radius: 3px; padding: 1px 5px;")
            top_h.addWidget(tag_lbl)

            name_lbl = _label(fname, size=11, color=C_TEXT, bold=True, mono=True)
            name_lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
            top_h.addWidget(name_lbl)

            meta_info = []
            if app_name:
                meta_info.append(app_name)
            if dur:
                meta_info.append(f"⏱ {dur}")
            if meta_info:
                dur_lbl = _label(" • ".join(meta_info), size=10, color=C_TEXT_MUTED, mono=True)
                top_h.addWidget(dur_lbl)

            rv.addLayout(top_h)

            if doc_title and doc_title != fname:
                doc_lbl = _label(f"Context: {doc_title[:60]}", size=10, color=C_TEXT_MUTED)
                doc_lbl.setContentsMargins(10, 0, 0, 0)
                rv.addWidget(doc_lbl)

            if fpath:
                path_lbl = _label(f"Path: {fpath[:65]}", size=9, color=C_TEXT_FAINT, mono=True)
                path_lbl.setContentsMargins(10, 0, 0, 0)
                rv.addWidget(path_lbl)

            layout.addWidget(row_w)

    def _refresh_appointments(self) -> None:
        """Refreshes the Planned Appointments & Deadlines section."""
        rows = self._db.get_entities_by_type("appointment", limit=15)
        layout = self._appointments_widget._layout  # type: ignore[attr-defined]
        while layout.count():
            item = layout.takeAt(0)
            if item and item.widget():
                w = item.widget()
                w.setParent(None)
                w.deleteLater()

        if not rows:
            layout.addWidget(_label("No scheduled appointments or deadlines detected yet.", size=11, color=C_TEXT_FAINT))
            return

        seen_titles: set[str] = set()
        for r in rows:
            payload = r.get("payload", "{}")
            try:
                d = json.loads(payload) if isinstance(payload, str) else payload
            except Exception:  # noqa: BLE001
                d = {}

            title = (d.get("title") or "Scheduled Event").strip()
            time_val = d.get("time") or d.get("date") or ""
            deadline = d.get("deadline") or d.get("reminder") or ""

            norm = title.lower().strip()
            if norm in seen_titles or len(norm) < 4:
                continue
            seen_titles.add(norm)

            row_w = QWidget()
            row_w.setStyleSheet(f"""
                QWidget {{
                    background: {C_SURFACE_ELEV};
                    border: 1px solid {C_BORDER};
                    border-radius: 6px;
                }}
            """)
            rh = QHBoxLayout(row_w)
            rh.setContentsMargins(10, 8, 10, 8)
            rh.setSpacing(8)

            tag_lbl = _label("APPT", size=9, color=C_AMBER, bold=True, mono=True)
            tag_lbl.setStyleSheet(f"background: {C_AMBER}15; border: 1px solid {C_AMBER}40; border-radius: 3px; padding: 1px 5px;")
            rh.addWidget(tag_lbl)

            t_lbl = _label(title[:65], size=11, color=C_TEXT, bold=True)
            t_lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
            rh.addWidget(t_lbl)

            meta_str = deadline or time_val or ""
            if meta_str:
                m_lbl = _label(meta_str[:25], size=10, color=C_AMBER, mono=True)
                rh.addWidget(m_lbl)

            layout.addWidget(row_w)

    def _refresh_links(self) -> None:
        rows = self._db.get_entities_by_type("url_reference", limit=30)
        layout = self._links_widget._layout  # type: ignore[attr-defined]
        while layout.count():
            item = layout.takeAt(0)
            if item and item.widget():
                w = item.widget()
                w.setParent(None)
                w.deleteLater()

        clean_urls: list[tuple[str, str, str, str]] = []  # (url, href, title, time_part)
        seen_urls: set[str] = set()

        for r in rows:
            payload = r.get("payload", "{}")
            try:
                d = json.loads(payload) if isinstance(payload, str) else payload
            except Exception:  # noqa: BLE001
                d = {}

            url = (d.get("url") or "").strip()
            title = (d.get("title") or "").strip()
            if not url or len(url) < 8:
                continue
            if any(junk in url.lower() for junk in ["localhost", "127.0.0.1", "riom-dashboard", "favicon.ico"]):
                continue
            # Filter out meeting conference links so meetings are NOT duplicated in Web Resources
            if any(m_host in url.lower() for m_host in ["meet.google.com", "zoom.us", "teams.microsoft.com", "teams.live.com", "webex.com"]):
                continue

            # Clean trailing punctuation
            url_clean = re.sub(r"[.,;:!?()\[\]{}\'\"<>|\\]+$", "", url).strip()
            norm_key = re.sub(r"^https?://(www\.)?", "", url_clean, flags=re.IGNORECASE).rstrip("/").lower()
            if norm_key in seen_urls or len(norm_key) < 5:
                continue

            seen_urls.add(norm_key)
            href = url_clean if url_clean.startswith(("http://", "https://")) else f"https://{url_clean}"

            ts_str = r.get("captured_at") or r.get("created_at") or ""
            try:
                ts = datetime.fromisoformat(ts_str).astimezone()
                time_part = ts.strftime("%H:%M")
            except Exception:  # noqa: BLE001
                time_part = ""

            clean_urls.append((url_clean, href, title, time_part))

        if not clean_urls:
            layout.addWidget(_label("No web links captured yet.", size=11, color=C_TEXT_FAINT))
            return

        for url, href, title, time_part in clean_urls[:6]:
            row_w = QWidget()
            row_w.setStyleSheet(f"""
                QWidget {{
                    background: {C_SURFACE_ELEV};
                    border: 1px solid {C_BORDER};
                    border-radius: 6px;
                }}
            """)
            rv = QVBoxLayout(row_w)
            rv.setContentsMargins(10, 8, 10, 8)
            rv.setSpacing(4)

            # Top row: Source Domain Tag + Title (if present) / Domain + Timestamp
            top_h = QHBoxLayout()
            top_h.setContentsMargins(0, 0, 0, 0)
            top_h.setSpacing(6)

            domain_match = re.search(r"https?://(?:www\.)?([^/]+)", href)
            domain_str = domain_match.group(1) if domain_match else "WEB"

            # Clean category tag
            tag_str = "MEET" if "meet.google.com" in url or "zoom.us" in url else \
                      "GIT" if "github.com" in url or "gitlab.com" in url else \
                      "DOCS" if "docs.google.com" in url or "notion.so" in url else "WEB"

            tag_lbl = _label(tag_str, size=9, color=C_CYAN, bold=True, mono=True)
            tag_lbl.setStyleSheet(f"background: {C_CYAN}15; border: 1px solid {C_CYAN}40; border-radius: 3px; padding: 1px 5px;")
            top_h.addWidget(tag_lbl)

            display_title = title if title and title != url else domain_str
            t_lbl = _label(display_title[:60], size=11, color=C_TEXT, bold=True)
            t_lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
            top_h.addWidget(t_lbl)

            if time_part:
                time_lbl = _label(time_part, size=10, color=C_TEXT_MUTED, mono=True)
                top_h.addWidget(time_lbl)

            rv.addLayout(top_h)

            # Bottom row: Clickable URL
            url_h = QHBoxLayout()
            url_h.setContentsMargins(10, 0, 0, 0)
            url_h.setSpacing(6)

            display_url = f"{url[:65]}..." if len(url) > 65 else url
            link_lbl = _label(
                f"<a href='{href}' style='color: {C_CYAN}; text-decoration: underline;'>{display_url}</a>",
                size=10, color=C_CYAN, mono=True
            )
            link_lbl.setOpenExternalLinks(True)
            link_lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
            link_lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
            url_h.addWidget(link_lbl)

            rv.addLayout(url_h)
            layout.addWidget(row_w)

    def _refresh_entities(self) -> None:
        """Rebuild entity chips for people, orgs, projects with strict noise filtering."""
        new_flow = QWidget()
        new_flow.setStyleSheet("background: transparent; border: none;")
        vbox = QVBoxLayout(new_flow)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(6)

        _IGNORED_ENTITIES = {
            "whatsapp", "webwhatsapp", "gmail", "gmad", "google", "brave", "netmirror",
            "cypress", "github", "youtube", "facebook", "instagram", "twitter", "microsoft",
            "apple", "yahoo", "localhost", "python", "pyside", "pyside6"
        }

        seen: set[str] = set()
        current_row_widget: Optional[QWidget] = None
        current_row_layout: Optional[QHBoxLayout] = None
        row_count = 0

        for entity_type in ("person", "organization", "project"):
            rows = self._db.get_entities_by_type(entity_type, limit=20)
            for r in rows:
                name = self._parse_entity_label(r, ["name"]).strip()
                if not name or len(name) < 2:
                    continue
                if name.lower() in _IGNORED_ENTITIES:
                    continue
                if re.search(r"\d{3,}", name):
                    name = re.sub(r"[0-9_\-]+", "", name).strip()
                    if name.lower().startswith("soham"):
                        name = "Soham"

                if name and name not in seen:
                    seen.add(name)
                    if current_row_layout is None or row_count > 4:
                        current_row_widget = QWidget()
                        current_row_widget.setStyleSheet("background: transparent; border: none;")
                        current_row_layout = QHBoxLayout(current_row_widget)
                        current_row_layout.setContentsMargins(0, 0, 0, 0)
                        current_row_layout.setSpacing(6)
                        vbox.addWidget(current_row_widget)
                        row_count = 0
                    chip = EntityChip(name, entity_type)
                    current_row_layout.addWidget(chip)
                    row_count += 1

        if not seen:
            vbox.addWidget(_label("No collaborators or projects recognized yet.", size=11, color=C_TEXT_FAINT))

        old = self._entities_flow
        self._entities_container_layout.replaceWidget(old, new_flow)
        old.setParent(None)
        old.deleteLater()
        self._entities_flow = new_flow

    def _refresh_verification(self) -> None:
        conn = self._db.get_session()
        rows = conn.execute(
            "SELECT fact_id, fact_type, verification_status, fact FROM fact_evidences "
            "ORDER BY created_at DESC LIMIT 8"
        ).fetchall()
        rows = [dict(r) for r in rows]

        layout = self._verification_widget._layout  # type: ignore[attr-defined]
        while layout.count():
            item = layout.takeAt(0)
            if item and item.widget():
                w = item.widget()
                w.setParent(None)
                w.deleteLater()

        if not rows:
            layout.addWidget(_label("No verified facts recorded yet.", size=11, color=C_TEXT_FAINT))
            return

        for r in rows:
            row_w = QWidget()
            row_w.setStyleSheet(f"""
                QWidget {{
                    background: {C_SURFACE_ELEV};
                    border: 1px solid {C_BORDER};
                    border-radius: 6px;
                }}
            """)
            rh = QHBoxLayout(row_w)
            rh.setContentsMargins(8, 6, 8, 6)
            rh.setSpacing(8)

            badge = VerificationBadge(r.get("verification_status", ""))
            rh.addWidget(badge)

            fact_raw = r.get("fact", "{}")
            try:
                fact_d = json.loads(fact_raw)
                fact_text = fact_d.get("title") or fact_d.get("value") or fact_d.get("file_name") or fact_d.get("name") or str(fact_d)[:60]
            except Exception:  # noqa: BLE001
                fact_text = str(fact_raw)[:60]

            txt = _label(fact_text, size=11, color=C_TEXT, bold=True)
            txt.setWordWrap(True)
            txt.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
            rh.addWidget(txt)
            layout.addWidget(row_w)

    def _refresh_timeline(self) -> None:
        rows = self._db.get_capture_records(limit=20)
        if not rows:
            return

        _APP_COLORS = {
            "chrome":   "#38bdf8",
            "code":     "#10b981",
            "msedge":   "#38bdf8",
            "firefox":  "#f59e0b",
            "outlook":  "#f59e0b",
            "teams":    "#a855f7",
            "slack":    "#38bdf8",
            "zoom":     "#38bdf8",
            "brave":    "#f59e0b",
        }

        entries: list[tuple[str, str, str, str]] = []
        seen_apps: set[str] = set()
        for row in reversed(rows):
            app = (row.get("application") or "Desktop").split(".")[0]
            title = (row.get("window_title") or app)[:40]
            ts_str = row.get("captured_at", "")
            try:
                ts = datetime.fromisoformat(ts_str).astimezone()
                time_fmt = ts.strftime("%H:%M")
            except Exception:  # noqa: BLE001
                time_fmt = "—:——"

            key = f"{time_fmt}-{app}"
            if key in seen_apps:
                continue
            seen_apps.add(key)

            color = _APP_COLORS.get(app.lower(), C_TEXT_MUTED)
            kind = "CODE" if "code" in app.lower() else "MEET" if any(m in app.lower() for m in ["meet", "zoom", "teams"]) else "WEB"
            entries.append((time_fmt, title or app, kind, color))

        if entries:
            self._timeline.update_entries(entries[-6:])

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _parse_entity_label(self, row: dict, keys: list[str]) -> str:
        payload = row.get("payload", "{}")
        try:
            d = json.loads(payload) if isinstance(payload, str) else payload
        except Exception:  # noqa: BLE001
            return str(payload)[:60]
        for k in keys:
            v = d.get(k)
            if v:
                return str(v)[:80]
        return str(d)[:60]


def run_app(minimized_to_tray: bool = False) -> None:
    """Entry point for launching the PySide6 dashboard."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    window = MainWindow()
    if not minimized_to_tray:
        window.show()
    else:
        logger.info("RIOM started directly minimized in system tray.")
    sys.exit(app.exec())
