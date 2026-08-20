"""
capture/mic_listener.py

Real-time microphone speech-to-text listener.
Runs as a background thread, continuously listens on the microphone,
and appends transcribed speech to the active meeting transcript .txt file.

Uses:
  - sounddevice  — low-latency cross-platform audio capture
  - SpeechRecognition — Google free STT API (requires internet)

The listener auto-activates when a meeting is detected on screen
(via the pipeline coordinator) and auto-pauses when not in a meeting.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Callable

logger = logging.getLogger(__name__)

# Indian Standard Time — UTC+05:30
_IST = timezone(timedelta(hours=5, minutes=30))


class MicTranscriptListener:
    """
    Background thread that listens on the microphone and appends
    timestamped speech lines to a .txt transcript file in real time.

    Usage:
        listener = MicTranscriptListener(transcript_path=Path("meeting.txt"))
        listener.start()
        # ... meeting happens ...
        listener.stop()
    """

    def __init__(
        self,
        transcript_path: Optional[Path] = None,
        transcript_dir: Optional[Path] = None,
        language: str = "en-US",
        phrase_timeout: float = 3.0,       # seconds of silence to end a phrase
        on_transcript: Optional[Callable[[str, str], None]] = None,  # (timestamp, text)
    ) -> None:
        self._transcript_path = transcript_path
        self._transcript_dir = transcript_dir
        self._language = language
        self._phrase_timeout = phrase_timeout
        self._on_transcript = on_transcript

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._active_meeting_title: Optional[str] = None
        self._lock = threading.Lock()

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def set_transcript_path(self, path: Path) -> None:
        """Update the active transcript file path (called when a new meeting is detected)."""
        with self._lock:
            self._transcript_path = path
            logger.info("[MIC_STT] Writing mic transcript to: %s", path)

    def set_meeting_title(self, title: str, transcript_dir: Optional[Path] = None) -> None:
        """Set the active meeting — creates/updates the transcript file path."""
        from config.settings import settings
        from processing.meeting_notes import sanitize_filename
        with self._lock:
            self._active_meeting_title = title
            target_dir = transcript_dir or self._transcript_dir or settings.data_dir / "meeting_notes"
            target_dir.mkdir(parents=True, exist_ok=True)
            date_prefix = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            safe_title = sanitize_filename(title)
            self._transcript_path = target_dir / f"{date_prefix}_{safe_title}.txt"
            logger.info("[MIC_STT] Active meeting: '%s' → %s", title, self._transcript_path)

    def start(self) -> None:
        """Start the background mic listener thread."""
        if self.is_running:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._listen_loop,
            daemon=True,
            name="MicSpeechToText",
        )
        self._thread.start()
        logger.info("[MIC_STT] Microphone speech-to-text listener started.")

    def stop(self) -> None:
        """Stop the background mic listener thread."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        self._thread = None
        logger.info("[MIC_STT] Microphone speech-to-text listener stopped.")

    def _append_to_transcript(self, text: str) -> None:
        """Append a timestamped speech line to the active .txt transcript file."""
        with self._lock:
            path = self._transcript_path
        if path is None:
            return

        time_tag = datetime.now(_IST).strftime("%H:%M:%S")
        line = f"[{time_tag}] 🎤 {text}\n"

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            # If file doesn't exist yet, write a minimal header
            if not path.exists():
                title = self._active_meeting_title or path.stem.replace("_", " ")
                header = (
                    f"Meeting: {title}\n"
                    f"Date:     {datetime.now(_IST).strftime('%Y-%m-%d %H:%M IST')}\n"
                    f"{'─' * 60}\n\n"
                )
                path.write_text(header, encoding="utf-8")

            with path.open("a", encoding="utf-8") as f:
                f.write(line)

            logger.debug("[MIC_STT] Appended to transcript: %s", text[:80])

            if self._on_transcript:
                self._on_transcript(time_tag, text)

        except Exception as exc:  # noqa: BLE001
            logger.warning("[MIC_STT] Failed to write transcript line: %s", exc)

    def _listen_loop(self) -> None:
        """Main loop: capture audio in chunks and transcribe via Google STT."""
        try:
            import speech_recognition as sr
        except ImportError:
            logger.error(
                "[MIC_STT] SpeechRecognition not installed. Run: pip install SpeechRecognition sounddevice"
            )
            return

        recognizer = sr.Recognizer()
        recognizer.pause_threshold = self._phrase_timeout
        recognizer.energy_threshold = 300       # auto-adjusts to ambient noise
        recognizer.dynamic_energy_threshold = True

        logger.info("[MIC_STT] Calibrating microphone for ambient noise...")

        try:
            with sr.Microphone() as source:
                recognizer.adjust_for_ambient_noise(source, duration=2)
                logger.info(
                    "[MIC_STT] Microphone ready (energy_threshold=%.0f). Listening...",
                    recognizer.energy_threshold,
                )

                while not self._stop_event.is_set():
                    try:
                        # listen() blocks until speech is detected then ends on silence
                        audio = recognizer.listen(source, timeout=5, phrase_time_limit=30)
                    except sr.WaitTimeoutError:
                        # No speech within timeout — just loop back
                        continue
                    except Exception as listen_exc:
                        logger.debug("[MIC_STT] Listen error: %s", listen_exc)
                        continue

                    if self._stop_event.is_set():
                        break

                    # Transcribe in a background thread so we don't block listening
                    audio_copy = audio
                    t = threading.Thread(
                        target=self._transcribe_and_save,
                        args=(recognizer, audio_copy),
                        daemon=True,
                    )
                    t.start()

        except OSError as e:
            logger.error(
                "[MIC_STT] Microphone not available: %s. "
                "Check that a microphone is connected and not in use by another app.",
                e,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("[MIC_STT] Unexpected error in mic listener: %s", exc, exc_info=True)

    def _transcribe_and_save(self, recognizer, audio) -> None:
        """Transcribe one audio chunk and save it to the transcript file."""
        try:
            import speech_recognition as sr
            text = recognizer.recognize_google(audio, language=self._language)
            text = text.strip()
            if text:
                logger.info("[MIC_STT] Transcribed: %s", text)
                self._append_to_transcript(text)
        except Exception:
            # UnknownValueError (couldn't understand), RequestError (no internet), etc.
            pass


# ---------------------------------------------------------------------------
# Singleton accessor — used by PipelineCoordinator and UI
# ---------------------------------------------------------------------------

_listener_instance: Optional[MicTranscriptListener] = None
_listener_lock = threading.Lock()


def get_mic_listener(
    transcript_dir: Optional[Path] = None,
    on_transcript: Optional[Callable[[str, str], None]] = None,
) -> MicTranscriptListener:
    """Return the global MicTranscriptListener, creating it if needed."""
    global _listener_instance
    with _listener_lock:
        if _listener_instance is None:
            from config.settings import settings
            _listener_instance = MicTranscriptListener(
                transcript_dir=transcript_dir or settings.data_dir / "meeting_notes",
                on_transcript=on_transcript,
            )
    return _listener_instance
