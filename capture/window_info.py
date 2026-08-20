"""
capture/window_info.py

Retrieves the active application name and window title at the moment
of capture.  This is inherently platform-specific.

Platform support
----------------
Windows  (implemented)
    Uses ctypes / win32gui via the standard pywin32 package, or falls
    back to ctypes alone if pywin32 is not installed.  Both approaches
    call GetForegroundWindow() + GetWindowText() from user32.dll.

macOS    (stub — safe fallback)
    Requires the 'pyobjc-framework-AppKit' or 'pyobjc-framework-Quartz'
    package.  The current stub returns None so the system keeps running.
    Full implementation: use AppKit.NSWorkspace.sharedWorkspace()
    .frontmostApplication() and Quartz.CGWindowListCopyWindowInfo().

Linux    (stub — safe fallback)
    Requires 'python-xlib' or 'wnck'.  Reads _NET_ACTIVE_WINDOW via
    Xlib.  The current stub returns None so the system keeps running.

All stubs return (None, None) so the pipeline never crashes on an
unsupported platform — the capture record simply has no window metadata.

Responsibilities
----------------
- Provide WindowInfo dataclass.
- Provide WindowInfoProvider.get() -> WindowInfo | None.
- Never raise; always catch and log.
"""
from __future__ import annotations

import logging
import platform
import sys
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

_OS = platform.system()


@dataclass
class WindowInfo:
    """Active window metadata at the moment of capture."""
    application: Optional[str]   # Process/app name, e.g. "chrome.exe" / "Google Chrome"
    window_title: Optional[str]  # Window title text, e.g. "GitHub – Google Chrome"


class WindowInfoProvider:
    """
    Returns the currently active window's application name and title.

    Usage::

        provider = WindowInfoProvider()
        info = provider.get()
        if info:
            print(info.application, info.window_title)
    """

    def get(self) -> Optional[WindowInfo]:
        """
        Return WindowInfo for the foreground window, or None on failure.
        Never raises.
        """
        try:
            if _OS == "Windows":
                return self._get_windows()
            elif _OS == "Darwin":
                return self._get_macos()
            elif _OS == "Linux":
                return self._get_linux()
            else:
                return None
        except Exception as exc:  # noqa: BLE001
            logger.debug("WindowInfoProvider.get() failed silently: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Windows implementation
    # ------------------------------------------------------------------

    @staticmethod
    def _get_windows() -> Optional[WindowInfo]:
        """
        Uses ctypes to call win32 APIs directly — no extra package needed.

        GetForegroundWindow() → HWND of the active window.
        GetWindowText()       → window title string.
        GetWindowThreadProcessId() + OpenProcess() + QueryFullProcessImageName()
                              → full executable path, from which we take the stem.
        """
        import ctypes
        import ctypes.wintypes as wintypes

        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32

        # --- Window title ---
        hwnd = user32.GetForegroundWindow()
        if not hwnd:
            return WindowInfo(application=None, window_title=None)

        length = user32.GetWindowTextLengthW(hwnd) + 1
        buf = ctypes.create_unicode_buffer(length)
        user32.GetWindowTextW(hwnd, buf, length)
        title = buf.value.strip() or None

        # --- Process name ---
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        app_name: Optional[str] = None
        if handle:
            try:
                exe_buf = ctypes.create_unicode_buffer(1024)
                size = wintypes.DWORD(1024)
                if kernel32.QueryFullProcessImageNameW(handle, 0, exe_buf, ctypes.byref(size)):
                    from pathlib import Path
                    app_name = Path(exe_buf.value).stem
            finally:
                kernel32.CloseHandle(handle)

        return WindowInfo(application=app_name, window_title=title)

    # ------------------------------------------------------------------
    # macOS stub
    # ------------------------------------------------------------------

    @staticmethod
    def _get_macos() -> Optional[WindowInfo]:
        """
        Stub — returns None.

        Full implementation requires:
            pip install pyobjc-framework-AppKit
        Then:
            from AppKit import NSWorkspace
            app = NSWorkspace.sharedWorkspace().frontmostApplication()
            name = app.localizedName()
        """
        logger.debug("Window info not yet implemented on macOS.")
        return None

    # ------------------------------------------------------------------
    # Linux stub
    # ------------------------------------------------------------------

    @staticmethod
    def _get_linux() -> Optional[WindowInfo]:
        """
        Stub — returns None.

        Full implementation requires:
            pip install python-xlib
        Then read _NET_ACTIVE_WINDOW from the root window via Xlib.
        """
        logger.debug("Window info not yet implemented on Linux.")
        return None
