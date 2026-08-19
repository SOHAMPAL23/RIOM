"""
file_manager.py

Manages the on-disk image files produced by the screen capture pipeline.

Responsibilities:
- Determine file paths for new screenshots.
- Save numpy arrays as compressed WebP files.
- Build relative paths (portable across machines).
- Delete images when a frame record is removed (privacy/storage cleanup).
- Estimate current disk usage.

Design decisions:
- WebP is used for image storage: ~60-80% smaller than PNG with near-
  lossless quality for screen content, and natively supported by Pillow.
- Images are organised in daily subdirectories: images/YYYY-MM-DD/<id>.webp
  This makes it easy to prune old data.
- The data root directory is created if it does not exist.
"""
from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


class FileManager:
    """
    Manages screenshot files on disk.

    Args:
        data_dir: Root directory where images are stored.
                  e.g. ~/.ambient_screen/
        webp_quality: WebP compression quality (0-100). 85 is a good default.
    """

    def __init__(self, data_dir: Path, webp_quality: int = 85) -> None:
        self._data_dir = data_dir
        self._images_dir = data_dir / "images"
        self._videos_dir = data_dir / "videos"
        self._webp_quality = webp_quality
        self._images_dir.mkdir(parents=True, exist_ok=True)
        self._videos_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def sanitize_name(text: Optional[str], max_len: int = 24) -> str:
        """Sanitizes application or window title for safe filename usage."""
        if not text:
            return ""
        import re
        # Remove file extension from app if present (e.g. chrome.exe -> chrome)
        if text.lower().endswith(".exe"):
            text = text[:-4]
        # Replace non-alphanumeric characters with underscore
        clean = re.sub(r"[^\w]", "_", text).strip("_")
        # Collapse multiple underscores
        clean = re.sub(r"_+", "_", clean)
        return clean[:max_len].rstrip("_")

    def save_frame(
        self,
        frame: np.ndarray,
        timestamp: datetime,
        frame_id: Optional[int] = None,
        application: Optional[str] = None,
        window_title: Optional[str] = None,
    ) -> str:
        """
        Save a BGR numpy array as a compressed WebP file with sensible contextual naming.

        Args:
            frame:        BGR numpy array from MSS.
            timestamp:    Capture timestamp (used for directory structure & filename).
            frame_id:     Database frame ID used in the filename.
            application:  Active foreground application name.
            window_title: Active window title.

        Returns:
            Relative path string (relative to data_dir) for database storage.
        """
        day_dir = self._images_dir / timestamp.strftime("%Y-%m-%d")
        day_dir.mkdir(parents=True, exist_ok=True)

        app_slug = self.sanitize_name(application, max_len=16)
        title_slug = self.sanitize_name(window_title, max_len=20)
        context_parts = [p for p in (app_slug, title_slug) if p]
        context_str = f"_{'_'.join(context_parts)}" if context_parts else ""

        ts_str = timestamp.strftime("%Y%m%d_%H%M%S")
        if frame_id is not None:
            filename = f"{ts_str}{context_str}_{frame_id:08d}.webp"
        else:
            filename = f"{ts_str}{context_str}_{timestamp.strftime('%f')}.webp"

        abs_path = day_dir / filename

        cv2.imwrite(
            str(abs_path),
            frame,
            [cv2.IMWRITE_WEBP_QUALITY, self._webp_quality],
        )

        # Return relative path for portability
        return str(abs_path.relative_to(self._data_dir))

    def rename_to_id(
        self,
        current_rel_path: str,
        frame_id: int,
        timestamp: datetime,
        application: Optional[str] = None,
        window_title: Optional[str] = None,
    ) -> str:
        """
        Rename a frame file from its temporary timestamp-based name to a
        sensible, stable ID-based name (e.g. 20260819_103000_VSCode_main_py_00000042.webp).

        Returns the new relative path. If renaming fails, returns the
        original path unchanged (a non-fatal situation).
        """
        current_abs = self.absolute_path(current_rel_path)
        if not current_abs.exists():
            return current_rel_path

        day_dir = self._images_dir / timestamp.strftime("%Y-%m-%d")
        day_dir.mkdir(parents=True, exist_ok=True)

        app_slug = self.sanitize_name(application, max_len=16)
        title_slug = self.sanitize_name(window_title, max_len=20)
        context_parts = [p for p in (app_slug, title_slug) if p]
        context_str = f"_{'_'.join(context_parts)}" if context_parts else ""
        ts_str = timestamp.strftime("%Y%m%d_%H%M%S")

        new_name = f"{ts_str}{context_str}_{frame_id:08d}.webp"
        new_abs = day_dir / new_name
        try:
            current_abs.rename(new_abs)
            return str(new_abs.relative_to(self._data_dir))
        except OSError:
            return current_rel_path  # Keep old path on rename failure

    def get_video_path(
        self,
        timestamp: datetime,
        segment_index: int = 0,
        application: Optional[str] = None,
    ) -> Path:
        """Constructs an absolute path for a segmented screen video recording file."""
        day_dir = self._videos_dir / timestamp.strftime("%Y-%m-%d")
        day_dir.mkdir(parents=True, exist_ok=True)
        app_slug = self.sanitize_name(application, max_len=16)
        app_str = f"_{app_slug}" if app_slug else ""
        filename = f"{timestamp.strftime('%Y%m%d_%H%M%S')}{app_str}_seg{segment_index:04d}.mp4"
        return day_dir / filename

    @property
    def images_dir(self) -> Path:
        return self._images_dir

    @property
    def videos_dir(self) -> Path:
        return self._videos_dir

    def absolute_path(self, relative_path: str) -> Path:
        """Resolve a stored relative path back to an absolute filesystem path."""
        return self._data_dir / relative_path

    def delete_frame(self, relative_path: str) -> bool:
        """
        Delete an image file from disk.

        Returns:
            True if deleted, False if the file did not exist.
        """
        abs_path = self.absolute_path(relative_path)
        if abs_path.exists():
            abs_path.unlink()
            return True
        return False

    def disk_usage_bytes(self) -> int:
        """Return total bytes used by all stored images and videos."""
        img_bytes = sum(f.stat().st_size for f in self._images_dir.rglob("*.webp"))
        vid_bytes = sum(f.stat().st_size for f in self._videos_dir.rglob("*.mp4"))
        return img_bytes + vid_bytes

    def prune_before(self, cutoff_date: datetime) -> int:
        """
        Delete all day-directories older than cutoff_date.

        Returns:
            Number of day-directories removed.
        """
        removed = 0
        for parent_dir in (self._images_dir, self._videos_dir):
            if not parent_dir.exists():
                continue
            for day_dir in parent_dir.iterdir():
                if not day_dir.is_dir():
                    continue
                try:
                    day = datetime.strptime(day_dir.name, "%Y-%m-%d")
                    if day < cutoff_date:
                        shutil.rmtree(day_dir)
                        removed += 1
                except ValueError:
                    pass
        return removed

