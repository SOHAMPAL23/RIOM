# Stage 1: Continuous Screen Capture

This document covers how to install, configure, and run the capture
module of the Ambient Screen Understanding system.

---

## What it does

| Feature | Detail |
|---|---|
| Screen capture | Uses [MSS](https://python-mss.readthedocs.io/) — zero native deps |
| Change detection | OpenCV pixel-diff; skips static screens |
| Window metadata | Foreground app name + window title per frame |
| Storage | Compressed WebP in `~/.ambient_screen/images/YYYY-MM-DD/` |
| Database | SQLite at `~/.ambient_screen/ambient.db` |
| Logging | Console + rotating file at `~/.ambient_screen/ambient.log` |
| Privacy | All data stays **100% local** in Stage 1 |

---

## Quick Start (Windows)

### 1. Prerequisites

- Python 3.11 or 3.12
- A virtual environment is strongly recommended

```powershell
# From the project root (ambient_screen/)
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

> **Minimum packages for Stage 1 only** (no OCR or LLM needed yet):
> ```
> pip install mss opencv-python numpy pydantic pydantic-settings
> ```

### 2. Configure

Copy the template and optionally edit it:

```powershell
copy .env.example .env
```

Key settings for Stage 1 (all optional — defaults work out of the box):

| Variable | Default | Description |
|---|---|---|
| `AMBIENT_DATA_DIR` | `~/.ambient_screen` | Where images and DB are stored |
| `AMBIENT_CAPTURE_INTERVAL_SECONDS` | `5.0` | Seconds between capture attempts |
| `AMBIENT_CHANGE_THRESHOLD` | `0.02` | Pixel-diff sensitivity (0 = capture everything) |
| `AMBIENT_MONITOR_INDEX` | `1` | 1 = primary monitor; 2, 3 … for additional monitors |
| `AMBIENT_LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `AMBIENT_WEBP_QUALITY` | `85` | Image compression (0–100) |
| `AMBIENT_RETENTION_DAYS` | `30` | Auto-prune images older than N days (0 = keep forever) |

### 3. Run

```powershell
# From the ambient_screen/ directory
python -m capture.run_capture
```

You will see:

```
  Ambient Screen Capture running.
  Data directory : C:\Users\YourName\.ambient_screen
  Log file       : C:\Users\YourName\.ambient_screen\ambient.log
  Interval       : 5.0s
  Press Ctrl-C to stop.
```

Every 30 seconds a status line is printed to the log:

```
2026-08-14 10:30:00  INFO      run_capture  Status — saved=42  skipped=318  errors=0  disk=8.3 MB
```

Press **Ctrl-C** to stop cleanly.

### 4. Run Tests

```powershell
# From the ambient_screen/ directory
pytest tests/test_capture.py -v
```

Expected output:

```
tests/test_capture.py::TestChangeDetector::test_first_frame_always_accepted PASSED
tests/test_capture.py::TestChangeDetector::test_identical_frame_is_rejected  PASSED
...
tests/test_capture.py::TestScreenRecorderLifecycle::test_stop_joins_thread   PASSED
```

---

## Output Files

### Images

```
~/.ambient_screen/
  images/
    2026-08-14/
      00000001.webp   ← Frame ID as 8-digit zero-padded integer
      00000002.webp
      ...
  ambient.db          ← SQLite database
  ambient.log         ← Rotating log (max 10 MB × 5 files)
```

### Database Schema (Stage 1 columns)

```sql
SELECT id, captured_at, image_path, application, window_title, monitor,
       width, height
FROM frames
ORDER BY captured_at DESC
LIMIT 10;
```

You can query this with any SQLite browser (e.g. **DB Browser for SQLite**).

---

## CaptureRecord Model

Every accepted frame produces one `CaptureRecord`:

```python
CaptureRecord(
    id=42,
    timestamp=datetime(2026, 8, 14, 10, 30, 0, tzinfo=timezone.utc),
    image_path="images/2026-08-14/00000042.webp",
    application="Code",           # VS Code process name
    window_title="screen_recorder.py — ambient_screen",
    monitor=1,
    width=2560,
    height=1440,
    file_size_bytes=104832,
)
```

---

## Platform-Specific Limitations

### Windows ✅ (fully supported)

- Window title and application name are read via `ctypes` + Win32 API
  (`GetForegroundWindow`, `QueryFullProcessImageNameW`).
- No external packages beyond `requirements.txt` are required.
- MSS captures all monitors; use `AMBIENT_MONITOR_INDEX` to pick one.
- **UAC-elevated processes**: The recorder runs as a normal user. If the
  foreground window belongs to an elevated process (e.g. Task Manager
  running as Administrator), `GetWindowText` will return an empty string.
  The capture still succeeds; only the window title will be `None`.

### macOS ⚠️ (capture works; window info is a stub)

- MSS works natively on macOS — captures succeed.
- Window title / app name returns `None` (stub in `window_info.py`).
  Full implementation requires `pyobjc-framework-AppKit`.
- **Screen Recording permission**: macOS 10.15+ requires explicit user
  consent in *System Preferences → Privacy → Screen Recording*. Without
  it, MSS will capture a blank black frame.

### Linux ⚠️ (capture works; window info is a stub)

- MSS works on X11 desktops.
- Window title / app name returns `None` (stub in `window_info.py`).
  Full implementation requires `python-xlib` and a running X session.
- Wayland is **not supported** by MSS. On Wayland-only systems,
  capture will fail. Use `XWayland` compatibility mode as a workaround.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| No images created | Change threshold too high | Lower `AMBIENT_CHANGE_THRESHOLD` to `0.0` to capture everything |
| `mss.exception.ScreenShotError` | Monitor index out of range | Use `AMBIENT_MONITOR_INDEX=1` (primary) |
| All `window_title` values are `None` | Elevated process or non-Windows | Expected — see platform notes above |
| Disk filling up fast | Threshold too low / interval too short | Increase `AMBIENT_CHANGE_THRESHOLD` or `AMBIENT_CAPTURE_INTERVAL_SECONDS` |
| Permission denied writing to data_dir | Wrong path configured | Set `AMBIENT_DATA_DIR` to a writable directory |
