from __future__ import annotations

from pathlib import Path

from sorawm.configs import ACTIVE_VIDEO_FILE, RESOURCES_DIR, ROOT

ACTIVE_SOURCE_FILE = RESOURCES_DIR / "active_source.txt"


def _normalise(path: Path, relative: bool) -> str:
    if relative:
        try:
            return str(path.relative_to(ROOT)).replace("\\", "/")
        except ValueError:
            return str(path).replace("\\", "/")
    return str(path)


def set_active_source(path: Path) -> None:
    resolved = Path(path).expanduser().resolve()
    ACTIVE_SOURCE_FILE.parent.mkdir(parents=True, exist_ok=True)
    ACTIVE_SOURCE_FILE.write_text(str(resolved), encoding="utf-8")


def _ensure_active_source_record() -> Path | None:
    """Ensure we can recover the most recently activated model path."""
    if ACTIVE_SOURCE_FILE.exists():
        raw = ACTIVE_SOURCE_FILE.read_text(encoding="utf-8").strip()
        if raw:
            return Path(raw)
    default = (RESOURCES_DIR / "best.pt").resolve()
    if default.exists():
        set_active_source(default)
        return default
    return None


def get_active_source(relative: bool = True) -> str | None:
    resolved = _ensure_active_source_record()
    if resolved is None:
        return None
    return _normalise(resolved, relative=relative)


def _ensure_active_video_record() -> Path | None:
    if ACTIVE_VIDEO_FILE.exists():
        raw = ACTIVE_VIDEO_FILE.read_text(encoding="utf-8").strip()
        if raw:
            candidate = Path(raw)
            if candidate.exists():
                return candidate
    return None


def set_active_video(path: Path | None) -> None:
    ACTIVE_VIDEO_FILE.parent.mkdir(parents=True, exist_ok=True)
    if path is None:
        if ACTIVE_VIDEO_FILE.exists():
            ACTIVE_VIDEO_FILE.unlink()
        return
    resolved = Path(path).expanduser().resolve()
    ACTIVE_VIDEO_FILE.write_text(str(resolved), encoding="utf-8")


def get_active_video(relative: bool = True) -> str | None:
    resolved = _ensure_active_video_record()
    if resolved is None:
        return None
    return _normalise(resolved, relative=relative)
