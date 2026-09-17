"""Upload retention — minimize permanent storage of identity document images."""

from __future__ import annotations

import time
from pathlib import Path
from typing import List

from app.config import Settings


def store_upload(settings: Settings, stored_name: str, content: bytes) -> Path:
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    dest = settings.upload_dir / stored_name
    # Refuse path escape
    if dest.resolve().parent != settings.upload_dir.resolve():
        raise ValueError("Invalid storage path.")
    dest.write_bytes(content)
    return dest


def delete_upload(settings: Settings, stored_name: str) -> bool:
    if not stored_name:
        return False
    path = settings.upload_dir / Path(stored_name).name
    if path.exists() and path.is_file():
        path.unlink(missing_ok=True)
        return True
    return False


def purge_expired_uploads(settings: Settings) -> List[str]:
    """Delete upload files older than upload_ttl_seconds. Returns deleted names."""
    ttl = int(settings.upload_ttl_seconds)
    if ttl <= 0:
        return []
    deleted: List[str] = []
    now = time.time()
    upload_dir = settings.upload_dir
    if not upload_dir.exists():
        return deleted
    for path in upload_dir.iterdir():
        if not path.is_file() or path.name.startswith("."):
            continue
        age = now - path.stat().st_mtime
        if age > ttl:
            path.unlink(missing_ok=True)
            deleted.append(path.name)
    return deleted
