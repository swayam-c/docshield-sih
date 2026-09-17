"""Pytest fixtures for DOCSHIELD AI Phase 2."""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest
from PIL import Image
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture()
def png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (320, 200), color=(40, 90, 120)).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture()
def client(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()

    monkeypatch.setenv("DOCSHIELD_DB_URL", f"sqlite:///{db_path.as_posix()}")
    monkeypatch.setenv("DOCSHIELD_UPLOAD_DIR", str(upload_dir))
    monkeypatch.setenv("DOCSHIELD_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("DOCSHIELD_TEMP_DIR", str(tmp_path / "temp"))
    monkeypatch.setenv("DOCSHIELD_API_KEY", "")
    monkeypatch.setenv("DOCSHIELD_ML_MODE", "mock")
    monkeypatch.setenv("DOCSHIELD_RETAIN_UPLOADS", "false")
    monkeypatch.setenv("DOCSHIELD_ALLOW_PDF", "true")
    monkeypatch.setenv("DOCSHIELD_RATE_LIMIT_PER_MINUTE", "0")

    # Reset settings cache and rebuild app/engine with test env
    from app.config import get_settings
    from app.ml.device import get_device
    from app.ml.registry import clear_registry

    get_settings.cache_clear()
    clear_registry()
    get_device.cache_clear()

    from app.db import Base
    import app.db as db_module

    engine = create_engine(
        f"sqlite:///{db_path.as_posix()}",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    # Import models then create tables
    from app.db import models  # noqa: F401

    Base.metadata.create_all(bind=engine)

    db_module.engine = engine
    db_module.SessionLocal = TestingSessionLocal

    from app.main import create_app

    application = create_app()

    def override_get_db():
        session = TestingSessionLocal()
        try:
            yield session
        finally:
            session.close()

    from app.db import get_db

    application.dependency_overrides[get_db] = override_get_db

    with TestClient(application) as test_client:
        yield test_client

    get_settings.cache_clear()
