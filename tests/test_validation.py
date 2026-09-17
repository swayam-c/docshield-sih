"""Unit tests for image validation helpers."""

import io

import pytest
from PIL import Image

from app.config import Settings
from app.core.validation import UploadValidationError, validate_image_bytes


def _png(w=100, h=80) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color=(10, 20, 30)).save(buf, format="PNG")
    return buf.getvalue()


def test_validate_ok():
    settings = Settings(max_upload_bytes=5_000_000)
    result = validate_image_bytes(_png(), "doc.png", settings)
    assert result.width == 100
    assert result.mime_type == "image/png"


def test_validate_size_limit():
    settings = Settings(max_upload_bytes=10)
    with pytest.raises(UploadValidationError) as exc:
        validate_image_bytes(_png(), "doc.png", settings)
    assert exc.value.code == "FILE_TOO_LARGE"
