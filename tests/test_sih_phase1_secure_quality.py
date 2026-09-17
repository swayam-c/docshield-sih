"""SIH Phase 1 — secure input + image quality tests."""

from __future__ import annotations

import io

import pytest
from PIL import Image


def _png(w=640, h=400) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color=(210, 210, 205)).save(buf, format="PNG")
    return buf.getvalue()


# Minimal one-page PDF (blank MediaBox) — sufficient for signature + render smoke
_MIN_PDF = b"""%PDF-1.4
1 0 obj<< /Type /Catalog /Pages 2 0 R >>endobj
2 0 obj<< /Type /Pages /Kids [3 0 R] /Count 1 >>endobj
3 0 obj<< /Type /Page /Parent 2 0 R /MediaBox [0 0 400 300] /Contents 4 0 R >>endobj
4 0 obj<< /Length 0 >>stream
endstream
endobj
xref
0 5
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000206 00000 n 
trailer<< /Size 5 /Root 1 0 R >>
startxref
253
%%EOF
"""


def test_reject_exe_polyglot():
    from app.config import Settings
    from app.core.validation import UploadValidationError, validate_image_bytes

    settings = Settings(allow_pdf=True, max_upload_bytes=1_000_000)
    with pytest.raises(UploadValidationError) as exc:
        validate_image_bytes(b"MZ\x90\x00fake", "doc.png", settings)
    assert exc.value.code == "DANGEROUS_CONTENT"


def test_reject_html_content():
    from app.config import Settings
    from app.core.validation import UploadValidationError, validate_image_bytes

    settings = Settings(allow_pdf=False)
    with pytest.raises(UploadValidationError) as exc:
        validate_image_bytes(b"<html><body>x</body></html>", "x.png", settings)
    assert exc.value.code == "DANGEROUS_CONTENT"


def test_magic_byte_mismatch():
    from app.config import Settings
    from app.core.validation import UploadValidationError, validate_image_bytes

    settings = Settings()
    with pytest.raises(UploadValidationError) as exc:
        validate_image_bytes(_png(), "photo.jpg", settings)
    assert exc.value.code == "MIME_MISMATCH"


def test_quality_scores_percentages():
    from app.pipeline.quality import analyze_image_quality

    report = analyze_image_quality(_png(1200, 800))
    data = report.to_dict()
    assert "scores" in data
    for key in ("resolution", "sharpness", "lighting", "perspective", "overall_quality"):
        assert key in report.scores
        assert 0.0 <= report.scores[key] <= 100.0
    assert "compression" in data
    assert "NOT document authenticity" in data["note"]


def test_document_upload_endpoint(client):
    res = client.post(
        "/api/document/upload",
        files={"file": ("card.png", io.BytesIO(_png()), "image/png")},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ACCEPTED"
    assert body["upload"]["retained"] is False
    assert body["stored_filename"] is None


def test_system_status(client):
    body = client.get("/api/system/status").json()
    assert body["modules"]["Database"] == "READY"
    assert "secure_input" in body
    assert body["demo_banner"]


def test_analyze_alias_document_analyze(client):
    res = client.post(
        "/api/document/analyze",
        files={"file": ("card.png", io.BytesIO(_png(1000, 640)), "image/png")},
    )
    assert res.status_code == 200
    body = res.json()
    assert "scores" in body["image_quality"]
    assert body["upload"]["retained"] is False


def test_pdf_intake_renders_first_page():
    pytest.importorskip("pypdfium2")
    from app.config import Settings
    from app.core.validation import validate_image_bytes

    settings = Settings(
        allow_pdf=True,
        max_upload_bytes=5_000_000,
        allowed_extensions=".png,.jpg,.jpeg,.webp,.pdf",
    )
    result = validate_image_bytes(_MIN_PDF, "scan.pdf", settings)
    assert result.source_format == "PDF"
    assert result.mime_type == "image/png"
    assert result.width > 0 and result.height > 0
    assert result.content.startswith(b"\x89PNG")


def test_pdf_disabled():
    from app.config import Settings
    from app.core.validation import UploadValidationError, validate_image_bytes

    settings = Settings(allow_pdf=False, allowed_extensions=".png,.jpg,.jpeg,.webp")
    with pytest.raises(UploadValidationError) as exc:
        validate_image_bytes(_MIN_PDF, "scan.pdf", settings)
    assert exc.value.code in {"PDF_DISABLED", "BAD_EXTENSION"}
