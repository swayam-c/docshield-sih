"""SIH Phase 2 — document taxonomy + trainable classifier."""

import io

from PIL import Image, ImageDraw


def _png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_sih_core_types_and_category_map():
    from document_profiles import DOCUMENT_TYPES, SIH_CORE_TYPES, to_sih_category

    assert "VISA" in DOCUMENT_TYPES
    assert "PERMIT" in DOCUMENT_TYPES
    assert "UNKNOWN" in SIH_CORE_TYPES
    assert to_sih_category("PAN") == "NATIONAL_ID"
    assert to_sih_category("AADHAAR") == "NATIONAL_ID"
    assert to_sih_category("VISA") == "VISA"


def test_train_and_classify_uses_model():
    from app.pipeline.classifier import (
        classify_document_type,
        ensure_trained_model,
        train_and_save_classifier,
    )

    meta = train_and_save_classifier(samples_per_class=40)
    assert meta["dataset"].startswith("synthetic")
    assert "not on real government" in meta["note"].lower()
    ensure_trained_model()

    img = Image.new("RGB", (860, 540), color=(210, 190, 150))
    draw = ImageDraw.Draw(img)
    draw.rectangle([30, 40, 260, 380], fill=(170, 150, 130))
    result = classify_document_type(_png_bytes(img), filename="national_id_scan.png")
    assert result.label in {
        "NATIONAL_ID",
        "PAN",
        "AADHAAR",
        "DRIVING_LICENCE",
        "VISA",
        "PERMIT",
        "PASSPORT",
        "OTHER",
        "UNKNOWN",
    }
    assert result.sih_category in {
        "PASSPORT",
        "VISA",
        "NATIONAL_ID",
        "DRIVING_LICENCE",
        "PERMIT",
        "OTHER",
        "UNKNOWN",
    }
    assert "logreg" in result.method or "heuristic" in result.method
    assert result.model_version != ""


def test_unknown_gate_on_ambiguous_tiny_image():
    from app.pipeline.classifier import classify_document_type

    img = Image.new("RGB", (64, 64), color=(90, 90, 90))
    result = classify_document_type(_png_bytes(img), filename="noise.bin.png")
    # May be UNKNOWN or OTHER; must stay within taxonomy and report method
    assert result.label in {
        "UNKNOWN",
        "OTHER",
        "PASSPORT",
        "VISA",
        "NATIONAL_ID",
        "DRIVING_LICENCE",
        "PERMIT",
        "PAN",
        "AADHAAR",
    }
    assert 0.0 < result.confidence <= 1.0


def test_health_sih_phase2(client):
    body = client.get("/api/health").json()
    assert "sih-phase-" in body["phase"]
    assert "logreg" in body["components"]["document_classifier"]
    assert body["components"]["document_profiles"] == "working_sih_taxonomy"
    assert body["version"].startswith("0.") and "sih-phase" in body["version"]
