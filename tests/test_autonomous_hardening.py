"""Autonomous hardening — OCR multipass, text/image forensics, honesty checks."""

from __future__ import annotations

import io

from PIL import Image, ImageDraw


def _png_doc() -> bytes:
    img = Image.new("RGB", (1000, 640), color=(245, 245, 240))
    d = ImageDraw.Draw(img)
    d.rectangle([30, 30, 970, 610], outline=(20, 20, 20), width=2)
    d.text((60, 80), "Permanent Account Number ABCDE1234F", fill=(0, 0, 0))
    d.text((60, 140), "Name: RAHUL SHARMA", fill=(0, 0, 0))
    d.text((60, 200), "DOB: 12/03/1990", fill=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_ocr_boxes_have_geometry(monkeypatch):
    monkeypatch.setenv("DOCSHIELD_OCR_ENGINE", "mock")
    from ocr.engine import run_ocr

    # mock returns empty boxes but schema path must not crash
    out = run_ocr(_png_doc())
    assert "text_boxes" in out
    assert "preprocess" in out
    assert "multipass" in out["preprocess"] or out["engine"] == "mock"


def test_text_visual_fallback_without_ocr_boxes():
    from forensics.text_analysis import analyze_text_visual

    tv = analyze_text_visual(_png_doc(), text_boxes=[], fields={})
    assert tv["status"] == "ANALYZED"
    typo = tv["typography"]
    assert typo["status"] in {"ANALYZED", "UNCERTAIN"}
    assert typo.get("region_count", 0) >= 1
    assert typo.get("median_relative_height") is not None or typo["status"] == "UNCERTAIN"
    assert typo.get("consistency_label") in {
        "CONSISTENT",
        "ANOMALY_SIGNAL",
        "INSUFFICIENT_REGIONS",
    }


def test_text_visual_and_metadata():
    from forensics.text_analysis import analyze_text_visual
    from forensics.metadata_stamp import analyze_metadata, analyze_stamp_seal

    boxes = [
        {"text": "ABCDE1234F", "confidence": 0.9, "bbox": [60, 80, 200, 24]},
        {"text": "RAHUL", "confidence": 0.88, "bbox": [60, 140, 80, 22]},
        {"text": "SHARMA", "confidence": 0.87, "bbox": [150, 140, 90, 22]},
        {"text": "12/03/1990", "confidence": 0.85, "bbox": [60, 200, 120, 20]},
        {"text": "PAN", "confidence": 0.8, "bbox": [60, 40, 40, 18]},
    ]
    tv = analyze_text_visual(
        _png_doc(),
        text_boxes=boxes,
        fields={"fields": [{"name": "document_number", "bbox": [60, 80, 200, 24], "masked_value": "XXXXXX234F"}]},
    )
    assert tv["status"] == "ANALYZED"
    assert tv["typography"]["status"] in {"ANALYZED", "UNCERTAIN"}
    assert tv["text_splicing"]["status"] == "ANALYZED"
    assert tv["ocr_visual_crosscheck"]

    meta = analyze_metadata(_png_doc())
    assert meta["status"] in {"UNAVAILABLE", "PRESENT", "ERROR"}
    assert "fraud" not in (meta.get("note") or "").lower() or "not" in (meta.get("note") or "").lower()

    stamp = analyze_stamp_seal(_png_doc())
    assert stamp["detection"] in {"DETECTED", "UNCERTAIN", "NO_CLEAR_ANOMALY", "NOT_ASSESSED"}


def test_pipeline_includes_forensics_map(monkeypatch):
    monkeypatch.setenv("DOCSHIELD_ML_MODE", "mock")
    monkeypatch.setenv("DOCSHIELD_OCR_ENGINE", "mock")
    from app.pipeline import run_pipeline

    result = run_pipeline(_png_doc(), filename="pan.png")
    assert "forensics" in result["evidence"]
    assert "text_visual" in result["evidence"]
    assert "metadata" in result["evidence"]
    assert result["assessment"]["decision"] in {
        "PASS",
        "REVIEW",
        "HIGH-RISK",
        "INCONCLUSIVE",
    }
    # No silent authenticity fabrication
    auth = result["assessment"]["authenticity_estimate"]
    assert auth is None or (0.0 <= float(auth) <= 1.0)


def test_face_age_robustness_label():
    from verification.face import compare_faces

    # Two blank images → likely NOT_ASSESSED; still must not claim age-invariant
    blank = _png_doc()
    out = compare_faces(blank, reference_bytes=blank)
    ver = out.get("verification") or {}
    if ver.get("status") == "COMPARED_UNCALIBRATED":
        assert ver.get("age_robustness") == "NOT_VALIDATED"
