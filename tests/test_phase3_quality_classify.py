"""Tests for image quality and document classification."""

import io

from PIL import Image, ImageDraw, ImageFilter


def _png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_quality_sharp_image_has_reasonable_score():
    from app.pipeline.quality import analyze_image_quality

    img = Image.new("RGB", (1200, 760), color=(240, 240, 235))
    draw = ImageDraw.Draw(img)
    for y in range(40, 720, 28):
        draw.rectangle([40, y, 1160, y + 14], fill=(30, 30, 30))
    report = analyze_image_quality(_png_bytes(img))
    assert 0.0 <= report.overall_quality <= 1.0
    assert report.resolution in {"GOOD", "FAIR", "POOR"}
    assert "authenticity" not in (report.reason or "").lower()
    d = report.to_dict()
    assert "NOT document authenticity" in d["note"] or "NOT" in d["note"]


def test_quality_blurry_triggers_inconclusive_flag():
    from app.pipeline.quality import analyze_image_quality

    img = Image.new("RGB", (200, 120), color=(128, 128, 128))
    img = img.filter(ImageFilter.GaussianBlur(radius=8))
    report = analyze_image_quality(_png_bytes(img))
    assert report.blur == "HIGH" or report.insufficient_for_analysis
    assert report.overall_quality < 0.7


def test_classifier_returns_known_labels():
    from app.pipeline.classifier import classify_document_type
    from document_profiles import DOCUMENT_TYPES

    # Card-like aspect ~1.586
    img = Image.new("RGB", (858, 540), color=(220, 200, 160))
    draw = ImageDraw.Draw(img)
    draw.rectangle([40, 40, 280, 360], fill=(180, 160, 140))
    result = classify_document_type(_png_bytes(img), filename="sample_pan_card.png")
    assert result.label in DOCUMENT_TYPES
    assert 0.0 < result.confidence <= 1.0
    assert "heuristic" in result.method or "logreg" in result.method
    assert hasattr(result, "sih_category")
    assert isinstance(result.alternatives, list)


def test_classifier_filename_prior_passport():
    from app.pipeline.classifier import classify_document_type

    img = Image.new("RGB", (700, 500), color=(40, 70, 140))
    result = classify_document_type(_png_bytes(img), filename="my_passport_scan.jpg")
    assert result.label in {
        "PASSPORT",
        "OTHER",
        "DRIVING_LICENCE",
        "AADHAAR",
        "PAN",
        "VISA",
        "NATIONAL_ID",
        "PERMIT",
        "UNKNOWN",
    }
    # Passport should be competitive given blue + filename
    labels = [result.label] + [a["label"] for a in result.alternatives]
    assert "PASSPORT" in labels or result.label == "PASSPORT"


def test_analyze_api_phase3(client, png_bytes):
    res = client.post(
        "/api/analyze",
        files={"file": ("doc.png", io.BytesIO(png_bytes), "image/png")},
    )
    assert res.status_code == 200
    body = res.json()
    assert "image_quality" in body
    assert "document_type" in body
    assert body["assessment"]["evidence_quality"] is not None
    auth = body["assessment"]["authenticity_estimate"]
    assert auth is None or (0.0 <= float(auth) <= 1.0)
    assert body["assessment"]["decision"] in {"PENDING", "INCONCLUSIVE"}
    assert body["image_quality"]["overall_quality"] == body["assessment"]["evidence_quality"]

    case = client.get(f"/api/case/{body['case_id']}").json()
    assert case["document_type"] == body["document_type"]["label"]
    assert case["decision"] == body["assessment"]["decision"]


def test_quality_endpoint(client, png_bytes):
    res = client.post(
        "/api/quality",
        files={"file": ("doc.png", io.BytesIO(png_bytes), "image/png")},
    )
    assert res.status_code == 200
    assert "overall_quality" in res.json()


def test_classify_endpoint(client, png_bytes):
    res = client.post(
        "/api/classify",
        files={"file": ("aadhaar_demo.png", io.BytesIO(png_bytes), "image/png")},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["label"] in {
        "PAN",
        "AADHAAR",
        "PASSPORT",
        "DRIVING_LICENCE",
        "VISA",
        "NATIONAL_ID",
        "PERMIT",
        "OTHER",
        "UNKNOWN",
    }
    assert "sih_category" in body


def test_health_phase3_flags(client):
    body = client.get("/api/health").json()
    assert body["components"]["image_quality"] == "working"
    assert "heuristic" in body["components"]["document_classifier"] or "logreg" in body[
        "components"
    ]["document_classifier"]
    assert body["phase"]  # e.g. sih-phase-2-...
    assert "phase" in body["phase"]


def test_profiles_registered():
    from document_profiles import DOCUMENT_TYPES, SIH_CORE_TYPES, get_profile, list_profiles

    assert len(list_profiles()) == len(DOCUMENT_TYPES)
    assert get_profile("PASSPORT").supports_mrz is True
    assert get_profile("PAN").supports_mrz is False
    assert get_profile("VISA").document_type == "VISA"
    assert get_profile("UNKNOWN").document_type == "UNKNOWN"
    assert set(DOCUMENT_TYPES) >= {
        "PAN",
        "AADHAAR",
        "PASSPORT",
        "DRIVING_LICENCE",
        "VISA",
        "NATIONAL_ID",
        "PERMIT",
        "OTHER",
        "UNKNOWN",
    }
    assert "UNKNOWN" in SIH_CORE_TYPES
