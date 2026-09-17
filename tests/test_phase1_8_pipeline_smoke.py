"""End-to-end Phase 1–8 pipeline smoke across supported document types."""

from __future__ import annotations

import io

import pytest
from PIL import Image, ImageDraw, ImageFont


def _png(text_lines, w=900, h=600, color=(240, 240, 235)) -> bytes:
    img = Image.new("RGB", (w, h), color=color)
    draw = ImageDraw.Draw(img)
    y = 40
    for line in text_lines:
        draw.text((40, y), line, fill=(20, 20, 20))
        y += 28
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(autouse=True)
def _mock_ml(monkeypatch):
    monkeypatch.setenv("DOCSHIELD_ML_MODE", "mock")


def _run(image_bytes: bytes, filename: str):
    from app.pipeline import run_pipeline

    return run_pipeline(image_bytes, filename=filename)


def test_pipeline_passport_smoke():
    img = _png(
        [
            "REPUBLIC OF INDIA PASSPORT",
            "Surname / Nom",
            "SHARMA",
            "Given Names / Prenoms",
            "RAHUL KUMAR",
            "Passport No. A1234567",
            "Nationality INDIAN",
            "Date of Birth 15/03/1990",
            "Sex M",
            "Date of Issue 01/01/2020",
            "Date of Expiry 31/12/2030",
        ]
    )
    result = _run(img, "passport_sample.png")
    sr = result["screening_result"]
    assert "17_decision" in sr["phase_status"]
    auth = result["assessment"]["authenticity_estimate"]
    assert auth is None or (0.0 <= float(auth) <= 1.0)
    face = result["evidence"].get("face") or {}
    assert face.get("status") in {"NOT_DETECTED", "DETECTED", "MULTIPLE_DETECTED", "ERROR"}
    assert result["evidence"]["tampering"]["status"] in {"ANALYZED", "ERROR"}
    assert result["evidence"]["tampering"].get("concern") in {
        "HEURISTIC_REVIEW_ONLY",
        "NOT_ASSESSED",
    }
    loc = result["evidence"]["tampering"].get("localization") or {}
    assert loc.get("status") in {"LOCALIZED", "ERROR", "NOT_ASSESSED"}
    fields = result["evidence"]["fields"]
    # Completeness uses EXTRACTED only — may be low with mock OCR; status must be honest
    for f in fields.get("fields") or []:
        if (f.get("ocr_confidence") or 0) < 0.15:
            assert f.get("validation") != "PASS"
            assert f.get("status") != "PASS"
    mrz = result["evidence"]["mrz"]
    assert mrz["status"] in {"NOT_DETECTED", "PARSED", "PARTIAL", "NOT_APPLICABLE", "NOT_ASSESSED"}
    ai = result["evidence"]["ai_forensics"]
    assert ai.get("efficientnet", {}).get("status") in {
        "FEATURES_EXTRACTED",
        "SKIPPED",
        "ERROR",
        "MODEL_NOT_READY",
    }
    head = (ai.get("efficientnet") or {}).get("authenticity_head") or {}
    if head.get("status") == "SCORED":
        assert head.get("concern") == "UNCALIBRATED_SYNTHETIC"


def test_pipeline_aadhaar_mrz_na():
    img = _png(["AADHAAR", "Name: TEST USER", "DOB: 01/01/1990", "1234 5678 9012"])
    result = _run(img, "aadhaar_card.png")
    # If classified as AADHAAR/PAN/DL without MRZ → NOT_APPLICABLE
    label = result["document_type_label"]
    mrz = result["evidence"]["mrz"]
    from document_profiles import get_profile

    if not get_profile(label).supports_mrz:
        assert mrz["status"] == "NOT_APPLICABLE"


def test_pipeline_visa_smoke():
    img = _png(
        [
            "VISA",
            "Name: PRIYA NAYAR",
            "Visa Number: V123456789",
            "Visa Type: Tourist",
            "Nationality: IND",
        ]
    )
    result = _run(img, "visa_entry.png")
    auth = result["assessment"]["authenticity_estimate"]
    assert auth is None or (0.0 <= float(auth) <= 1.0)
    assert "screening_result" in result


def test_pipeline_dl_smoke():
    img = _png(["DRIVING LICENCE", "Name: DRIVER ONE", "DL No: MH12 20110012345", "DOB: 01/01/1990"])
    result = _run(img, "dl_sample.png")
    from document_profiles import get_profile

    label = result["document_type_label"]
    if not get_profile(label).supports_mrz:
        assert result["evidence"]["mrz"]["status"] == "NOT_APPLICABLE"


def test_pipeline_unknown_image():
    img = _png(["BANANA BREAD RECIPE", "flour 2 cups", "bake 350F"], color=(180, 200, 120))
    result = _run(img, "grocery_receipt.png")
    auth = result["assessment"]["authenticity_estimate"]
    assert auth is None or (0.0 <= float(auth) <= 1.0)
    # Decision engine may triage to REVIEW / HIGH-RISK; must not invent PASS on junk text
    assert result["assessment"]["decision"] in {
        "PENDING",
        "INCONCLUSIVE",
        "REVIEW",
        "HIGH-RISK",
    }
    assert result["assessment"]["decision"] != "PASS"


def test_pipeline_low_quality_inconclusive():
    # Tiny dark blurry-ish image
    img = Image.new("RGB", (40, 30), color=(5, 5, 5))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    result = _run(buf.getvalue(), "tiny_dark.png")
    assert result["assessment"]["decision"] == "INCONCLUSIVE"
    auth = result["assessment"]["authenticity_estimate"]
    assert auth is None or (0.0 <= float(auth) <= 1.0)
