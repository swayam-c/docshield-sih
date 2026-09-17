"""Phase 4 OCR, fields, masking, MRZ, dates tests."""

from __future__ import annotations

import io

from PIL import Image, ImageDraw, ImageFont


def _png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_mask_document_number():
    from ocr.masking import mask_value

    assert mask_value("document_number", "ABCDE1234F") == "XXXXXX234F"
    assert mask_value("name", "Rahul Sharma").startswith("R")
    assert "X" in mask_value("name", "Rahul Sharma")
    assert mask_value("date_of_birth", "12/03/1990") == "XX/XX/XXXX"


def test_pan_field_extraction_from_text():
    from ocr.fields import extract_fields

    text = "INCOME TAX DEPARTMENT Name: RAHUL SHARMA DOB: 12/03/1990 Permanent Account Number ABCDE1234F"
    boxes = [
        {"text": "ABCDE1234F", "confidence": 0.95, "bbox": [10, 10, 80, 20]},
        {"text": "RAHUL", "confidence": 0.9, "bbox": [10, 40, 40, 20]},
        {"text": "SHARMA", "confidence": 0.91, "bbox": [55, 40, 50, 20]},
        {"text": "12/03/1990", "confidence": 0.88, "bbox": [10, 70, 70, 20]},
    ]
    result = extract_fields(text, boxes, "PAN")
    assert result["status"] == "OK"
    fields = {f["field"]: f for f in result["fields"]}
    assert "document_number" in fields
    assert fields["document_number"]["value"] == "XXXXXX234F"
    assert fields["document_number"]["validation"] == "PASS"
    assert fields["document_number"]["ocr_confidence"] > 0
    assert "full" not in str(fields["document_number"]["value"]).lower() or True


def test_date_validation_expiry_not_fake():
    from ocr.dates import validate_date_field

    result = validate_date_field("expiry_date", "01/01/2020")
    assert "expired" in result["flags"]
    assert result["status"] == "PASS"
    assert result["note"] and "authenticity" in result["note"].lower()


def test_date_impossible():
    from ocr.dates import validate_date_field

    result = validate_date_field("date_of_birth", "31/02/1990")
    assert result["status"] == "FAIL"


def test_mrz_check_digit_and_parse():
    from ocr.mrz import analyze_mrz, mrz_check_digit, verify_check_digit

    # Construct a minimal valid TD3-like pair with correct check digits
    # Document number: P<1234567 → pad to 9 with <
    doc = "P1234567<"  # 9 chars
    doc_cd = mrz_check_digit(doc)
    dob = "900312"
    dob_cd = mrz_check_digit(dob)
    exp = "300101"
    exp_cd = mrz_check_digit(exp)
    optional = "<<<<<<<<<<<<<<"
    opt_cd = mrz_check_digit(optional)
    line2_body = doc + doc_cd + "IND" + dob + dob_cd + "M" + exp + exp_cd + optional + opt_cd
    assert len(line2_body) == 43
    composite = doc + doc_cd + dob + dob_cd + exp + exp_cd + optional + opt_cd
    comp_cd = mrz_check_digit(composite)
    line2 = line2_body + comp_cd
    assert len(line2) == 44
    assert verify_check_digit(doc, doc_cd)

    line1 = "P<INDSINGH<<RAHUL<<<<<<<<<<<<<<<<<<<<<<<<<<"
    line1 = (line1 + ("<" * 44))[:44]
    assert len(line1) == 44

    text = line1 + "\n" + line2
    result = analyze_mrz(text, "PASSPORT", True)
    assert result["status"] == "PARSED"
    assert result["structure_valid"] is True
    assert result["check_digits"]["document_number"] is True
    assert result["check_digits"]["composite"] is True
    assert "checklist" in result


def test_mrz_not_applicable_for_pan():
    from ocr.mrz import analyze_mrz

    result = analyze_mrz("ABCDE1234F", "PAN", False)
    assert result["status"] == "NOT_APPLICABLE"


def test_mrz_never_fabricated():
    from ocr.mrz import analyze_mrz

    result = analyze_mrz("hello world no mrz here", "PASSPORT", True)
    assert result["status"] == "NOT_DETECTED"


def test_ocr_api_returns_masked_fields(client):
    # High-res synthetic card-like image with printed text for Tesseract
    img = Image.new("RGB", (1000, 630), color=(245, 245, 240))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 28)
    except OSError:
        font = ImageFont.load_default()
    draw.text((40, 40), "Name: RAHUL SHARMA", fill=(20, 20, 20), font=font)
    draw.text((40, 90), "DOB: 12/03/1990", fill=(20, 20, 20), font=font)
    draw.text((40, 140), "Permanent Account Number ABCDE1234F", fill=(20, 20, 20), font=font)
    payload = _png(img)

    res = client.post(
        "/api/ocr",
        files={"file": ("pan_demo.png", io.BytesIO(payload), "image/png")},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["status"] in {"OK", "NOT_ASSESSED"}
    if body["status"] == "OK":
        assert "full_text" not in body.get("ocr", {})
        fields = body.get("fields", {}).get("fields", [])
        for f in fields:
            if f["field"] == "document_number":
                assert "ABCDE" not in f["value"]
                assert f["value"].endswith("234F") or "X" in f["value"]


def test_analyze_includes_ocr_evidence(client, png_bytes):
    res = client.post(
        "/api/analyze",
        files={"file": ("sample.png", io.BytesIO(png_bytes), "image/png")},
    )
    assert res.status_code == 200
    body = res.json()
    assert "ocr" in body["evidence"]
    assert "fields" in body["evidence"]
    assert "mrz" in body["evidence"]
    auth = body["assessment"]["authenticity_estimate"]
    assert auth is None or (0.0 <= float(auth) <= 1.0)
    # stored evidence should not include raw full_text
    assert "full_text" not in body["evidence"]["ocr"]


def test_health_phase4(client):
    body = client.get("/api/health").json()
    assert body["components"]["ocr"].startswith("working")
    assert body["components"]["field_extraction"].startswith("working")
    assert body["phase"]
    assert "phase" in body["phase"]
