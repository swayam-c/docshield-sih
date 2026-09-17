"""SIH Phase 3 — enriched OCR field extraction."""

from __future__ import annotations


def test_visa_fields_extracted():
    from ocr.fields import extract_fields

    text = (
        "Name: PRIYA NAYAR\n"
        "Nationality: IND\n"
        "Sex: F\n"
        "DOB: 15/08/1992\n"
        "Visa Number: V123456789\n"
        "Visa Type: Tourist\n"
        "Entries: Multiple\n"
        "Duration of Stay: 90 days\n"
        "Issue Date: 01/01/2024\n"
        "Expiry Date: 01/01/2025\n"
    )
    boxes = [
        {"text": "PRIYA", "confidence": 0.9, "bbox": [10, 10, 40, 16]},
        {"text": "NAYAR", "confidence": 0.9, "bbox": [55, 10, 40, 16]},
        {"text": "V123456789", "confidence": 0.95, "bbox": [10, 80, 90, 16]},
        {"text": "Tourist", "confidence": 0.88, "bbox": [10, 100, 50, 16]},
    ]
    result = extract_fields(text, boxes, "VISA")
    assert result["status"] in {"OK", "PARTIAL"}
    by = {f["field"]: f for f in result["fields"]}
    assert "visa_number" in by or "document_number" in by
    assert "visa_type" in by
    assert "entries" in by
    assert "nationality" in by
    assert by["visa_type"]["value"] == "Tourist"
    assert "source_region" in by["visa_type"] or by["visa_type"]["bounding_box"] is None
    assert result["field_completeness"] is not None
    assert "missing_expected_fields" in result


def test_passport_fields_with_region_and_sex():
    from ocr.fields import extract_fields

    text = (
        "Name: RAHUL SINGH\n"
        "Passport No: A1234567\n"
        "Nationality: IND\n"
        "Sex: M\n"
        "Date of Birth: 12/03/1990\n"
        "Date of Issue: 01/06/2018\n"
        "Date of Expiry: 01/06/2028\n"
    )
    boxes = [{"text": "A1234567", "confidence": 0.97, "bbox": [20, 40, 70, 18]}]
    result = extract_fields(text, boxes, "PASSPORT")
    by = {f["field"]: f for f in result["fields"]}
    assert by["document_number"]["validation"] == "PASS"
    assert by["document_number"]["value"].endswith("4567")
    assert "XXXX" in by["document_number"]["value"] or by["document_number"]["value"].startswith("X")
    assert by["document_number"]["source_region"]["x"] == 20
    assert by["sex"]["value"] == "M"
    assert by["nationality"]["value"] == "IND"


def test_date_ordering_consistency_fail():
    from ocr.fields import extract_fields

    text = (
        "Name: TEST USER\n"
        "DOB: 01/01/2000\n"
        "Issue Date: 01/01/1990\n"
        "Expiry Date: 01/01/2025\n"
        "Permanent Account Number ABCDE1234F\n"
    )
    result = extract_fields(text, [], "PAN")
    statuses = {c["check"]: c["status"] for c in result["consistency_checks"]}
    assert statuses.get("issue_after_dob") == "FAIL"


def test_pan_still_masks_and_passes():
    from ocr.fields import extract_fields

    text = "Name: RAHUL SHARMA DOB: 12/03/1990 Permanent Account Number ABCDE1234F Father Name: RAM SHARMA"
    boxes = [{"text": "ABCDE1234F", "confidence": 0.95, "bbox": [10, 10, 80, 20]}]
    result = extract_fields(text, boxes, "PAN")
    by = {f["field"]: f for f in result["fields"]}
    assert by["document_number"]["value"] == "XXXXXX234F"
    assert "father_name" in by


def test_health_sih_phase3(client):
    body = client.get("/api/health").json()
    assert "sih-phase-" in body["phase"]
    assert body["components"]["field_extraction"] == "working_profile_driven"
    assert body["version"].startswith("0.") and "sih-phase" in body["version"]


def test_reconstruct_full_text_lines():
    from ocr.engine import TextBox, reconstruct_full_text

    boxes = [
        TextBox("Name:", 0.9, [0, 0, 10, 10], line_num=1, block_num=1),
        TextBox("RAHUL", 0.9, [12, 0, 10, 10], line_num=1, block_num=1),
        TextBox("DOB:", 0.9, [0, 20, 10, 10], line_num=2, block_num=1),
        TextBox("12/03/1990", 0.9, [12, 20, 10, 10], line_num=2, block_num=1),
    ]
    text = reconstruct_full_text(boxes)
    assert "\n" in text
    assert "RAHUL" in text
