"""Passport OCR / zero-confidence honesty / classification aliases."""

from __future__ import annotations

from ocr.fields import extract_fields
from ocr.passport_fields import extract_passport_fields


def test_passport_extracts_surname_given_and_number():
    text = """
    REPUBLIC OF INDIA
    Surname / Nom
    SHARMA
    Given Names / Prenoms
    RAHUL KUMAR
    Passport No. A1234567
    Nationality / Nationalite
    INDIAN
    Date of Birth 15/03/1990
    Sex / Sexe M
    Date of Issue 01/01/2020
    Date of Expiry 31/12/2030
    """
    boxes = [
        {"text": "SHARMA", "confidence": 0.92, "bbox": [10, 20, 80, 20]},
        {"text": "RAHUL", "confidence": 0.91, "bbox": [10, 50, 60, 20]},
        {"text": "KUMAR", "confidence": 0.90, "bbox": [80, 50, 60, 20]},
        {"text": "A1234567", "confidence": 0.95, "bbox": [10, 100, 100, 20]},
        {"text": "INDIAN", "confidence": 0.88, "bbox": [10, 140, 80, 20]},
        {"text": "15/03/1990", "confidence": 0.85, "bbox": [10, 180, 100, 20]},
        {"text": "M", "confidence": 0.99, "bbox": [10, 220, 20, 20]},
    ]
    result = extract_fields(text, boxes, "PASSPORT")
    assert result["document_type"] == "PASSPORT"
    by_name = {f["field"]: f for f in result["fields"]}
    assert by_name["name"]["status"] == "EXTRACTED"
    assert by_name["passport_number"]["status"] == "EXTRACTED"
    assert by_name["nationality"]["status"] == "EXTRACTED"
    assert result["field_completeness"] is not None
    assert result["field_completeness"] > 0.4


def test_zero_confidence_never_pass():
    fields, _ = extract_passport_fields(
        "Father's Name: UNKNOWN PERSON\nName: TEST USER",
        [{"text": "UNKNOWN", "confidence": 0.0, "bbox": [1, 1, 10, 10]}],
    )
    # father_name is not a passport expected extractor field in passport_fields
    # Ensure any value with unmatched/zero conf is not PASS
    for f in fields:
        if f.get("value_present") and (f.get("ocr_confidence") is None or f["ocr_confidence"] <= 0):
            assert f["validation"] != "PASS"
            assert f["status"] in {"UNCERTAIN", "LOW_CONFIDENCE", "NOT_DETECTED", "MISSING"}


def test_generic_field_zero_conf_not_pass():
    # Generic path: name found but no matching box conf → WARNING / LOW_CONFIDENCE
    result = extract_fields(
        "Name: ALICE WONDER",
        [{"text": "OTHER", "confidence": 0.0, "bbox": [0, 0, 5, 5]}],
        "PAN",
    )
    name_fields = [f for f in result["fields"] if f["field"] == "name"]
    if name_fields:
        assert name_fields[0]["validation"] != "PASS"
        assert name_fields[0]["status"] in {"UNCERTAIN", "LOW_CONFIDENCE", "EXTRACTED"}


def test_classifier_to_dict_aliases():
    from app.pipeline.classifier import ClassificationResult

    r = ClassificationResult(
        label="PASSPORT",
        confidence=0.81,
        sih_category="PASSPORT",
        alternatives=[],
        method="test",
        features={},
        profile="passport",
        model_version="v1",
    )
    d = r.to_dict()
    assert d["document_type"] == "passport"
    assert d["document_type_confidence"] == 0.81
    assert d["label"] == "PASSPORT"
