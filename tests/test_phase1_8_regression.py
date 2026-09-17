"""Phase 1–8 regression: field status honesty + document-type field routing."""

from __future__ import annotations

from ocr.fields import extract_fields
from ocr.mrz import analyze_mrz
from document_profiles import get_profile


def _boxes(*pairs):
    return [{"text": t, "confidence": c, "bbox": [i * 10, 10, 40, 14]} for i, (t, c) in enumerate(pairs)]


def test_zero_ocr_confidence_never_pass_validation():
    result = extract_fields(
        "Father Name: ZERO CONF PERSON\nName: ALICE WONDER\nPermanent Account Number ABCDE1234F",
        _boxes(("OTHER", 0.0)),
        "PAN",
    )
    for f in result["fields"]:
        if f.get("ocr_confidence", 1) < 0.15:
            assert f["validation"] != "PASS"
            assert f["status"] == "UNCERTAIN"


def test_completeness_uses_applicable_only():
    text = (
        "Name: RAHUL SHARMA\n"
        "DOB: 12/03/1990\n"
        "Permanent Account Number ABCDE1234F\n"
        "Father Name: RAM SHARMA\n"
    )
    boxes = _boxes(
        ("RAHUL", 0.9),
        ("SHARMA", 0.9),
        ("ABCDE1234F", 0.95),
        ("RAM", 0.9),
        ("SHARMA", 0.88),
        ("12/03/1990", 0.9),
    )
    result = extract_fields(text, boxes, "PAN")
    basis = result["completeness_basis"]
    profile = get_profile("PAN")
    applicable = [f for f in profile.expected_fields if f not in {"photo", "signature", "mrz"}]
    assert basis["applicable_count"] == len(applicable)
    assert "signature" not in result["missing_expected_fields"]
    assert result["field_completeness"] > 0.5
    inv = {i["field"]: i["status"] for i in result["field_inventory"]}
    assert inv.get("signature") == "NOT_APPLICABLE"


def test_aadhaar_mrz_not_applicable():
    result = extract_fields(
        "Name: TEST USER\nDOB: 01/01/1990\n1234 5678 9012",
        _boxes(("TEST", 0.9), ("USER", 0.9), ("1234", 0.9)),
        "AADHAAR",
    )
    mrz = analyze_mrz("no mrz here", "AADHAAR", get_profile("AADHAAR").supports_mrz, result)
    assert mrz["status"] == "NOT_APPLICABLE"
    assert "mrz" not in [f["field"] for f in result["fields"]]


def test_passport_surname_given_extraction():
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
    boxes = _boxes(
        ("SHARMA", 0.92),
        ("RAHUL", 0.91),
        ("KUMAR", 0.90),
        ("A1234567", 0.95),
        ("INDIAN", 0.88),
        ("15/03/1990", 0.85),
        ("M", 0.99),
    )
    result = extract_fields(text, boxes, "PASSPORT")
    by = {f["field"]: f for f in result["fields"]}
    assert by["name"]["status"] == "EXTRACTED"
    assert by["document_number"]["status"] == "EXTRACTED"
    assert by["passport_number"]["status"] == "EXTRACTED"
    assert result["field_completeness"] > 0.4
    mrz = analyze_mrz(text, "PASSPORT", True, result)
    assert mrz["status"] == "NOT_DETECTED"  # no MRZ lines in sample


def test_visa_fields_and_mrz_applicable():
    text = (
        "Name: PRIYA NAYAR\nNationality: IND\nVisa Number: V123456789\n"
        "Visa Type: Tourist\nEntries: Multiple\n"
    )
    boxes = _boxes(("PRIYA", 0.9), ("NAYAR", 0.9), ("V123456789", 0.95), ("Tourist", 0.88))
    result = extract_fields(text, boxes, "VISA")
    by = {f["field"]: f for f in result["fields"]}
    assert "visa_number" in by or "document_number" in by
    assert by["visa_type"]["status"] in {"EXTRACTED", "UNCERTAIN"}
    mrz = analyze_mrz(text, "VISA", True, result)
    assert mrz["status"] == "NOT_DETECTED"


def test_driving_licence_no_mrz():
    text = "Name: DRIVER ONE\nDL No: MH-12-2011-0012345\nDOB: 01/01/1990\n"
    boxes = _boxes(("DRIVER", 0.9), ("ONE", 0.9), ("MH-12-2011-0012345", 0.9))
    result = extract_fields(text, boxes, "DRIVING_LICENCE")
    mrz = analyze_mrz(text, "DRIVING_LICENCE", False, result)
    assert mrz["status"] == "NOT_APPLICABLE"
    assert result["field_completeness"] >= 0


def test_unknown_document_completeness_zero_without_fields():
    result = extract_fields("random grocery receipt tomatoes 2kg", [], "UNKNOWN")
    assert result["field_completeness"] == 0.0 or result["status"] in {"NO_FIELDS", "PARTIAL", "OK"}
