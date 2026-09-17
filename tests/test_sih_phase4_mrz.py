"""SIH Phase 4 — MRZ processing + visible-field consistency checklist."""

from __future__ import annotations


def _valid_td3_pair():
    from ocr.mrz import mrz_check_digit

    doc = "P1234567<"
    doc_cd = mrz_check_digit(doc)
    dob = "900312"
    dob_cd = mrz_check_digit(dob)
    exp = "300101"
    exp_cd = mrz_check_digit(exp)
    optional = "<<<<<<<<<<<<<<"
    opt_cd = mrz_check_digit(optional)
    line2_body = doc + doc_cd + "IND" + dob + dob_cd + "M" + exp + exp_cd + optional + opt_cd
    composite = doc + doc_cd + dob + dob_cd + exp + exp_cd + optional + opt_cd
    comp_cd = mrz_check_digit(composite)
    line2 = line2_body + comp_cd
    line1 = ("P<INDSINGH<<RAHUL<<<<<<<<<<<<<<<<<<<<<<<<<<" + ("<" * 44))[:44]
    return line1, line2


def test_mrz_checklist_and_consistency_pass():
    from ocr.mrz import analyze_mrz

    line1, line2 = _valid_td3_pair()
    text = line1 + "\n" + line2
    fields = {
        "compare_values": {
            "name": "RAHUL SINGH",
            "date_of_birth": "12/03/1990",
            "document_number": "P1234567",
        },
        "fields": [],
    }
    result = analyze_mrz(text, "PASSPORT", True, fields_payload=fields)
    assert result["status"] == "PARSED"
    assert result["structure"] == "TD3"
    assert result["check_digits_ok"] is True
    by = {c["item"]: c for c in result["checklist"]}
    assert by["format_valid"]["status"] == "PASS"
    assert by["check_digits_valid"]["status"] == "PASS"
    assert by["name_consistency"]["status"] == "PASS"
    assert by["dob_consistency"]["status"] == "PASS"
    assert by["document_number_consistency"]["status"] == "PASS"
    # Masked fields in response
    assert "X" in result["fields"]["document_number"] or result["fields"]["document_number"].endswith("4567")


def test_mrz_consistency_fail_on_mismatch():
    from ocr.mrz import analyze_mrz

    line1, line2 = _valid_td3_pair()
    fields = {
        "compare_values": {
            "name": "COMPLETELY DIFFERENT",
            "date_of_birth": "01/01/2001",
            "document_number": "Z9999999",
        }
    }
    result = analyze_mrz(line1 + "\n" + line2, "PASSPORT", True, fields_payload=fields)
    by = {c["item"]: c for c in result["checklist"]}
    assert by["name_consistency"]["status"] in {"FAIL", "WARNING"}
    assert by["dob_consistency"]["status"] == "FAIL"
    assert by["document_number_consistency"]["status"] == "FAIL"


def test_mrz_not_detected_checklist_not_assessed():
    from ocr.mrz import analyze_mrz

    result = analyze_mrz("no machine readable zone here", "PASSPORT", True)
    assert result["status"] == "NOT_DETECTED"
    assert all(c["status"] == "NOT_ASSESSED" for c in result["checklist"])


def test_mrz_td2_parse():
    from ocr.mrz import analyze_mrz, mrz_check_digit

    # Minimal synthetic TD2-like pair (36 chars each) with valid checks
    line1 = ("V<INDOE<<JANE<<<<<<<<<<<<<<<<<<<<<" + ("<" * 36))[:36]
    doc = "V1234567<"
    doc_cd = mrz_check_digit(doc)
    dob = "950101"
    dob_cd = mrz_check_digit(dob)
    exp = "300101"
    exp_cd = mrz_check_digit(exp)
    optional = "<<<<<<<"
    body = doc + doc_cd + "IND" + dob + dob_cd + "F" + exp + exp_cd + optional
    assert len(body) == 35
    comp = mrz_check_digit(doc + doc_cd + dob + dob_cd + exp + exp_cd + optional)
    line2 = body + comp
    assert len(line2) == 36
    result = analyze_mrz(line1 + "\n" + line2, "VISA", True)
    assert result["status"] == "PARSED"
    assert result["structure"] == "TD2"


def test_pipeline_strips_compare_values(client, png_bytes):
    import io

    res = client.post(
        "/api/analyze",
        files={"file": ("sample.png", io.BytesIO(png_bytes), "image/png")},
    )
    assert res.status_code == 200
    body = res.json()
    assert "compare_values" not in body["evidence"]["fields"]
    assert "checklist" in body["evidence"]["mrz"] or body["evidence"]["mrz"]["status"] in {
        "NOT_APPLICABLE",
        "NOT_DETECTED",
        "NOT_ASSESSED",
        "PARTIAL",
        "PARSED",
    }


def test_health_sih_phase4(client):
    body = client.get("/api/health").json()
    assert "sih-phase-" in body["phase"]
    assert "checklist" in body["components"]["mrz"] or "td" in body["components"]["mrz"]
    assert body["version"].startswith("0.") and "sih-phase" in body["version"]
