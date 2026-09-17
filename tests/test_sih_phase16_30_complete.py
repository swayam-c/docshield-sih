"""SIH Phases 16–30 — confidence, decision, reports, audit, completion gate."""

from __future__ import annotations

import io

from PIL import Image, ImageDraw


def _docish() -> bytes:
    img = Image.new("RGB", (1100, 700), color=(245, 245, 240))
    draw = ImageDraw.Draw(img)
    draw.rectangle([40, 40, 1060, 660], outline=(30, 30, 30), width=3)
    draw.text((80, 80), "INCOME TAX DEPARTMENT", fill=(20, 20, 20))
    draw.text((80, 140), "Permanent Account Number ABCDE1234F", fill=(20, 20, 20))
    draw.text((80, 200), "Name: RAHUL SHARMA", fill=(20, 20, 20))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_confidence_engine_scores():
    from evidence.confidence import build_confidence_evidence

    ce = build_confidence_evidence(
        evidence_quality=0.8,
        doc_type_confidence=0.7,
        ocr_confidence=0.75,
        field_completeness=0.6,
        fusion={"status": "FUSED", "authentic_probability": 0.7, "head_agreement": "AGREE"},
        cross_validation={"pass_rate": 0.8},
        calibration={"status": "CALIBRATED"},
        tampering={"severity": "LOW"},
        vision_ok={"efficientnet": True, "vit": True, "clip": True},
    )
    assert ce["status"] == "SCORED"
    assert 0.0 <= ce["overall_confidence"] <= 1.0
    assert ce["concern"] == "ASSESSMENT_SUPPORT_ONLY"


def test_decision_inconclusive_preserved():
    from evidence.decision import decide_screening

    d = decide_screening(
        prior_decision="INCONCLUSIVE",
        evidence_quality=0.2,
        authenticity_estimate=None,
        confidence=0.3,
        image_quality={"insufficient_for_analysis": True},
    )
    assert d["decision"] == "INCONCLUSIVE"
    assert d["legal_authenticity"] is False
    assert d["officer_verdict"]["code"] == "INCONCLUSIVE"


def test_decision_high_risk_from_low_auth():
    from evidence.decision import decide_screening

    d = decide_screening(
        prior_decision="PENDING",
        evidence_quality=0.8,
        authenticity_estimate=0.1,
        confidence=0.7,
        fusion={"status": "FUSED", "authentic_probability": 0.1, "tampered_probability": 0.9},
        cross_validation={"overall": "PASS", "counts": {"FAIL": 0, "WARNING": 0}},
        tampering={"severity": "LOW_SIGNAL", "review_score": 0.1},
        calibration={"status": "CALIBRATED"},
    )
    assert d["decision"] == "HIGH-RISK"
    assert d["officer_verdict"]["code"] == "SUSPICIOUS"


def test_decision_face_not_detected_is_inconclusive():
    from evidence.decision import decide_screening

    d = decide_screening(
        prior_decision="PENDING",
        evidence_quality=0.85,
        authenticity_estimate=0.88,
        confidence=0.7,
        fusion={"status": "FUSED", "authentic_probability": 0.7, "tampered_probability": 0.3},
        cross_validation={"overall": "PASS", "counts": {"FAIL": 0, "WARNING": 0}},
        tampering={"severity": "LOW_SIGNAL", "review_score": 0.2, "flags": []},
        calibration={"status": "CALIBRATED", "proxy_calibrated": True},
        face={"status": "NOT_DETECTED", "verification": {"status": "NOT_ASSESSED"}},
    )
    assert d["decision"] == "INCONCLUSIVE"
    assert d["officer_verdict"]["code"] == "INCONCLUSIVE"
    assert d["officer_verdict"]["identity_status"] == "FACE_NOT_DETECTED"
    assert d["officer_verdict"]["document_status"] == "AUTHENTIC"


def test_decision_pass_requires_face_match():
    from evidence.decision import decide_screening

    d = decide_screening(
        prior_decision="PENDING",
        evidence_quality=0.85,
        authenticity_estimate=0.9,
        confidence=0.72,
        fusion={"status": "FUSED", "authentic_probability": 0.75},
        cross_validation={"overall": "PASS", "counts": {"FAIL": 0, "WARNING": 0}},
        tampering={"severity": "LOW_SIGNAL", "review_score": 0.18},
        calibration={"status": "CALIBRATED", "proxy_calibrated": True},
        face={
            "status": "DETECTED",
            "face_count": 1,
            "verification": {
                "status": "COMPARED_UNCALIBRATED",
                "result": "MATCH",
                "decision": "INCONCLUSIVE_HIGH_SIMILARITY",
                "similarity": 0.92,
                "calibration": "UNCALIBRATED",
                "calibrated": False,
                "match_label": "MATCH",
            },
            "reference": {"status": "DETECTED", "face_count": 1},
            "liveness": {"decision": "PASSIVE_OK_UNCALIBRATED", "status": "PASSIVE_ASSESSED"},
        },
    )
    assert d["decision"] == "PASS"
    assert d["officer_verdict"]["label"] == "PASS — AUTHENTIC"
    assert d["officer_verdict"]["identity_status"] == "MATCH"


def test_decision_authentic_no_match_is_review():
    from evidence.decision import decide_screening

    d = decide_screening(
        prior_decision="PENDING",
        evidence_quality=0.85,
        authenticity_estimate=0.88,
        confidence=0.7,
        fusion={"status": "FUSED", "authentic_probability": 0.7},
        cross_validation={"overall": "PASS", "counts": {"FAIL": 0, "WARNING": 0}},
        tampering={"severity": "LOW_SIGNAL", "review_score": 0.15},
        calibration={"status": "CALIBRATED"},
        face={
            "status": "DETECTED",
            "face_count": 1,
            "verification": {
                "status": "COMPARED_UNCALIBRATED",
                "result": "NO_MATCH",
                "decision": "INCONCLUSIVE_LOW_SIMILARITY",
                "similarity": 0.2,
                "calibration": "UNCALIBRATED",
                "calibrated": False,
                "match_label": "NO_MATCH",
            },
            "reference": {"status": "DETECTED", "face_count": 1},
            "liveness": {"decision": "PASSIVE_OK_UNCALIBRATED", "status": "PASSIVE_ASSESSED"},
        },
    )
    assert d["decision"] == "INCONCLUSIVE"
    assert d["officer_verdict"]["code"] == "INCONCLUSIVE"
    assert d["officer_verdict"]["identity_status"] == "NO_MATCH"


def test_decision_tampering_with_match_stays_tampering():
    from evidence.decision import decide_screening

    d = decide_screening(
        prior_decision="PENDING",
        evidence_quality=0.85,
        authenticity_estimate=0.7,
        confidence=0.7,
        fusion={"status": "FUSED", "authentic_probability": 0.6},
        cross_validation={"overall": "WARNING", "counts": {"FAIL": 0, "WARNING": 1}},
        tampering={"severity": "ELEVATED_REVIEW", "review_score": 0.8, "flags": ["ela_inconsistency"]},
        calibration={"status": "CALIBRATED"},
        face={
            "status": "DETECTED",
            "verification": {
                "status": "COMPARED_UNCALIBRATED",
                "result": "MATCH",
                "decision": "INCONCLUSIVE_HIGH_SIMILARITY",
                "similarity": 0.91,
                "calibration": "UNCALIBRATED",
                "calibrated": False,
                "match_label": "MATCH",
            },
            "reference": {"face_count": 1},
            "liveness": {"decision": "PASSIVE_OK_UNCALIBRATED"},
        },
    )
    assert d["decision"] == "HIGH-RISK"
    assert d["officer_verdict"]["code"] == "TAMPERING_INDICATED"
    assert d["officer_verdict"]["identity_status"] == "MATCH"


def test_uncalibrated_match_can_pass():
    from evidence.decision import decide_screening

    d = decide_screening(
        prior_decision="PENDING",
        evidence_quality=0.85,
        authenticity_estimate=0.9,
        confidence=0.72,
        fusion={"status": "FUSED", "authentic_probability": 0.75},
        cross_validation={"overall": "PASS", "counts": {"FAIL": 0, "WARNING": 0}},
        tampering={"severity": "LOW_SIGNAL", "review_score": 0.18},
        calibration={"status": "CALIBRATED", "proxy_calibrated": True},
        face={
            "status": "DETECTED",
            "face_count": 1,
            "verification": {
                "status": "COMPARED_UNCALIBRATED",
                "result": "MATCH",
                "decision": "INCONCLUSIVE_HIGH_SIMILARITY",
                "similarity": 0.918,
                "calibration": "UNCALIBRATED",
                "calibrated": False,
                "match_label": "MATCH",
            },
            "reference": {"status": "DETECTED", "face_count": 1},
            "liveness": {"decision": "PASSIVE_OK_UNCALIBRATED", "status": "PASSIVE_ASSESSED"},
        },
    )
    assert d["decision"] == "PASS"
    assert d["officer_verdict"]["label"] == "PASS — AUTHENTIC"
    assert d["officer_verdict"]["identity_calibration"] == "UNCALIBRATED"


def test_pipeline_decision_and_engines(monkeypatch):
    monkeypatch.setenv("DOCSHIELD_ML_MODE", "mock")
    from app.pipeline import run_pipeline

    result = run_pipeline(_docish(), filename="pan.png")
    assert result["assessment"]["decision"] in {
        "PASS",
        "REVIEW",
        "HIGH-RISK",
        "INCONCLUSIVE",
    }
    assert result["evidence"]["confidence_engine"]["status"] == "SCORED"
    assert result["evidence"]["decision_engine"]["status"] == "DECIDED"
    assert result["evidence"]["explainability"]["status"] == "READY"
    ps = result["screening_result"]["phase_status"]
    assert ps["16_confidence"] == "SCORED"
    assert ps["17_decision"] in {"PASS", "REVIEW", "HIGH-RISK", "INCONCLUSIVE"}
    assert ps["22_blockchain_ready"] == "LOCAL_NOT_CONNECTED"


def test_report_and_audit(client, png_bytes):
    res = client.post(
        "/api/analyze",
        files={"file": ("sample.png", io.BytesIO(png_bytes), "image/png")},
    )
    assert res.status_code == 200
    case_id = res.json()["case_id"]
    assert res.json()["assessment"]["decision"] in {
        "PASS",
        "REVIEW",
        "HIGH-RISK",
        "INCONCLUSIVE",
        "PENDING",
    }

    report = client.get(f"/api/report/{case_id}").json()
    assert report["status"] == "READY"
    assert "DOCSHIELD" in report["markdown"]
    assert report["blockchain"] == "NOT_CONNECTED"

    audit = client.get(f"/api/audit/case/{case_id}").json()
    assert audit["blockchain"] == "NOT_CONNECTED"
    assert any(e["action"] == "ANALYZE" for e in audit["events"])

    verify = client.get("/api/audit/chain/verify").json()
    assert verify["valid"] is True
    assert verify["blockchain"] == "NOT_CONNECTED"

    cases = client.get("/api/cases").json()
    assert any(c["case_id"] == case_id for c in cases)

    health = client.get("/api/health").json()
    assert "sih-phase-30" in health["phase"] or "complete" in health["phase"]
    assert health["version"].startswith("0.30.")
    assert health["components"]["blockchain"] == "NOT_CONNECTED"
    assert "pass_review" in health["components"]["decision_engine"]
