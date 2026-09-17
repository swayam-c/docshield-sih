"""SIH Phase 14 — cross-validation engine (review signals only)."""

from __future__ import annotations

import io

from PIL import Image


def _blank() -> bytes:
    img = Image.new("RGB", (320, 240), color=(200, 200, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_cross_validation_unit_quality_fail():
    from evidence.cross_validation import run_cross_validation

    result = run_cross_validation(
        quality={"insufficient_for_analysis": True, "overall_quality": 0.1},
        document_type={"label": "UNKNOWN", "confidence": 0.1},
        ocr={"status": "SKIPPED"},
        fields={},
        mrz={"status": "NOT_APPLICABLE"},
        face={"status": "NOT_DETECTED", "liveness": {"status": "NOT_APPLICABLE"}},
        vision={},
        fusion={"status": "NOT_ASSESSED"},
        tampering={"status": "NOT_ASSESSED"},
        profile_supports_mrz=False,
        profile_expects_photo=False,
    )
    assert result["status"] == "COMPLETED"
    assert result["overall"] == "FAIL"
    assert result["calibrated"] is False
    assert result["concern"] == "CROSS_VALIDATION_REVIEW_ONLY"
    assert any(c["item"] == "image_quality" and c["status"] == "FAIL" for c in result["checks"])
    assert "authenticity" not in result


def test_cross_validation_head_agreement():
    from evidence.cross_validation import run_cross_validation

    result = run_cross_validation(
        quality={"insufficient_for_analysis": False, "overall_quality": 0.8},
        document_type={"label": "PAN", "confidence": 0.7},
        ocr={"status": "OK"},
        fields={"field_completeness": 0.6},
        mrz={"status": "NOT_APPLICABLE"},
        face={
            "status": "NOT_DETECTED",
            "liveness": {"status": "NOT_APPLICABLE"},
        },
        vision={
            "efficientnet": {
                "authenticity_head": {
                    "status": "SCORED",
                    "authentic_probability": 0.7,
                }
            },
            "vit": {
                "authenticity_head": {
                    "status": "SCORED",
                    "authentic_probability": 0.2,
                }
            },
            "clip": {
                "supporting_evidence": {
                    "status": "SUPPORTING_SCORED",
                    "document_type_alignment": {
                        "agreement": "ALIGNED",
                        "top_label": "PAN",
                    },
                }
            },
        },
        fusion={"status": "FUSED", "concern": "UNCALIBRATED_FUSION"},
        tampering={"status": "ANALYZED", "severity": "LOW"},
        profile_supports_mrz=False,
        profile_expects_photo=False,
    )
    assert result["status"] == "COMPLETED"
    assert any(c["item"] == "head_agreement" and c["status"] == "WARNING" for c in result["checks"])
    assert any(c["item"] == "clip_type_alignment" and c["status"] == "PASS" for c in result["checks"])


def test_pipeline_includes_cross_validation(monkeypatch):
    monkeypatch.setenv("DOCSHIELD_ML_MODE", "mock")
    from app.pipeline import run_pipeline

    result = run_pipeline(_blank(), filename="doc.png")
    cv = result["evidence"]["cross_validation"]
    assert cv["status"] in {"COMPLETED", "ERROR"}
    assert result["screening_result"]["phase_status"]["14_cross_validation"] == cv["status"]
    assert "17_decision" in result["screening_result"]["phase_status"]
    auth = result["assessment"]["authenticity_estimate"]
    assert auth is None or (0.0 <= float(auth) <= 1.0)
    assert any("Cross-validation" in step for step in result["pipeline"])
    cons = result["evidence"]["consistency"]
    assert "cross_validation" in cons


def test_health_sih_phase14(client):
    body = client.get("/api/health").json()
    # Phase 15+ supersedes health `phase` string; cross-validation component remains.
    assert "rule_engine" in body["components"]["cross_validation"]
    assert "platt" in body["components"]["calibration"]
