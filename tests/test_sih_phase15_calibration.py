"""SIH Phase 15 — probability calibration (synthetic Platt)."""

from __future__ import annotations

import io

from PIL import Image, ImageDraw


def _blank() -> bytes:
    img = Image.new("RGB", (320, 240), color=(200, 200, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _docish() -> bytes:
    img = Image.new("RGB", (1100, 700), color=(245, 245, 240))
    draw = ImageDraw.Draw(img)
    draw.rectangle([40, 40, 1060, 660], outline=(30, 30, 30), width=3)
    draw.text((80, 80), "INCOME TAX DEPARTMENT", fill=(20, 20, 20))
    draw.text((80, 140), "Permanent Account Number ABCDE1234F", fill=(20, 20, 20))
    draw.text((80, 200), "Name: RAHUL SHARMA", fill=(20, 20, 20))
    draw.text((80, 260), "DOB: 12/03/1990", fill=(20, 20, 20))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_platt_maps_probability():
    from evidence.calibration import apply_platt, ensure_calibration_model, load_head
    from evidence.calibration import WEIGHTS_PATH, META_PATH

    ensure_calibration_model()
    model = load_head(WEIGHTS_PATH, META_PATH)
    assert model is not None
    p = apply_platt(0.7, model)
    assert 0.0 <= p <= 1.0


def test_calibration_skipped_inconclusive():
    from evidence.calibration import calibrate_fusion_score

    result = calibrate_fusion_score(
        fusion={"status": "FUSED", "authentic_probability": 0.6},
        evidence_quality_ok=False,
        decision="INCONCLUSIVE",
    )
    assert result["status"] == "SKIPPED"
    assert result["authenticity_estimate"] is None
    assert result["production_calibrated"] is False


def test_calibration_on_fused_score():
    from evidence.calibration import calibrate_fusion_score

    result = calibrate_fusion_score(
        fusion={"status": "FUSED", "authentic_probability": 0.62},
        evidence_quality_ok=True,
        decision="PENDING",
    )
    assert result["status"] == "CALIBRATED"
    assert result["authenticity_estimate"] is not None
    assert 0.0 <= float(result["authenticity_estimate"]) <= 1.0
    assert result["proxy_calibrated"] is True
    assert result["production_calibrated"] is False
    assert result["concern"] == "SYNTHETIC_PROXY_CALIBRATED"


def test_pipeline_sets_proxy_authenticity_estimate(monkeypatch):
    monkeypatch.setenv("DOCSHIELD_ML_MODE", "mock")
    from app.pipeline import run_pipeline

    result = run_pipeline(_docish(), filename="pan.png")
    cal = result["evidence"]["calibration"]
    assert cal["status"] in {"CALIBRATED", "SKIPPED", "NOT_ASSESSED", "ERROR"}
    assert result["screening_result"]["phase_status"]["15_calibration"] == cal["status"]
    assert "17_decision" in result["screening_result"]["phase_status"]
    auth = result["assessment"]["authenticity_estimate"]
    if cal["status"] == "CALIBRATED":
        assert auth is not None
        assert 0.0 <= float(auth) <= 1.0
        assert cal["production_calibrated"] is False
    else:
        assert auth is None
    assert any("calibration" in step.lower() or "Calibration" in step for step in result["pipeline"])


def test_inconclusive_keeps_null_estimate(monkeypatch):
    monkeypatch.setenv("DOCSHIELD_ML_MODE", "mock")
    from app.pipeline import run_pipeline

    # tiny / blank-ish image often insufficient quality
    result = run_pipeline(_blank(), filename="tiny.png")
    if result["assessment"]["decision"] == "INCONCLUSIVE":
        assert result["assessment"]["authenticity_estimate"] is None
        assert result["evidence"]["calibration"]["status"] == "SKIPPED"


def test_health_sih_phase15(client):
    body = client.get("/api/health").json()
    assert "complete" in body["phase"] or "sih-phase-30" in body["phase"]
    assert "platt" in body["components"]["calibration"]
    assert body["version"].startswith("0.30.")
