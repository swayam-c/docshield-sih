"""SIH Phase 9 — tampering heuristics."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageDraw


def _png(w=640, h=420, color=(230, 230, 225), draw_patch=False) -> bytes:
    img = Image.new("RGB", (w, h), color=color)
    draw = ImageDraw.Draw(img)
    for y in range(40, h - 40, 28):
        draw.rectangle([40, y, w - 40, y + 10], fill=(30, 30, 30))
    if draw_patch:
        # Paste a mismatched bright patch (forces residual/ELA energy)
        draw.rectangle([w // 2, h // 3, w // 2 + 80, h // 3 + 80], fill=(255, 40, 40))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_analyze_tampering_returns_signals():
    from forensics.tampering import analyze_tampering

    result = analyze_tampering(_png())
    assert result["status"] == "ANALYZED"
    assert result["concern"] == "HEURISTIC_REVIEW_ONLY"
    assert result["calibrated"] is False
    assert "ela_inconsistency" in result["signals"]
    assert "noise_inconsistency" in result["signals"]
    assert "copy_move_indicators" in result["signals"]
    assert result["localization"]["status"] == "LOCALIZED"
    assert result["localization"]["concern"] == "HEURISTIC_LOCALIZATION_ONLY"
    assert 0.0 <= result["review_score"] <= 1.0
    for key in (
        "ela_inconsistency",
        "compression_inconsistency",
        "noise_inconsistency",
        "edge_inconsistency",
        "local_image_anomaly",
        "copy_move_indicators",
    ):
        v = result["signals"][key]
        assert v == v and v not in (float("inf"), float("-inf"))
        assert 0.0 <= v <= 1.0


def test_tampering_does_not_claim_authenticity():
    from forensics.tampering import analyze_tampering

    result = analyze_tampering(_png(draw_patch=True))
    assert "authenticity" not in result
    assert result.get("note")
    assert "proof of fraud" in result["note"].lower() or "review" in result["note"].lower()


def test_pipeline_includes_tampering(monkeypatch):
    monkeypatch.setenv("DOCSHIELD_ML_MODE", "mock")
    from app.pipeline import run_pipeline

    result = run_pipeline(_png(), filename="doc.png")
    tamp = result["evidence"]["tampering"]
    assert tamp["status"] == "ANALYZED"
    auth = result["assessment"]["authenticity_estimate"]
    assert auth is None or (0.0 <= float(auth) <= 1.0)
    assert result["screening_result"]["phase_status"]["9_tampering"] == "ANALYZED"
    assert result["screening_result"]["phase_status"]["10_localization"] == "LOCALIZED"
    assert "17_decision" in result["screening_result"]["phase_status"]
    assert any("Tampering heuristics" in r for r in result["assessment"]["reasons"])


def test_tamper_api(client, png_bytes):
    res = client.post(
        "/api/tamper",
        files={"file": ("t.png", png_bytes, "image/png")},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ANALYZED"
    assert body["concern"] == "HEURISTIC_REVIEW_ONLY"


def test_health_sih_phase9(client):
    body = client.get("/api/health").json()
    assert "sih-phase-" in body["phase"]
    assert "heuristic" in body["components"]["tampering"]
    assert "heatmap" in body["components"]["tampering_localization"]
    assert body["version"].startswith("0.") and "sih-phase" in body["version"]


def test_analyze_keeps_null_authenticity_with_tampering(client, png_bytes):
    res = client.post(
        "/api/analyze",
        files={"file": ("doc.png", png_bytes, "image/png")},
    )
    assert res.status_code == 200
    data = res.json()
    auth = data["assessment"]["authenticity_estimate"]
    assert auth is None or (0.0 <= float(auth) <= 1.0)
    tamp = data["evidence"]["tampering"]
    assert tamp["status"] == "ANALYZED"
    assert tamp["localization"]["status"] == "LOCALIZED"
