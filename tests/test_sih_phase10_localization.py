"""SIH Phase 10 — tampering localization / heatmaps."""

from __future__ import annotations

import base64
import io

from PIL import Image, ImageDraw


def _png(w=640, h=420, patch=False) -> bytes:
    img = Image.new("RGB", (w, h), color=(230, 230, 225))
    draw = ImageDraw.Draw(img)
    for y in range(40, h - 40, 28):
        draw.rectangle([40, y, w - 40, y + 10], fill=(30, 30, 30))
    if patch:
        draw.rectangle([w // 2, h // 3, w // 2 + 90, h // 3 + 90], fill=(255, 20, 20))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_build_localization_grid_and_overlay():
    from forensics.localization import build_localization
    from ocr.preprocess import decode_bgr

    bgr = decode_bgr(_png(patch=True))
    loc = build_localization(bgr)
    assert loc["status"] == "LOCALIZED"
    assert loc["concern"] == "HEURISTIC_LOCALIZATION_ONLY"
    assert loc["calibrated"] is False
    assert loc["grid_rows"] == 24 and loc["grid_cols"] == 24
    assert len(loc["grid"]) == 24
    assert len(loc["grid"][0]) == 24
    assert 0.0 <= loc["peak_score"] <= 1.0
    assert loc["overlay_png_base64"]
    raw = base64.b64decode(loc["overlay_png_base64"])
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"


def test_tampering_includes_localization():
    from forensics.tampering import analyze_tampering

    result = analyze_tampering(_png(patch=True))
    loc = result["localization"]
    assert loc["status"] == "LOCALIZED"
    assert isinstance(loc.get("hotspots"), list)
    assert "authenticity_estimate" not in result


def test_pipeline_localization_phase_status(monkeypatch):
    monkeypatch.setenv("DOCSHIELD_ML_MODE", "mock")
    from app.pipeline import run_pipeline

    result = run_pipeline(_png(), filename="doc.png")
    auth = result["assessment"]["authenticity_estimate"]
    assert auth is None or (0.0 <= float(auth) <= 1.0)
    loc = result["evidence"]["tampering"]["localization"]
    assert loc["status"] == "LOCALIZED"
    ps = result["screening_result"]["phase_status"]
    assert ps["10_localization"] == "LOCALIZED"
    assert "17_decision" in ps


def test_health_sih_phase10(client):
    body = client.get("/api/health").json()
    assert "sih-phase-" in body["phase"]
    assert "heatmap" in body["components"]["tampering_localization"]
    assert body["version"].startswith("0.") and "sih-phase" in body["version"]


def test_analyze_returns_overlay(client, png_bytes):
    res = client.post(
        "/api/analyze",
        files={"file": ("doc.png", png_bytes, "image/png")},
    )
    assert res.status_code == 200
    data = res.json()
    auth = data["assessment"]["authenticity_estimate"]
    assert auth is None or (0.0 <= float(auth) <= 1.0)
    loc = data["evidence"]["tampering"]["localization"]
    assert loc["status"] == "LOCALIZED"
    assert loc.get("overlay_png_base64")
