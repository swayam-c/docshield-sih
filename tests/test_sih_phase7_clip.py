"""SIH Phase 7 — CLIP supporting evidence integration."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageDraw


def _png(w=1000, h=640, color=(230, 230, 225)) -> bytes:
    img = Image.new("RGB", (w, h), color=color)
    draw = ImageDraw.Draw(img)
    for y in range(40, h - 40, 30):
        draw.rectangle([40, y, w - 40, y + 12], fill=(25, 25, 25))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_clip_supporting_evidence_not_authenticity(monkeypatch):
    monkeypatch.setenv("DOCSHIELD_ML_MODE", "mock")
    from app.ml.clip_evidence import build_clip_supporting_evidence
    from app.ml.summarize import mock_embedding

    vec = mock_embedding(512, seed=7)
    evidence = build_clip_supporting_evidence(
        vec, predicted_document_type="PASSPORT", backend="mock"
    )
    assert evidence["status"] == "SUPPORTING_SCORED"
    assert evidence["concern"] == "NOT_AUTHENTICITY"
    assert evidence["calibrated"] is False
    align = evidence["document_type_alignment"]
    assert "top_label" in align
    assert "scores" in align
    assert "authenticity" in align["note"].lower() or "NOT" in align["note"]
    fr = evidence["fusion_ready"]
    assert fr["embedding_dim"] == 512
    assert len(fr["prototype_probability_vector"]) == len(fr["prototype_labels"])


def test_extract_clip_includes_supporting_block(monkeypatch):
    monkeypatch.setenv("DOCSHIELD_ML_MODE", "mock")
    from app.ml.clip_encoder import extract_clip_features
    from app.ml.device import get_device
    from app.ml.registry import clear_registry

    clear_registry()
    get_device.cache_clear()
    result = extract_clip_features(_png(), predicted_document_type="NATIONAL_ID")
    assert result["status"] == "FEATURES_EXTRACTED"
    assert result["supporting_evidence"]["status"] == "SUPPORTING_SCORED"
    assert result["supporting_evidence"]["concern"] == "NOT_AUTHENTICITY"


def test_vision_extract_fusion_inputs_ready(monkeypatch):
    monkeypatch.setenv("DOCSHIELD_ML_MODE", "mock")
    from app.ml.device import get_device
    from app.ml.extract import extract_vision_features
    from app.ml.registry import clear_registry

    clear_registry()
    get_device.cache_clear()
    result = extract_vision_features(_png(), predicted_document_type="VISA")
    assert result["clip_supporting_evidence"]["status"] == "SUPPORTING_SCORED"
    assert result["fusion"]["status"] == "DEFERRED_TO_PIPELINE"
    assert result["fusion"]["inputs_ready"]["clip"] is True
    assert result["fusion"]["inputs_ready"]["clip_prototype_vector"] is True


def test_analyze_clip_alignment_null_authenticity(client):
    res = client.post(
        "/api/analyze",
        files={"file": ("doc.png", io.BytesIO(_png()), "image/png")},
    )
    assert res.status_code == 200
    body = res.json()
    auth = body["assessment"]["authenticity_estimate"]
    assert auth is None or (0.0 <= float(auth) <= 1.0)
    if body["assessment"]["decision"] != "INCONCLUSIVE":
        support = body["evidence"]["ai_forensics"]["clip"]["supporting_evidence"]
        assert support["status"] == "SUPPORTING_SCORED"
        assert body["assessment"]["model_agreement"]["clip"] == "NOT_AUTHENTICITY"
        assert "clip_type_alignment" in body["assessment"]["model_agreement"]


def test_health_sih_phase7(client):
    body = client.get("/api/health").json()
    assert "sih-phase-" in body["phase"]
    assert "type_alignment" in body["components"]["clip_supporting_evidence"]
    assert body["version"].startswith("0.") and "sih-phase" in body["version"]
