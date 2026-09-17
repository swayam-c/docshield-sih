"""SIH Phase 8 — multimodal feature fusion."""

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


def test_train_and_fuse(monkeypatch):
    monkeypatch.setenv("DOCSHIELD_ML_MODE", "mock")
    from app.ml.authenticity_head import clear_head_cache
    from evidence.fusion import (
        FUSION_DIM,
        build_fusion_vector,
        fuse_evidence,
        train_and_save_fusion,
    )

    clear_head_cache()
    meta = train_and_save_fusion(samples_per_class=40)
    assert meta["architecture"] == "logistic_regression"
    assert meta["calibrated"] is False
    assert meta["embedding_dim"] == FUSION_DIM
    assert "not on real government" in meta["note"].lower()

    vision = {
        "efficientnet": {
            "status": "FEATURES_EXTRACTED",
            "fingerprint": [0.1] * 12,
            "authenticity_head": {
                "status": "SCORED",
                "authentic_probability": 0.7,
                "tampered_probability": 0.3,
            },
        },
        "vit": {
            "status": "FEATURES_EXTRACTED",
            "fingerprint": [0.05] * 12,
            "authenticity_head": {
                "status": "SCORED",
                "authentic_probability": 0.65,
                "tampered_probability": 0.35,
            },
        },
        "clip": {
            "status": "FEATURES_EXTRACTED",
            "fingerprint": [0.02] * 16,
            "supporting_evidence": {
                "fusion_ready": {
                    "fingerprint": [0.02] * 16,
                    "prototype_probability_vector": [0.4, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                }
            },
        },
    }
    vec, vmeta = build_fusion_vector(
        vision_features=vision,
        evidence_quality=0.8,
        ocr_confidence=0.7,
        field_completeness=0.5,
        doc_type_confidence=0.6,
        mrz={"status": "NOT_APPLICABLE"},
    )
    assert vec.shape[0] == FUSION_DIM
    assert vmeta["modalities_present"]["efficientnet"] is True

    fused = fuse_evidence(
        vision_features=vision,
        evidence_quality=0.8,
        ocr_confidence=0.7,
        field_completeness=0.5,
        doc_type_confidence=0.6,
    )
    assert fused["status"] == "FUSED"
    assert fused["calibrated"] is False
    assert fused["concern"] == "UNCALIBRATED_SYNTHETIC"
    assert abs(fused["authentic_probability"] + fused["tampered_probability"] - 1.0) < 1e-3


def test_analyze_includes_fusion_null_authenticity(client):
    res = client.post(
        "/api/analyze",
        files={"file": ("doc.png", io.BytesIO(_png()), "image/png")},
    )
    assert res.status_code == 200
    body = res.json()
    auth = body["assessment"]["authenticity_estimate"]
    assert auth is None or (0.0 <= float(auth) <= 1.0)
    if body["assessment"]["decision"] != "INCONCLUSIVE":
        fusion = body["evidence"]["fusion"]
        assert fusion["status"] == "FUSED"
        assert body["assessment"]["model_agreement"]["fusion"] == "UNCALIBRATED_SYNTHETIC"
        assert "fusion_authentic_probability" in body["assessment"]["model_agreement"]


def test_health_sih_phase8(client):
    body = client.get("/api/health").json()
    assert "sih-phase-" in body["phase"]
    assert "synthetic" in body["components"]["fusion"]
    assert body["version"].startswith("0.") and "sih-phase" in body["version"]


def test_fusion_not_assessed_without_vision():
    from evidence.fusion import fuse_evidence

    result = fuse_evidence(vision_features={}, evidence_quality=0.9)
    assert result["status"] == "NOT_ASSESSED"
