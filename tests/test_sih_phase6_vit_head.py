"""SIH Phase 6 — ViT authenticity head."""

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


def test_train_and_score_vit_head(monkeypatch):
    monkeypatch.setenv("DOCSHIELD_ML_MODE", "mock")
    from app.ml.authenticity_head import clear_head_cache
    from app.ml.summarize import mock_embedding
    from app.ml.vit_head import score_vit_authenticity, train_and_save_vit_head

    clear_head_cache()
    meta = train_and_save_vit_head(samples_per_class=40)
    assert meta["dataset"].startswith("synthetic")
    assert meta["calibrated"] is False
    assert meta["embedding_dim"] == 768
    assert "not on real government" in meta["note"].lower()

    vec = mock_embedding(768, seed=99)
    scored = score_vit_authenticity(vec)
    assert scored["status"] == "SCORED"
    assert 0.0 <= scored["authentic_probability"] <= 1.0
    assert abs(scored["authentic_probability"] + scored["tampered_probability"] - 1.0) < 1e-3
    assert scored["concern"] == "UNCALIBRATED_SYNTHETIC"


def test_extract_vit_includes_head(monkeypatch):
    monkeypatch.setenv("DOCSHIELD_ML_MODE", "mock")
    from app.ml.device import get_device
    from app.ml.registry import clear_registry
    from app.ml.vit import extract_vit_features

    clear_registry()
    get_device.cache_clear()
    result = extract_vit_features(_png())
    assert result["status"] == "FEATURES_EXTRACTED"
    assert result["concern"] == "NOT_SCORED"
    assert result["authenticity_head"]["status"] == "SCORED"


def test_analyze_vit_head_and_null_authenticity(client):
    res = client.post(
        "/api/analyze",
        files={"file": ("doc.png", io.BytesIO(_png()), "image/png")},
    )
    assert res.status_code == 200
    body = res.json()
    auth = body["assessment"]["authenticity_estimate"]
    assert auth is None or (0.0 <= float(auth) <= 1.0)
    if body["assessment"]["decision"] != "INCONCLUSIVE":
        vit_head = body["evidence"]["ai_forensics"]["vit"]["authenticity_head"]
        assert vit_head["status"] == "SCORED"
        assert body["assessment"]["model_agreement"]["vit"] == "UNCALIBRATED_SYNTHETIC"


def test_health_sih_phase6(client):
    body = client.get("/api/health").json()
    assert "sih-phase-" in body["phase"]
    assert "synthetic" in body["components"]["vit_authenticity_head"]
    assert body["version"].startswith("0.") and "sih-phase" in body["version"]


def test_vit_head_rejects_wrong_dim():
    from app.ml.vit_head import score_vit_authenticity

    assert score_vit_authenticity(np.zeros(1280))["status"] == "NOT_ASSESSED"
