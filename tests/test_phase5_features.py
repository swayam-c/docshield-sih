"""Phase 5 vision feature extraction tests (mock backends by default)."""

from __future__ import annotations

import io

from PIL import Image, ImageDraw


def _png(w=1000, h=640, color=(230, 230, 225)) -> bytes:
    img = Image.new("RGB", (w, h), color=color)
    draw = ImageDraw.Draw(img)
    for y in range(40, h - 40, 30):
        draw.rectangle([40, y, w - 40, y + 12], fill=(25, 25, 25))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_mock_feature_extraction(monkeypatch):
    monkeypatch.setenv("DOCSHIELD_ML_MODE", "mock")
    from app.ml.device import get_device
    from app.ml.registry import clear_registry
    from app.ml.extract import extract_vision_features

    clear_registry()
    get_device.cache_clear()

    result = extract_vision_features(_png())
    assert result["status"] == "OK"
    assert result["efficientnet"]["status"] == "FEATURES_EXTRACTED"
    assert result["vit"]["status"] == "FEATURES_EXTRACTED"
    assert result["clip"]["status"] == "FEATURES_EXTRACTED"
    assert result["efficientnet"]["concern"] == "NOT_SCORED"
    assert result["fusion"]["status"] in {"DEFERRED_TO_PIPELINE", "NOT_IMPLEMENTED", "FUSED"}
    assert "NOT a fraud" in result["efficientnet"]["note"] or "not" in result["note"].lower()


def test_features_endpoint(client):
    res = client.post(
        "/api/features",
        files={"file": ("card.png", io.BytesIO(_png()), "image/png")},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "OK"
    assert body["efficientnet"]["embedding_dim"] == 1280
    assert body["vit"]["embedding_dim"] == 768
    assert body["clip"]["embedding_dim"] == 512
    assert body["efficientnet"]["concern"] == "NOT_SCORED"


def test_analyze_includes_ai_forensics(client):
    # Use a large enough image so quality does not force INCONCLUSIVE skip
    res = client.post(
        "/api/analyze",
        files={"file": ("doc.png", io.BytesIO(_png()), "image/png")},
    )
    assert res.status_code == 200
    body = res.json()
    ai = body["evidence"]["ai_forensics"]
    assert ai["status"] in {"OK", "PARTIAL", "SKIPPED"}
    if body["assessment"]["decision"] != "INCONCLUSIVE":
        assert ai["efficientnet"]["status"] == "FEATURES_EXTRACTED"
        assert ai["vit"]["status"] == "FEATURES_EXTRACTED"
        assert ai["clip"]["status"] == "FEATURES_EXTRACTED"
    auth = body["assessment"]["authenticity_estimate"]
    assert auth is None or (0.0 <= float(auth) <= 1.0)
    assert "model_agreement" in body["assessment"]


def test_health_phase5(client):
    body = client.get("/api/health").json()
    assert "phase" in body["phase"]
    assert "working" in body["components"]["efficientnet"]
    assert "working" in body["components"]["vit"]
    assert "working" in body["components"]["clip"]
    assert "working" in body["components"]["fusion"] or body["components"]["fusion"] == "not_implemented"
