"""SIH Phase 13 — passive face liveness / PAD heuristics."""

from __future__ import annotations

import io

from PIL import Image, ImageDraw


def _blank() -> bytes:
    img = Image.new("RGB", (320, 240), color=(200, 200, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _flat_screenish() -> bytes:
    """Bright, low-texture frame (spoof-leaning cues without needing a real face)."""
    img = Image.new("RGB", (640, 480), color=(250, 250, 252))
    draw = ImageDraw.Draw(img)
    # faint grid (moire-ish)
    for x in range(0, 640, 8):
        draw.line([(x, 0), (x, 480)], fill=(235, 235, 240))
    for y in range(0, 480, 8):
        draw.line([(0, y), (640, y)], fill=(235, 235, 240))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_document_source_not_applicable():
    from verification.liveness import assess_passive_liveness

    result = assess_passive_liveness(_blank(), source="document")
    assert result["status"] == "NOT_APPLICABLE"
    assert result["decision"] == "NOT_APPLICABLE"
    assert result["calibrated"] is False


def test_live_without_face_inconclusive():
    from verification.liveness import assess_passive_liveness

    result = assess_passive_liveness(_blank(), source="live_camera")
    assert result["status"] in {"NOT_ASSESSED", "PASSIVE_ASSESSED"}
    assert result["decision"] == "INCONCLUSIVE"
    assert result["concern"] == "UNCALIBRATED_PASSIVE_PAD"
    assert result["calibrated"] is False


def test_live_returns_bounded_signals():
    from verification.liveness import assess_passive_liveness

    result = assess_passive_liveness(_flat_screenish(), source="live_camera")
    assert result["status"] in {"PASSIVE_ASSESSED", "NOT_ASSESSED"}
    if result["status"] == "PASSIVE_ASSESSED":
        assert 0.0 <= result["score_spoof_leaning"] <= 1.0
        assert result["decision"] in {
            "SUSPECT_PRESENTATION_ATTACK",
            "PASSIVE_OK_UNCALIBRATED",
            "INCONCLUSIVE",
        }
    assert "authenticity" not in result


def test_pipeline_document_liveness_na(monkeypatch):
    monkeypatch.setenv("DOCSHIELD_ML_MODE", "mock")
    from app.pipeline import run_pipeline

    result = run_pipeline(_blank(), filename="doc.png")
    live = result["evidence"]["face"]["liveness"]
    assert live["status"] == "NOT_APPLICABLE"
    assert result["screening_result"]["phase_status"]["13_liveness"] == "NOT_APPLICABLE"
    assert "17_decision" in result["screening_result"]["phase_status"]
    auth = result["assessment"]["authenticity_estimate"]
    assert auth is None or (0.0 <= float(auth) <= 1.0)


def test_camera_capture_runs_passive_pad(client, png_bytes):
    start = client.post("/api/camera/session/start").json()
    sid = start["session_id"]
    res = client.post(
        f"/api/camera/capture?session_id={sid}",
        files={"file": ("cam.png", png_bytes, "image/png")},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["liveness"]["status"] in {"PASSIVE_ASSESSED", "NOT_ASSESSED", "ERROR"}
    assert body["liveness"].get("calibrated") in {False, None}
    client.post(f"/api/camera/session/end?session_id={sid}")


def test_face_liveness_api(client, png_bytes):
    res = client.post(
        "/api/face/liveness?source=document",
        files={"file": ("d.png", png_bytes, "image/png")},
    )
    assert res.status_code == 200
    assert res.json()["status"] == "NOT_APPLICABLE"

    res2 = client.post(
        "/api/face/liveness?source=live_camera",
        files={"file": ("c.png", png_bytes, "image/png")},
    )
    assert res2.status_code == 200
    assert res2.json()["concern"] == "UNCALIBRATED_PASSIVE_PAD"


def test_health_sih_phase13(client):
    body = client.get("/api/health").json()
    # Phase 14+ supersedes the health `phase` string; liveness component remains.
    assert "passive" in body["components"]["face_liveness"]
    assert "cross_validation" in body["components"]
