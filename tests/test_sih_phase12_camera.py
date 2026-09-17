"""SIH Phase 12 — live camera capture sessions."""

from __future__ import annotations

import io

from PIL import Image


def _png() -> bytes:
    img = Image.new("RGB", (320, 240), color=(180, 180, 180))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_camera_session_lifecycle():
    from verification.camera import (
        analyze_camera_frame,
        end_camera_session,
        start_camera_session,
    )

    sess = start_camera_session(case_id="CASE-TEST")
    assert sess["status"] == "READY"
    assert sess["liveness"]["status"] == "READY"
    sid = sess["session_id"]

    captured = analyze_camera_frame(_png(), session_id=sid, compare_to_document=False)
    assert captured["status"] == "CAPTURED"
    assert captured["camera"]["status"] == "CAPTURED"
    assert captured["liveness"]["status"] in {
        "PASSIVE_ASSESSED",
        "NOT_ASSESSED",
        "ERROR",
    }
    assert captured["liveness"].get("concern") in {
        "UNCALIBRATED_PASSIVE_PAD",
        "NOT_ASSESSED",
    }

    closed = end_camera_session(sid)
    assert closed["status"] == "CLOSED"


def test_camera_capture_without_session():
    from verification.camera import analyze_camera_frame

    result = analyze_camera_frame(_png())
    assert result["status"] == "CAPTURED"
    assert result["liveness"]["status"] in {"PASSIVE_ASSESSED", "NOT_ASSESSED", "ERROR"}


def test_camera_attach_and_compare():
    from verification.camera import (
        analyze_camera_frame,
        attach_document_reference,
        end_camera_session,
        start_camera_session,
    )

    sess = start_camera_session()
    sid = sess["session_id"]
    assert attach_document_reference(sid, _png())["status"] == "ATTACHED"
    result = analyze_camera_frame(_png(), session_id=sid, compare_to_document=True)
    assert result["status"] == "CAPTURED"
    # Compare may be NOT_ASSESSED if no face detected on synthetic blank
    assert result["face"]["verification"]["status"] in {
        "NOT_ASSESSED",
        "COMPARED_UNCALIBRATED",
    }
    end_camera_session(sid)


def test_camera_api_status(client):
    res = client.get("/api/camera/status")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "READY"
    assert body["liveness"] == "working_passive_heuristic_v1"


def test_camera_api_capture(client, png_bytes):
    start = client.post("/api/camera/session/start")
    assert start.status_code == 200
    sid = start.json()["session_id"]
    res = client.post(
        f"/api/camera/capture?session_id={sid}",
        files={"file": ("cam.png", png_bytes, "image/png")},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "CAPTURED"
    assert body["liveness"]["status"] in {"PASSIVE_ASSESSED", "NOT_ASSESSED", "ERROR"}
    end = client.post(f"/api/camera/session/end?session_id={sid}")
    assert end.status_code == 200


def test_health_sih_phase12(client):
    body = client.get("/api/health").json()
    assert "sih-phase-" in body["phase"]
    assert "ephemeral" in body["components"]["live_camera"]
    assert "passive" in body["components"]["face_liveness"]
    assert body["version"].startswith("0.") and "sih-phase" in body["version"]


def test_pipeline_marks_camera_not_used(monkeypatch, png_bytes=None):
    monkeypatch.setenv("DOCSHIELD_ML_MODE", "mock")
    from app.pipeline import run_pipeline

    result = run_pipeline(_png(), filename="doc.png")
    assert result["evidence"]["face"]["camera"]["status"] == "NOT_USED"
    assert result["screening_result"]["phase_status"]["12_camera"] == "NOT_USED"
    assert "17_decision" in result["screening_result"]["phase_status"]
    auth = result["assessment"]["authenticity_estimate"]
    assert auth is None or (0.0 <= float(auth) <= 1.0)
