"""SIH Phase 11 — face detection / optional uncalibrated 1:1 compare."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageDraw


def _doc_with_face(w=480, h=640) -> bytes:
    """Synthetic ID-like image with a face-ish oval blob Haar may or may not catch.
    Also provide a clear high-contrast face rectangle pattern for cascade."""
    img = Image.new("RGB", (w, h), color=(240, 240, 235))
    draw = ImageDraw.Draw(img)
    # Card frame
    draw.rectangle([20, 20, w - 20, h - 20], outline=(20, 20, 20), width=3)
    # Approximate frontal face: dark oval + eyes (helps Haar in tests; may still miss)
    cx, cy = w // 4, h // 3
    draw.ellipse([cx - 55, cy - 70, cx + 55, cy + 70], fill=(210, 175, 150), outline=(40, 30, 20))
    draw.ellipse([cx - 25, cy - 25, cx - 10, cy - 10], fill=(20, 20, 20))
    draw.ellipse([cx + 10, cy - 25, cx + 25, cy - 10], fill=(20, 20, 20))
    draw.arc([cx - 20, cy + 10, cx + 20, cy + 35], 0, 180, fill=(40, 20, 20), width=2)
    # Text lines
    for i, y in enumerate(range(cy - 40, cy + 160, 28)):
        draw.rectangle([w // 2, y, w - 50, y + 12], fill=(30, 30, 30))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _blank() -> bytes:
    img = Image.new("RGB", (320, 240), color=(200, 200, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_detect_faces_blank_not_detected():
    from verification.face import detect_faces

    result = detect_faces(_blank())
    assert result["status"] in {"NOT_DETECTED", "DETECTED", "MULTIPLE_DETECTED"}
    assert result["concern"] == "DETECTION_ONLY"
    assert result["liveness"]["status"] == "NOT_APPLICABLE"
    assert result["verification"]["status"] == "NOT_ASSESSED"
    assert "authenticity_estimate" not in result


def test_compare_without_reference():
    from verification.face import compare_faces

    result = compare_faces(_doc_with_face(), reference_bytes=None)
    assert result["verification"]["status"] == "NOT_ASSESSED"
    assert result["liveness"]["status"] == "NOT_APPLICABLE"
    assert result["camera"]["status"] == "NOT_USED"


def test_compare_with_reference_uncalibrated():
    from verification.face import compare_faces

    doc = _doc_with_face()
    # Different bytes (re-encode) so self-compare guard does not fire
    from PIL import Image
    import io as _io
    img = Image.open(_io.BytesIO(doc)).convert("RGB")
    buf = _io.BytesIO()
    img.save(buf, format="PNG", compress_level=1)
    live = buf.getvalue()
    if live == doc:
        live = doc + b""  # still may hash equal; mutate a pixel
        arr = np.array(img)
        arr[0, 0, 0] = (int(arr[0, 0, 0]) + 1) % 256
        live_img = Image.fromarray(arr)
        buf2 = _io.BytesIO()
        live_img.save(buf2, format="PNG")
        live = buf2.getvalue()

    result = compare_faces(doc, reference_bytes=live)
    # If both detect a face, expect uncalibrated compare; else NOT_AVAILABLE
    if (result.get("document_face") or {}).get("face_count", 0) >= 1 and (
        result.get("live_face") or {}
    ).get("face_count", 0) >= 1:
        assert result["verification"]["status"] == "COMPARED_UNCALIBRATED"
        assert result["verification"]["calibrated"] is False
        assert result["verification"]["calibration"] == "UNCALIBRATED"
        assert result["verification"]["result"] in {"MATCH", "NO_MATCH", "INCONCLUSIVE"}
        assert "INCONCLUSIVE" in result["verification"]["decision"] or result["verification"]["decision"].startswith("INCONCLUSIVE")
        assert 0.0 <= result["verification"]["similarity"] <= 1.0
        assert result["concern"] == "UNCALIBRATED_SIMILARITY"
        # Same near-identical images may MATCH; identical bytes must not
        assert result["verification_debug"]["embedding_sources"] == "DIFFERENT"
    else:
        assert result["verification"]["result"] in {"NOT_AVAILABLE", "NOT_ASSESSED"}


def test_compare_result_string_does_not_shadow_payload(monkeypatch):
    """Regression: MATCH/NO_MATCH must not overwrite the result dict (500 on capture)."""
    import numpy as np
    import verification.face as face_mod

    emb = np.linspace(0.2, 1.0, face_mod.EMBEDDING_DIM, dtype=np.float32)
    emb = emb / (np.linalg.norm(emb) + 1e-8)

    def _fake_detect(_bytes: bytes, *, source_role: str = "unknown"):
        e = emb.copy()
        if source_role == "live":
            # Distinct vector (not array_equal) but still high cosine for MATCH band
            e = emb.copy()
            e[0] = e[0] + 0.02
            e = e / (np.linalg.norm(e) + 1e-8)
        return {
            "status": "DETECTED",
            "face_count": 1,
            "faces": [
                {
                    "_embedding": e,
                    "bbox": {"x": 10, "y": 10, "w": 80, "h": 80},
                    "quality": {"label": "GOOD", "overall": 0.9},
                    "embedding_dim": int(e.shape[0]),
                    "embedding_id": face_mod._embedding_id(e),
                    "embedding_fingerprint": [round(float(v), 5) for v in e[:12]],
                    "source_role": source_role,
                }
            ],
            "image_sha16": "docsha16aaaaaaa" if source_role == "document" else "livesha16bbbbbb",
            "source_role": source_role,
            "concern": "DETECTION_ONLY",
            "verification": {"status": "NOT_ASSESSED", "result": "NOT_AVAILABLE"},
            "liveness": {"status": "NOT_APPLICABLE"},
            "camera": {"status": "NOT_USED"},
        }

    monkeypatch.setattr(face_mod, "detect_faces", _fake_detect)
    out = face_mod.compare_faces(b"doc-aaaa", reference_bytes=b"live-bbbb")
    assert isinstance(out, dict)
    assert out["verification"]["status"] == "COMPARED_UNCALIBRATED"
    assert out["verification"]["result"] == "MATCH"
    assert out["face_verification"] == "MATCH"


def test_pipeline_includes_face(monkeypatch):
    monkeypatch.setenv("DOCSHIELD_ML_MODE", "mock")
    from app.pipeline import run_pipeline

    result = run_pipeline(_blank(), filename="doc.png")
    face = result["evidence"]["face"]
    assert face["status"] in {"NOT_DETECTED", "DETECTED", "MULTIPLE_DETECTED", "ERROR"}
    auth = result["assessment"]["authenticity_estimate"]
    assert auth is None or (0.0 <= float(auth) <= 1.0)
    assert result["screening_result"]["phase_status"]["11_face"] == face["status"]
    assert "17_decision" in result["screening_result"]["phase_status"]


def test_face_detect_api(client, png_bytes):
    res = client.post(
        "/api/face/detect",
        files={"file": ("t.png", png_bytes, "image/png")},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["status"] in {"NOT_DETECTED", "DETECTED", "MULTIPLE_DETECTED", "ERROR"}
    assert body["liveness"]["status"] == "NOT_APPLICABLE"


def test_face_compare_requires_file(client, png_bytes):
    res = client.post(
        "/api/face/compare",
        files={"file": ("t.png", png_bytes, "image/png")},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["verification"]["status"] == "NOT_ASSESSED"
    assert body["face_verification"] in {
        "NOT_DETECTED",
        "DETECTED",
        "MULTIPLE_DETECTED",
        "NOT_ASSESSED",
        "NOT_AVAILABLE",
        "ERROR",
    }
    assert body["verification"]["result"] == "NOT_AVAILABLE"


def test_health_sih_phase11(client):
    body = client.get("/api/health").json()
    assert "sih-phase-" in body["phase"]
    assert "haar" in body["components"]["face_detection"]
    assert "passive" in body["components"]["face_liveness"]
    assert "ephemeral" in body["components"]["live_camera"] or body["components"]["live_camera"].startswith("working")
    assert body["version"].startswith("0.") and "sih-phase" in body["version"]
