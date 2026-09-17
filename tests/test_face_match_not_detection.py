"""Regression: DETECTED ≠ MATCH; different person must not PASS."""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image, ImageDraw


def _paint_face(img: Image.Image, *, cx: int, cy: int, tone, eye_gap: int, mouth_y: int, seed: int) -> None:
    d = ImageDraw.Draw(img)
    rng = np.random.default_rng(seed)
    w, h = 70 + seed * 4, 90 + seed * 3
    d.ellipse([cx - w, cy - h, cx + w, cy + h], fill=tone, outline=(30, 20, 10))
    d.ellipse([cx - eye_gap - 12, cy - 28, cx - eye_gap + 4, cy - 12], fill=(15, 15, 15))
    d.ellipse([cx + eye_gap - 4, cy - 28, cx + eye_gap + 12, cy - 12], fill=(15, 15, 15))
    d.arc([cx - 22, mouth_y, cx + 22, mouth_y + 28], 20, 160, fill=(50, 20, 20), width=3)
    # local texture to diversify LBP
    arr = np.array(img)
    noise = rng.integers(-18, 19, size=(h * 2, w * 2, 3), dtype=np.int16)
    y0, x0 = max(0, cy - h), max(0, cx - w)
    y1, x1 = min(arr.shape[0], cy + h), min(arr.shape[1], cx + w)
    patch = arr[y0:y1, x0:x1].astype(np.int16)
    n = noise[: patch.shape[0], : patch.shape[1]]
    arr[y0:y1, x0:x1] = np.clip(patch + n, 0, 255).astype(np.uint8)
    img.paste(Image.fromarray(arr))


def _face_png(*, person: str) -> bytes:
    img = Image.new("RGB", (480, 640), color=(235, 235, 230))
    d = ImageDraw.Draw(img)
    d.rectangle([16, 16, 464, 624], outline=(20, 20, 20), width=3)
    if person == "A":
        _paint_face(img, cx=160, cy=220, tone=(210, 175, 150), eye_gap=28, mouth_y=240, seed=1)
    else:
        _paint_face(img, cx=160, cy=220, tone=(90, 70, 55), eye_gap=40, mouth_y=255, seed=9)
        # darker hair blob
        d = ImageDraw.Draw(img)
        d.ellipse([90, 110, 230, 180], fill=(20, 15, 10))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _auth_base(**face):
    from evidence.decision import decide_screening

    return decide_screening(
        prior_decision="PENDING",
        evidence_quality=0.85,
        authenticity_estimate=0.9,
        confidence=0.72,
        fusion={"status": "FUSED", "authentic_probability": 0.75},
        cross_validation={"overall": "PASS", "counts": {"FAIL": 0, "WARNING": 0}},
        tampering={"severity": "LOW_SIGNAL", "review_score": 0.15},
        calibration={"status": "CALIBRATED"},
        face=face,
    )


def test_detection_alone_never_pass():
    d = _auth_base(
        status="DETECTED",
        face_count=1,
        verification={"status": "NOT_ASSESSED", "result": "NOT_AVAILABLE"},
    )
    assert d["decision"] != "PASS"
    assert d["officer_verdict"]["identity_status"] in {"AWAITING_LIVE", "NOT_ASSESSED", "INCONCLUSIVE"}


def test_liveness_pass_alone_never_pass():
    d = _auth_base(
        status="DETECTED",
        face_count=1,
        verification={"status": "NOT_ASSESSED", "result": "NOT_AVAILABLE"},
        liveness={"decision": "PASSIVE_OK_UNCALIBRATED", "status": "PASSIVE_ASSESSED"},
        camera={"status": "CAPTURED"},
    )
    assert d["decision"] != "PASS"


def test_high_similarity_decision_string_alone_never_match_identity():
    """INCONCLUSIVE_HIGH_SIMILARITY without result=MATCH must not become PASS."""
    d = _auth_base(
        status="DETECTED",
        face_count=1,
        verification={
            "status": "COMPARED_UNCALIBRATED",
            "decision": "INCONCLUSIVE_HIGH_SIMILARITY",
            "similarity": 0.918,
            "calibration": "UNCALIBRATED",
            "calibrated": False,
            # intentionally omit result / match_label
        },
        reference={"status": "DETECTED", "face_count": 1},
        liveness={"decision": "PASSIVE_OK_UNCALIBRATED"},
        camera={"status": "CAPTURED"},
        document_face={"status": "DETECTED", "face_count": 1},
        live_face={"status": "DETECTED", "face_count": 1},
    )
    assert d["officer_verdict"]["identity_status"] != "MATCH"
    assert d["decision"] != "PASS"


def test_authentic_no_match_is_review_required():
    d = _auth_base(
        status="DETECTED",
        face_count=1,
        verification={
            "status": "COMPARED_UNCALIBRATED",
            "result": "NO_MATCH",
            "match_label": "NO_MATCH",
            "decision": "INCONCLUSIVE_LOW_SIMILARITY",
            "similarity": 0.41,
            "calibration": "UNCALIBRATED",
            "calibrated": False,
            "embedding_sources": "DIFFERENT",
        },
        reference={"status": "DETECTED", "face_count": 1},
        document_face={"status": "DETECTED", "face_count": 1},
        live_face={"status": "DETECTED", "face_count": 1},
        camera={"status": "CAPTURED"},
        liveness={"decision": "PASSIVE_OK_UNCALIBRATED"},
    )
    assert d["decision"] == "INCONCLUSIVE"
    assert d["officer_verdict"]["code"] == "INCONCLUSIVE"
    assert d["officer_verdict"]["identity_status"] == "NO_MATCH"


def test_authentic_match_can_pass():
    d = _auth_base(
        status="DETECTED",
        face_count=1,
        verification={
            "status": "COMPARED_UNCALIBRATED",
            "result": "MATCH",
            "match_label": "MATCH",
            "decision": "INCONCLUSIVE_HIGH_SIMILARITY",
            "similarity": 0.94,
            "calibration": "UNCALIBRATED",
            "calibrated": False,
            "embedding_sources": "DIFFERENT",
        },
        reference={"status": "DETECTED", "face_count": 1},
        document_face={"status": "DETECTED", "face_count": 1},
        live_face={"status": "DETECTED", "face_count": 1},
        camera={"status": "CAPTURED"},
        liveness={"decision": "PASSIVE_OK_UNCALIBRATED"},
    )
    assert d["decision"] == "PASS"
    assert d["officer_verdict"]["identity_status"] == "MATCH"


def test_self_compare_guard_blocks_match():
    d = _auth_base(
        status="DETECTED",
        face_count=1,
        verification={
            "status": "COMPARED_UNCALIBRATED",
            "result": "MATCH",
            "match_label": "MATCH",
            "similarity": 0.99,
            "calibration": "UNCALIBRATED",
            "calibrated": False,
            "embedding_sources": "SAME_IMAGE_BYTES",
        },
        document_face={"status": "DETECTED", "face_count": 1},
        live_face={"status": "DETECTED", "face_count": 1},
        camera={"status": "CAPTURED"},
    )
    assert d["decision"] != "PASS"
    assert d["officer_verdict"]["identity_status"] == "INCONCLUSIVE"


def test_compare_different_embeddings_not_auto_match(monkeypatch):
    import verification.face as face_mod

    rng = np.random.default_rng(0)
    emb_a = rng.normal(size=face_mod.EMBEDDING_DIM).astype(np.float32)
    emb_a = emb_a / (np.linalg.norm(emb_a) + 1e-8)
    emb_b = rng.normal(size=face_mod.EMBEDDING_DIM).astype(np.float32)
    emb_b = emb_b / (np.linalg.norm(emb_b) + 1e-8)

    def _fake_detect(image_bytes: bytes, *, source_role: str = "unknown"):
        emb = emb_a if source_role == "document" else emb_b
        return {
            "status": "DETECTED",
            "face_count": 1,
            "faces": [
                {
                    "_embedding": emb.copy(),
                    "bbox": {"x": 10, "y": 10, "w": 80, "h": 80},
                    "quality": {"label": "GOOD", "overall": 0.9},
                    "embedding_dim": emb.shape[0],
                    "embedding_id": face_mod._embedding_id(emb),
                    "embedding_fingerprint": [round(float(v), 5) for v in emb[:12]],
                    "source_role": source_role,
                }
            ],
            "image_sha16": "docsha" if source_role == "document" else "livesha",
            "source_role": source_role,
            "verification": {"status": "NOT_ASSESSED", "result": "NOT_AVAILABLE"},
            "liveness": {"status": "NOT_APPLICABLE"},
            "camera": {"status": "NOT_USED"},
            "concern": "DETECTION_ONLY",
        }

    monkeypatch.setattr(face_mod, "detect_faces", _fake_detect)
    out = face_mod.compare_faces(b"document-aaaa", reference_bytes=b"live-bbbb")
    assert out["document_face"]["status"] == "DETECTED"
    assert out["live_face"]["status"] == "DETECTED"
    assert out["verification"]["result"] in {"NO_MATCH", "INCONCLUSIVE"}
    assert out["verification"]["result"] != "MATCH"
    assert out["verification_debug"]["embedding_sources"] == "DIFFERENT"
    assert out["face_verification"] != "DETECTED"

    fused = _auth_base(
        status="DETECTED",
        face_count=1,
        verification=out["verification"],
        verification_debug=out["verification_debug"],
        document_face=out["document_face"],
        live_face=out["live_face"],
        reference=out["reference"],
        camera={"status": "CAPTURED"},
        liveness={"decision": "PASSIVE_OK_UNCALIBRATED"},
    )
    assert fused["decision"] != "PASS"


def test_compare_same_embedding_can_match(monkeypatch):
    import verification.face as face_mod

    emb = np.linspace(0.2, 1.0, face_mod.EMBEDDING_DIM, dtype=np.float32)
    emb = emb / (np.linalg.norm(emb) + 1e-8)

    def _fake_detect(image_bytes: bytes, *, source_role: str = "unknown"):
        e = emb.copy()
        if source_role == "live":
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
                    "embedding_dim": e.shape[0],
                    "embedding_id": face_mod._embedding_id(e),
                    "embedding_fingerprint": [round(float(v), 5) for v in e[:12]],
                    "source_role": source_role,
                }
            ],
            "image_sha16": "aaaaaaaa" if source_role == "document" else "bbbbbbbb",
            "source_role": source_role,
            "verification": {"status": "NOT_ASSESSED", "result": "NOT_AVAILABLE"},
            "liveness": {"status": "NOT_APPLICABLE"},
            "camera": {"status": "NOT_USED"},
            "concern": "DETECTION_ONLY",
        }

    monkeypatch.setattr(face_mod, "detect_faces", _fake_detect)
    out = face_mod.compare_faces(b"document-aaaa", reference_bytes=b"live-bbbb")
    assert out["verification"]["result"] == "MATCH"
    fused = _auth_base(
        status="DETECTED",
        face_count=1,
        verification=out["verification"],
        document_face=out["document_face"],
        live_face=out["live_face"],
        camera={"status": "CAPTURED"},
        liveness={"decision": "PASSIVE_OK_UNCALIBRATED"},
    )
    assert fused["decision"] == "PASS"


def test_compare_rejects_same_image_bytes(monkeypatch):
    import verification.face as face_mod

    emb = np.linspace(0.2, 1.0, face_mod.EMBEDDING_DIM, dtype=np.float32)
    emb = emb / (np.linalg.norm(emb) + 1e-8)

    def _fake_detect(image_bytes: bytes, *, source_role: str = "unknown"):
        return {
            "status": "DETECTED",
            "face_count": 1,
            "faces": [
                {
                    "_embedding": emb.copy(),
                    "bbox": {"x": 10, "y": 10, "w": 80, "h": 80},
                    "quality": {"label": "GOOD", "overall": 0.9},
                    "embedding_dim": emb.shape[0],
                    "embedding_id": face_mod._embedding_id(emb),
                    "embedding_fingerprint": [round(float(v), 5) for v in emb[:12]],
                    "source_role": source_role,
                }
            ],
            "image_sha16": "samehashsamehash",
            "source_role": source_role,
            "verification": {"status": "NOT_ASSESSED", "result": "NOT_AVAILABLE"},
            "liveness": {"status": "NOT_APPLICABLE"},
            "camera": {"status": "NOT_USED"},
            "concern": "DETECTION_ONLY",
        }

    monkeypatch.setattr(face_mod, "detect_faces", _fake_detect)
    png = b"same-image-bytes-aaaaaaaa"
    out = face_mod.compare_faces(png, reference_bytes=png)
    assert out["verification"]["result"] == "INCONCLUSIVE"
    assert out["verification_debug"]["embedding_sources"] == "SAME_IMAGE_BYTES"
