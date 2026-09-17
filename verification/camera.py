"""SIH Phase 12–13 — live camera capture + passive PAD on frames.

Ephemeral in-memory sessions. Challenge-response liveness not implemented.
Does NOT set authenticity_estimate.
"""

from __future__ import annotations

import threading
import time
import uuid
from typing import Any, Dict, Optional

from verification.face import analyze_document_face, compare_faces
from verification.liveness import assess_passive_liveness, gray_fingerprint

_lock = threading.Lock()
_sessions: Dict[str, Dict[str, Any]] = {}

SESSION_TTL_SEC = 600


def _purge_expired() -> None:
    now = time.time()
    dead = [sid for sid, s in _sessions.items() if now - s.get("created_at", 0) > SESSION_TTL_SEC]
    for sid in dead:
        _sessions.pop(sid, None)


def start_camera_session(*, case_id: Optional[str] = None) -> Dict[str, Any]:
    with _lock:
        _purge_expired()
        sid = uuid.uuid4().hex
        _sessions[sid] = {
            "session_id": sid,
            "case_id": case_id,
            "created_at": time.time(),
            "captures": 0,
            "last_capture": None,
            "document_face_bytes": None,
            "last_gray": None,
        }
    return {
        "status": "READY",
        "session_id": sid,
        "case_id": case_id,
        "ttl_seconds": SESSION_TTL_SEC,
        "liveness": {
            "status": "READY",
            "method": "passive_heuristic_v1",
            "note": "Passive PAD runs on each capture. Challenge-response not implemented.",
        },
        "note": (
            "Camera session is ephemeral and in-memory only. "
            "Frames are analyzed then discarded from session storage "
            "(last gray fingerprint kept briefly for motion cue)."
        ),
        "concern": "CAPTURE_ONLY",
    }


def end_camera_session(session_id: str) -> Dict[str, Any]:
    with _lock:
        existed = _sessions.pop(session_id, None) is not None
    return {
        "status": "CLOSED" if existed else "NOT_FOUND",
        "session_id": session_id,
    }


def attach_document_reference(session_id: str, document_bytes: bytes) -> Dict[str, Any]:
    with _lock:
        _purge_expired()
        sess = _sessions.get(session_id)
        if not sess:
            return {"status": "NOT_FOUND", "session_id": session_id}
        sess["document_face_bytes"] = document_bytes
        sess["document_attached"] = True
    return {
        "status": "ATTACHED",
        "session_id": session_id,
        "note": "Document reference attached for optional uncalibrated selfie compare.",
    }


def analyze_camera_frame(
    frame_bytes: bytes,
    *,
    session_id: Optional[str] = None,
    compare_to_document: bool = False,
) -> Dict[str, Any]:
    """Analyze a live camera frame: face detect + passive PAD (+ optional compare)."""
    face = analyze_document_face(frame_bytes)
    verification = {
        "status": "NOT_ASSESSED",
        "reason": "No document reference compare requested.",
    }

    prev_gray = None
    sess_meta: Dict[str, Any] = {}
    doc_bytes = None

    if session_id:
        with _lock:
            _purge_expired()
            sess = _sessions.get(session_id)
            if not sess:
                return {
                    "status": "SESSION_NOT_FOUND",
                    "session_id": session_id,
                    "camera": {"status": "ERROR"},
                    "face": face,
                    "liveness": {
                        "status": "NOT_ASSESSED",
                        "note": "Session missing — PAD not run.",
                    },
                    "concern": "CAPTURE_ONLY",
                }
            prev_gray = sess.get("last_gray")
            sess["captures"] = int(sess.get("captures") or 0) + 1
            sess_meta = {
                "session_id": session_id,
                "captures": sess["captures"],
                "case_id": sess.get("case_id"),
                "document_attached": bool(sess.get("document_attached")),
            }
            if compare_to_document:
                doc_bytes = sess.get("document_face_bytes")

    liveness = assess_passive_liveness(
        frame_bytes,
        source="live_camera",
        previous_frame_gray=prev_gray,
    )

    # Store gray fingerprint for next motion cue
    if session_id:
        with _lock:
            sess = _sessions.get(session_id)
            if sess is not None:
                sess["last_gray"] = gray_fingerprint(frame_bytes)
                sess["last_capture"] = {
                    "face_status": face.get("status"),
                    "face_count": face.get("face_count"),
                    "liveness_decision": liveness.get("decision"),
                    "ts": time.time(),
                }

    if compare_to_document and doc_bytes:
        try:
            compared = compare_faces(doc_bytes, reference_bytes=frame_bytes)
            verification = compared.get("verification") or {
                "status": "NOT_ASSESSED",
                "result": "NOT_AVAILABLE",
                "reason": "Compare could not run.",
            }
            # Keep document vs live faces separate — never overwrite verification with detection
            face = {
                **face,
                "verification": verification,
                "verification_debug": compared.get("verification_debug"),
                "document_face": compared.get("document_face"),
                "live_face": compared.get("live_face"),
                "reference": compared.get("reference") or face.get("reference"),
                "document_face_status": (compared.get("document_face") or {}).get("status")
                or compared.get("status"),
                "document_face_count": (compared.get("document_face") or {}).get("face_count")
                or compared.get("face_count"),
                "face_verification": compared.get("face_verification") or verification.get("result"),
            }
        except Exception as exc:  # noqa: BLE001 — never 500 the capture path
            verification = {
                "status": "ERROR",
                "result": "NOT_AVAILABLE",
                "reason": f"Compare failed: {exc}",
                "calibration": "NOT_ASSESSED",
            }
            face = {**face, "verification": verification}

    camera_block = {
        "status": "CAPTURED",
        "source": "live_camera_frame",
        "session": sess_meta or None,
        "frame_bytes_sha16": __import__("hashlib").sha256(frame_bytes).hexdigest()[:16],
        "captured_at": time.time(),
        "note": (
            "Frame captured via client camera and analyzed server-side. "
            "Passive PAD is uncalibrated; not challenge-response proof. "
            "Liveness/PAD ≠ identity MATCH."
        ),
    }

    face_out = {
        **{k: v for k, v in face.items() if k not in {"camera", "liveness"}},
        "camera": camera_block,
        "liveness": liveness,
        "verification": verification if compare_to_document else (face.get("verification") or verification),
    }

    return {
        "status": "CAPTURED",
        "camera": camera_block,
        "face": face_out,
        "liveness": liveness,
        "verification_debug": face_out.get("verification_debug"),
        "concern": "CAPTURE_AND_PASSIVE_PAD",
        "calibrated": False,
        "note": (
            "Live camera capture + passive PAD heuristics for officer review. "
            "DETECTED ≠ MATCH. Does not prove identity, calibrated liveness, or authenticity."
        ),
    }


def camera_module_status() -> Dict[str, Any]:
    with _lock:
        _purge_expired()
        active = len(_sessions)
    return {
        "status": "READY",
        "active_sessions": active,
        "ttl_seconds": SESSION_TTL_SEC,
        "liveness": "working_passive_heuristic_v1",
        "persistence": "ephemeral_memory_only",
        "concern": "CAPTURE_AND_PASSIVE_PAD",
    }
