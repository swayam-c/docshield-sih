"""SIH Phase 13 — passive face liveness / PAD heuristics (review-only).

Classical single-frame (and optional multi-frame motion) cues:
  - face presence / size
  - texture flatness (print/screen proxy)
  - specular / moiré energy
  - color naturalness
  - optional frame-to-frame motion (static replay cue)

NEVER a calibrated anti-spoof verdict. Never sets authenticity_estimate.
Challenge-response (blink/pose) is out of scope for this phase.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

import cv2
import numpy as np

from ocr.preprocess import decode_bgr


def _finite(x: float, default: float = 0.0) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    if math.isnan(v) or math.isinf(v):
        return default
    return v


def _clip01(x: float) -> float:
    return float(np.clip(_finite(x), 0.0, 1.0))


def _largest_face_crop(bgr: np.ndarray, faces: List[Dict[str, Any]]) -> Optional[np.ndarray]:
    if not faces:
        return None
    h, w = bgr.shape[:2]
    f0 = faces[0]
    bbox = f0.get("bbox") or {}
    x, y, fw, fh = int(bbox.get("x", 0)), int(bbox.get("y", 0)), int(bbox.get("w", 0)), int(bbox.get("h", 0))
    pad = int(0.12 * max(fw, fh))
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(w, x + fw + pad), min(h, y + fh + pad)
    crop = bgr[y0:y1, x0:x1]
    return crop if crop.size else None


def _texture_flatness(gray: np.ndarray) -> float:
    """High flatness → possible print/screen (low micro-texture). Returns spoof-leaning score."""
    lap = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    # Live faces often have more micro-texture; very low lap → flat
    live_like = _clip01(lap / 250.0)
    return _clip01(1.0 - live_like)


def _moire_energy(gray: np.ndarray) -> float:
    """Periodic high-frequency energy proxy for screen capture."""
    f = np.fft.fft2(gray.astype(np.float32))
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift)
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    # Ring band away from DC
    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r0, r1 = 0.12 * min(h, w), 0.45 * min(h, w)
    band = mag[(dist >= r0) & (dist <= r1)]
    if band.size == 0:
        return 0.0
    # Peakiness of band
    mean = float(np.mean(band)) + 1e-6
    peak = float(np.percentile(band, 99))
    return _clip01((peak / mean - 1.0) / 8.0)


def _specular_ratio(bgr: np.ndarray) -> float:
    """Very bright clipped highlights can indicate screen glare."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    return _clip01(float(np.mean(v >= 245)) / 0.08)


def _color_cast_score(bgr: np.ndarray) -> float:
    """Strong unnatural cast → higher spoof-leaning score."""
    means = bgr.reshape(-1, 3).mean(axis=0)
    # BGR balance deviation
    mx = float(np.max(means)) + 1e-6
    bal = float(np.std(means / mx))
    return _clip01(bal / 0.35)


def _motion_score(prev_gray: Optional[np.ndarray], gray: np.ndarray) -> Optional[float]:
    """Return motion magnitude 0–1; None if no previous frame."""
    if prev_gray is None:
        return None
    a = cv2.resize(prev_gray, (64, 64))
    b = cv2.resize(gray, (64, 64))
    diff = cv2.absdiff(a, b)
    return _clip01(float(np.mean(diff)) / 40.0)


def assess_passive_liveness(
    image_bytes: bytes,
    *,
    source: str = "unknown",
    previous_frame_gray: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Passiveive PAD heuristics.

    source:
      - live_camera → full passive assessment
      - document → NOT_APPLICABLE (printed photo expected)
    """
    if source == "document":
        return {
            "status": "NOT_APPLICABLE",
            "method": "passive_heuristic_v1",
            "decision": "NOT_APPLICABLE",
            "score_spoof_leaning": None,
            "signals": {},
            "calibrated": False,
            "concern": "NOT_APPLICABLE",
            "note": (
                "Liveness/PAD does not apply to the printed/photo face on a document image. "
                "Use live camera capture for passive PAD cues."
            ),
        }

    try:
        bgr = decode_bgr(image_bytes)
    except ValueError as exc:
        return {
            "status": "ERROR",
            "error": str(exc),
            "decision": "INCONCLUSIVE",
            "calibrated": False,
            "concern": "NOT_ASSESSED",
            "note": "Liveness not assessed — decode failed.",
        }

    from verification.face import detect_faces  # lazy — avoid circular import with face.py

    face_pack = detect_faces(image_bytes)
    # Strip private embeddings
    faces_pub = []
    for f in face_pack.get("faces") or []:
        faces_pub.append({k: v for k, v in f.items() if k != "_embedding"})

    if face_pack.get("status") in {"NOT_DETECTED", "ERROR"} or not (face_pack.get("faces") or []):
        return {
            "status": "NOT_ASSESSED",
            "method": "passive_heuristic_v1",
            "decision": "INCONCLUSIVE",
            "reason": "No face detected for PAD.",
            "face_status": face_pack.get("status"),
            "signals": {},
            "calibrated": False,
            "concern": "UNCALIBRATED_PASSIVE_PAD",
            "note": "Passive PAD requires a detectable face on the live frame.",
        }

    crop = _largest_face_crop(bgr, face_pack["faces"])
    if crop is None:
        return {
            "status": "NOT_ASSESSED",
            "decision": "INCONCLUSIVE",
            "calibrated": False,
            "concern": "UNCALIBRATED_PASSIVE_PAD",
            "note": "Face crop unavailable.",
        }

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    flat = _texture_flatness(gray)
    moire = _moire_energy(gray)
    specular = _specular_ratio(crop)
    cast = _color_cast_score(crop)
    quality = (faces_pub[0].get("quality") or {}) if faces_pub else {}
    area = float(quality.get("area_fraction") or 0.0)
    size_pen = _clip01(0.03 / max(area, 1e-6) - 1.0)  # tiny faces → inconclusive lean

    motion = _motion_score(previous_frame_gray, cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
    static_penalty = 0.0
    if motion is not None and motion < 0.02:
        static_penalty = 0.35  # near-identical frames → replay/static cue

    # Spoof-leaning aggregate (NOT probability of attack)
    spoof = _clip01(
        0.30 * flat
        + 0.25 * moire
        + 0.20 * specular
        + 0.10 * cast
        + 0.15 * static_penalty
        + 0.10 * size_pen
    )

    flags: List[str] = []
    if flat >= 0.65:
        flags.append("flat_texture")
    if moire >= 0.55:
        flags.append("moire_like_energy")
    if specular >= 0.55:
        flags.append("specular_glare")
    if static_penalty > 0:
        flags.append("low_frame_motion")
    if area < 0.02:
        flags.append("face_too_small")

    if spoof >= 0.62 or len(flags) >= 3:
        decision = "SUSPECT_PRESENTATION_ATTACK"
        status = "PASSIVE_ASSESSED"
    elif spoof <= 0.35 and not flags:
        decision = "PASSIVE_OK_UNCALIBRATED"
        status = "PASSIVE_ASSESSED"
    else:
        decision = "INCONCLUSIVE"
        status = "PASSIVE_ASSESSED"

    return {
        "status": status,
        "method": "passive_heuristic_v1",
        "decision": decision,
        "score_spoof_leaning": round(spoof, 4),
        "flags": flags,
        "signals": {
            "texture_flatness": round(flat, 4),
            "moire_energy": round(moire, 4),
            "specular_ratio": round(specular, 4),
            "color_cast": round(cast, 4),
            "face_area_fraction": round(area, 4),
            "frame_motion": None if motion is None else round(motion, 4),
        },
        "face_status": face_pack.get("status"),
        "face_count": face_pack.get("face_count"),
        "calibrated": False,
        "concern": "UNCALIBRATED_PASSIVE_PAD",
        "note": (
            "Passive PAD heuristics are review signals only. "
            "PASSIVE_OK_UNCALIBRATED is not proof of a live person. "
            "SUSPECT_PRESENTATION_ATTACK is not proof of fraud. "
            "Challenge-response liveness is not implemented."
        ),
    }


def gray_fingerprint(image_bytes: bytes) -> Optional[np.ndarray]:
    """Downscaled gray frame for session motion checks."""
    try:
        bgr = decode_bgr(image_bytes)
    except ValueError:
        return None
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (96, 96))
