"""Image / photo / stamp region splicing cues + EXIF metadata.

Review signals only. Missing metadata ≠ fraud.
"""

from __future__ import annotations

import io
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ExifTags

from ocr.preprocess import decode_bgr


def analyze_metadata(image_bytes: bytes) -> Dict[str, Any]:
    try:
        img = Image.open(io.BytesIO(image_bytes))
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "ERROR",
            "error": repr(exc),
            "existance": "UNAVAILABLE",
            "note": "Could not open image for metadata.",
        }
    try:
        raw = img.getexif()
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "UNAVAILABLE",
            "existance": "UNAVAILABLE",
            "error": repr(exc),
            "note": "EXIF unavailable — missing metadata is not proof of fraud.",
        }
    if not raw:
        return {
            "status": "UNAVAILABLE",
            "existance": "UNAVAILABLE",
            "tag_count": 0,
            "note": "No EXIF tags present — missing metadata is not proof of fraud.",
        }
    tags = {}
    for k, v in raw.items():
        name = ExifTags.TAGS.get(k, str(k))
        try:
            tags[name] = str(v)[:120]
        except Exception:
            tags[name] = "<unreadable>"
    # Soft heuristic: Software tag alone is not suspicious; empty GPS is normal
    return {
        "status": "PRESENT",
        "existance": "PRESENT",
        "tag_count": len(tags),
        "tags_sample": {k: tags[k] for k in list(tags)[:12]},
        "suspicious": False,
        "note": "Metadata present. Absence elsewhere must not be treated as fraud proof.",
    }


def analyze_stamp_seal(image_bytes: bytes) -> Dict[str, Any]:
    """Heuristic red/blue circular stamp cue — not official template matching."""
    try:
        bgr = decode_bgr(image_bytes)
    except ValueError as exc:
        return {"status": "ERROR", "error": str(exc), "detection": "NOT_ASSESSED"}

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # Reddish + bluish masks (common ink colors)
    mask_r1 = cv2.inRange(hsv, (0, 70, 50), (10, 255, 255))
    mask_r2 = cv2.inRange(hsv, (170, 70, 50), (180, 255, 255))
    mask_b = cv2.inRange(hsv, (90, 60, 40), (130, 255, 255))
    mask = cv2.bitwise_or(cv2.bitwise_or(mask_r1, mask_r2), mask_b)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    frac = float(np.mean(mask > 0))
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=40,
        param1=80,
        param2=40,
        minRadius=12,
        maxRadius=min(gray.shape[:2]) // 3,
    )
    circle_count = 0 if circles is None else int(circles.shape[1])
    if circle_count > 0 and frac > 0.002:
        return {
            "status": "ANALYZED",
            "detection": "DETECTED",
            "circle_candidates": circle_count,
            "colored_ink_fraction": round(frac, 5),
            "note": "Possible stamp/seal-like circular region — not matched to official templates.",
        }
    if frac > 0.01:
        return {
            "status": "ANALYZED",
            "detection": "UNCERTAIN",
            "circle_candidates": circle_count,
            "colored_ink_fraction": round(frac, 5),
            "note": "Colored ink present without clear circular geometry.",
        }
    return {
        "status": "ANALYZED",
        "detection": "NO_CLEAR_ANOMALY",
        "circle_candidates": circle_count,
        "colored_ink_fraction": round(frac, 5),
        "note": "No clear stamp/seal cue. NOT_APPLICABLE for documents without seals.",
    }


def analyze_photo_region_splicing(
    image_bytes: bytes,
    face_bbox: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Photo-region residual / boundary cues when a face bbox exists."""
    try:
        bgr = decode_bgr(image_bytes)
    except ValueError as exc:
        return {"status": "ERROR", "error": str(exc)}

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    if not face_bbox or len(face_bbox) < 4:
        return {
            "status": "NOT_APPLICABLE",
            "note": "No document face bbox for photo-region splicing analysis.",
        }

    # face bbox may be x,y,w,h
    x, y, bw, bh = [int(v) for v in face_bbox[:4]]
    x2, y2 = min(w, x + bw), min(h, y + bh)
    x, y = max(0, x), max(0, y)
    if x2 - x < 8 or y2 - y < 8:
        return {"status": "UNCERTAIN", "note": "Face crop too small."}

    # Expand slightly for boundary ring
    pad = max(4, min(bw, bh) // 10)
    outer = gray[max(0, y - pad) : min(h, y2 + pad), max(0, x - pad) : min(w, x2 + pad)]
    inner = gray[y:y2, x:x2]
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    residual = cv2.absdiff(gray, blur).astype(np.float32)
    inner_r = residual[y:y2, x:x2]
    ring = residual[max(0, y - pad) : min(h, y2 + pad), max(0, x - pad) : min(w, x2 + pad)].copy()
    # zero out inner in ring approx by mean comparison
    inner_mean = float(np.mean(inner_r)) + 1e-6
    outer_mean = float(np.mean(ring)) + 1e-6
    ratio = outer_mean / inner_mean
    status = "NO_CLEAR_ANOMALY"
    if ratio > 2.0 or ratio < 0.4:
        status = "ANOMALY_DETECTED"
    elif ratio > 1.6 or ratio < 0.55:
        status = "UNCERTAIN"

    return {
        "status": "ANALYZED",
        "region": "photo",
        "evidence_type": "boundary_residual_ratio",
        "score": round(min(1.0, abs(np.log(ratio)) / 2.0), 4),
        "bbox": [x, y, bw, bh],
        "detection": status,
        "note": "Photo-region boundary cue is heuristic — not proof of photo swap.",
    }


def build_forensic_map(
    *,
    tampering: Dict[str, Any],
    text_visual: Dict[str, Any],
    photo: Dict[str, Any],
    stamp: Dict[str, Any],
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "status": "READY",
        "text": {
            "typography": (text_visual or {}).get("typography"),
            "splicing": (text_visual or {}).get("text_splicing"),
            "ocr_visual_crosscheck": (text_visual or {}).get("ocr_visual_crosscheck"),
        },
        "image": {
            "ela": ((tampering or {}).get("signals") or {}).get("ela_inconsistency"),
            "copy_move": ((tampering or {}).get("signals") or {}).get("copy_move_indicators"),
            "noise": ((tampering or {}).get("signals") or {}).get("noise_inconsistency"),
            "localization": (tampering or {}).get("localization"),
            "photo": photo,
            "stamp": stamp,
        },
        "metadata": metadata,
        "note": (
            "Combined forensic map for review. Anomalies are not definitive fraud proof."
        ),
    }
