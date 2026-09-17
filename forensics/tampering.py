"""SIH Phase 9–10 — visual tampering heuristics + localization (review signals only).

Phase 9: classical inconsistency cues (ELA, noise, edge, copy-move).
Phase 10: coarse spatial heatmap / hotspots from residual blends.

NONE of these prove fraud or authenticity. Flags and heatmaps are for human review.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from forensics.localization import build_localization
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


def _ela_inconsistency(bgr: np.ndarray, quality: int = 90) -> Tuple[float, Dict[str, float]]:
    """Error Level Analysis proxy via JPEG re-encode residual energy."""
    ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return 0.0, {"mean_residual": 0.0, "std_residual": 0.0, "max_z": 0.0}
    recompressed = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    if recompressed is None:
        return 0.0, {"mean_residual": 0.0, "std_residual": 0.0, "max_z": 0.0}
    if recompressed.shape != bgr.shape:
        recompressed = cv2.resize(recompressed, (bgr.shape[1], bgr.shape[0]))

    diff = cv2.absdiff(bgr, recompressed).astype(np.float32)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    mean_r = float(np.mean(gray))
    std_r = float(np.std(gray))

    # Tile max z-score of residual energy
    h, w = gray.shape[:2]
    energies: List[float] = []
    step = max(24, min(h, w) // 12)
    tile = max(24, step)
    for y in range(0, max(1, h - tile), step):
        for x in range(0, max(1, w - tile), step):
            energies.append(float(np.mean(gray[y : y + tile, x : x + tile])))
    en = np.asarray(energies, dtype=np.float32) if energies else np.asarray([0.0], dtype=np.float32)
    if float(en.std()) > 1e-6:
        max_z = float((en.max() - en.mean()) / (en.std() + 1e-6))
    else:
        max_z = 0.0

    # Map residual intensity + spatial outliers → [0,1] inconsistency
    score = _clip01(0.55 * (mean_r / 25.0) + 0.45 * (max_z / 6.0))
    return score, {
        "mean_residual": round(mean_r, 4),
        "std_residual": round(std_r, 4),
        "max_z": round(max_z, 4),
        "jpeg_quality_probe": quality,
    }


def _compression_blockiness(gray: np.ndarray) -> float:
    h, w = gray.shape[:2]
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    residual = cv2.absdiff(gray, blur).astype(np.float32)
    block_vars: List[float] = []
    for y in range(0, max(1, h - 8), 16):
        for x in range(0, max(1, w - 8), 16):
            block_vars.append(float(np.var(residual[y : y + 8, x : x + 8])))
    arr = np.asarray(block_vars, dtype=np.float32) if block_vars else np.asarray([0.0])
    return _clip01(float(np.std(arr) / (np.mean(arr) + 1e-6)) / 5.0)


def _noise_inconsistency(gray: np.ndarray) -> float:
    noise_map = cv2.Laplacian(gray, cv2.CV_64F)
    h, w = gray.shape[:2]
    tiles: List[float] = []
    for y in range(0, max(1, h - 32), 32):
        for x in range(0, max(1, w - 32), 32):
            tiles.append(float(np.std(noise_map[y : y + 32, x : x + 32])))
    arr = np.asarray(tiles, dtype=np.float32) if tiles else np.asarray([0.0])
    return _clip01(float(np.std(arr) / (np.mean(arr) + 1e-6)) / 5.0)


def _edge_inconsistency(gray: np.ndarray) -> float:
    h, _ = gray.shape[:2]
    edges = cv2.Canny(gray, 80, 160)
    top = float(np.mean(edges[: max(1, h // 3)] > 0))
    bot = float(np.mean(edges[2 * h // 3 :] > 0))
    return _clip01(abs(top - bot) * 4.0)


def _local_anomaly(gray: np.ndarray) -> float:
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    residual = cv2.absdiff(gray, blur).astype(np.float32)
    h, w = gray.shape[:2]
    energies: List[float] = []
    for y in range(0, max(1, h - 48), 48):
        for x in range(0, max(1, w - 48), 48):
            energies.append(float(np.mean(residual[y : y + 48, x : x + 48])))
    en = np.asarray(energies, dtype=np.float32) if energies else np.asarray([0.0])
    if float(en.std()) > 1e-6:
        return _clip01((float(en.max()) - float(en.mean())) / (float(en.std()) + 1e-6) / 6.0)
    return 0.0


def _copy_move_indicator(gray: np.ndarray) -> float:
    h, w = gray.shape[:2]
    if min(h, w) < 128:
        return 0.0
    patch = gray[h // 4 : h // 4 + 48, w // 4 : w // 4 + 48]
    if patch.size == 0 or patch.shape[0] < 16 or patch.shape[1] < 16:
        return 0.0
    res = cv2.matchTemplate(gray, patch, cv2.TM_CCOEFF_NORMED)
    cy, cx = h // 4, w // 4
    res[max(0, cy - 10) : cy + 58, max(0, cx - 10) : cx + 58] = 0
    return _clip01(float(res.max()))


def analyze_tampering(image_bytes: bytes) -> Dict[str, Any]:
    """Run Phase-9 tampering heuristics. Never fabricates authenticity."""
    try:
        bgr = decode_bgr(image_bytes)
    except ValueError as exc:
        return {
            "status": "ERROR",
            "error": str(exc),
            "flags": [],
            "signals": {},
            "concern": "NOT_ASSESSED",
            "note": "Tampering heuristics not assessed. Anomalies ≠ proof of fraud.",
            "localization": {
                "status": "NOT_ASSESSED",
                "note": "Localization skipped because image decode failed.",
            },
        }

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    ela_score, ela_meta = _ela_inconsistency(bgr)
    compression = _compression_blockiness(gray)
    noise = _noise_inconsistency(gray)
    edge = _edge_inconsistency(gray)
    local = _local_anomaly(gray)
    copy_move = _copy_move_indicator(gray)

    signals = {
        "ela_inconsistency": round(ela_score, 4),
        "compression_inconsistency": round(compression, 4),
        "noise_inconsistency": round(noise, 4),
        "edge_inconsistency": round(edge, 4),
        "local_image_anomaly": round(local, 4),
        "copy_move_indicators": round(copy_move, 4),
        "ela_detail": ela_meta,
        "metadata": {
            "status": "NOT_ASSESSED",
            "note": "EXIF/metadata forensics not enabled.",
        },
    }

    flags: List[str] = []
    thresholds = {
        "ela_inconsistency": 0.55,
        "compression_inconsistency": 0.55,
        "noise_inconsistency": 0.55,
        "edge_inconsistency": 0.55,
        "local_image_anomaly": 0.60,
        "copy_move_indicators": 0.85,
    }
    for key, thr in thresholds.items():
        if float(signals[key]) > thr:
            flags.append(key)

    # Aggregate review risk (not authenticity probability)
    weights = {
        "ela_inconsistency": 0.25,
        "compression_inconsistency": 0.15,
        "noise_inconsistency": 0.15,
        "edge_inconsistency": 0.10,
        "local_image_anomaly": 0.20,
        "copy_move_indicators": 0.15,
    }
    review_score = _clip01(
        sum(float(signals[k]) * w for k, w in weights.items())
    )

    if review_score >= 0.65:
        severity = "ELEVATED_REVIEW"
    elif review_score >= 0.40 or flags:
        severity = "MILD_REVIEW"
    else:
        severity = "LOW_SIGNAL"

    try:
        localization = build_localization(bgr)
    except Exception as exc:  # noqa: BLE001 — never crash Phase 9 signals
        localization = {
            "status": "ERROR",
            "error": repr(exc),
            "note": "Localization failed; scalar tampering signals still available.",
            "concern": "NOT_ASSESSED",
        }

    return {
        "status": "ANALYZED",
        "readiness": "READY",
        "severity": severity,
        "review_score": round(review_score, 4),
        "signals": signals,
        "flags": flags,
        "image_size": {"width": int(w), "height": int(h)},
        "concern": "HEURISTIC_REVIEW_ONLY",
        "calibrated": False,
        "localization": localization,
        "note": (
            "Tampering heuristics and heatmaps are review signals only. "
            "No flag or hotspot is definitive proof of fraud or authenticity. "
            "Does not set authenticity_estimate."
        ),
    }


# Backward-compatible alias used by older imports
def analyze_image_forensics(image_bytes: bytes) -> Dict[str, Any]:
    return analyze_tampering(image_bytes)
