"""SIH Phase 11 — face detection / extraction / optional 1:1 compare.

Uses OpenCV Haar cascade (no new heavy face SDKs).
DETECTION ≠ VERIFICATION ≠ MATCH.

Similarity is a prototype fingerprint score (UNCALIBRATED).
It is NOT an identity probability and must not be treated as such.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ocr.preprocess import decode_bgr
from verification.liveness import assess_passive_liveness

# Prototype fingerprint bands (not a calibrated biometric threshold).
# MATCH requires high agreement on a weak embedding → keep bar strict to reduce
# false MATCH when different people are presented.
MATCH_SIMILARITY_MIN = 0.96
NO_MATCH_SIMILARITY_MAX = 0.70
EMBEDDING_DIM = 160
METRIC_NAME = "cosine_similarity_mapped_01"
EMBEDDING_METHOD = "lbp_grid_hist_v2_uncalibrated"


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


def _cascade() -> cv2.CascadeClassifier:
    path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    clf = cv2.CascadeClassifier(path)
    if clf.empty():
        raise RuntimeError(f"Failed to load Haar cascade: {path}")
    return clf


def _face_quality(gray_face: np.ndarray, full_h: int, full_w: int, bbox: Tuple[int, int, int, int]) -> Dict[str, Any]:
    x, y, w, h = bbox
    area_frac = (w * h) / float(max(1, full_h * full_w))
    lap = float(cv2.Laplacian(gray_face, cv2.CV_64F).var())
    sharp = _clip01(lap / 500.0)
    mean = float(np.mean(gray_face))
    bright = _clip01(1.0 - abs(mean - 128.0) / 128.0)
    contrast = _clip01(float(np.std(gray_face)) / 64.0)
    size_ok = _clip01(area_frac / 0.08)
    overall = _clip01(0.35 * sharp + 0.25 * bright + 0.20 * contrast + 0.20 * size_ok)
    if overall >= 0.65 and area_frac >= 0.02:
        label = "GOOD"
    elif overall >= 0.40:
        label = "FAIR"
    else:
        label = "POOR"
    return {
        "overall": round(overall, 4),
        "label": label,
        "sharpness": round(sharp, 4),
        "brightness": round(bright, 4),
        "contrast": round(contrast, 4),
        "area_fraction": round(area_frac, 4),
    }


def _lbp_hist(gray: np.ndarray, bins: int = 32) -> np.ndarray:
    """Uniform-ish LBP histogram (8-neighbor) — more discriminative than gray hist alone."""
    g = gray.astype(np.int16)
    if g.shape[0] < 3 or g.shape[1] < 3:
        return np.zeros(bins, dtype=np.float32)
    center = g[1:-1, 1:-1]
    codes = np.zeros_like(center, dtype=np.uint8)
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    for i, (dy, dx) in enumerate(offsets):
        neigh = g[1 + dy : g.shape[0] - 1 + dy, 1 + dx : g.shape[1] - 1 + dx]
        codes |= ((neigh >= center).astype(np.uint8) << i)
    hist, _ = np.histogram(codes.ravel(), bins=bins, range=(0, 256), density=True)
    return hist.astype(np.float32)


def _embedding(gray_face: np.ndarray, dim: int = EMBEDDING_DIM) -> np.ndarray:
    """Uncalibrated face fingerprint: LBP + spatial gray grids (not a biometric model)."""
    face = cv2.resize(gray_face, (64, 64), interpolation=cv2.INTER_AREA)
    face = cv2.equalizeHist(face)
    parts: List[np.ndarray] = [_lbp_hist(face, bins=40)]
    # 2x2 spatial gray histograms
    for y0 in (0, 32):
        for x0 in (0, 32):
            cell = face[y0 : y0 + 32, x0 : x0 + 32]
            h = cv2.calcHist([cell], [0], None, [16], [0, 256]).flatten().astype(np.float32)
            h = h / (np.linalg.norm(h) + 1e-8)
            parts.append(h)
    # coarse downsample
    small = cv2.resize(face, (8, 8)).flatten().astype(np.float32)
    small = small / (np.linalg.norm(small) + 1e-8)
    parts.append(small)
    vec = np.concatenate(parts)
    if vec.shape[0] < dim:
        vec = np.pad(vec, (0, dim - vec.shape[0]))
    else:
        vec = vec[:dim]
    vec = vec / (np.linalg.norm(vec) + 1e-8)
    return vec.astype(np.float32)


def _embedding_id(emb: np.ndarray) -> str:
    raw = np.round(emb.astype(np.float64), 6).tobytes()
    return hashlib.sha256(raw).hexdigest()[:16]


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / ((np.linalg.norm(a) + 1e-8) * (np.linalg.norm(b) + 1e-8)))


def detect_faces(image_bytes: bytes, *, source_role: str = "unknown") -> Dict[str, Any]:
    """Detect faces. Detection never implies MATCH."""
    try:
        bgr = decode_bgr(image_bytes)
    except ValueError as exc:
        return {
            "status": "ERROR",
            "error": str(exc),
            "face_count": 0,
            "faces": [],
            "source_role": source_role,
            "verification": {"status": "NOT_ASSESSED", "result": "NOT_AVAILABLE"},
            "liveness": {"status": "NOT_ASSESSED", "note": "Liveness deferred to a later SIH phase."},
            "concern": "NOT_ASSESSED",
            "note": "Face detection failed. Detection ≠ identity verification.",
        }

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    try:
        cascade = _cascade()
    except RuntimeError as exc:
        return {
            "status": "ERROR",
            "error": str(exc),
            "face_count": 0,
            "faces": [],
            "source_role": source_role,
            "verification": {"status": "NOT_ASSESSED", "result": "NOT_AVAILABLE"},
            "liveness": {"status": "NOT_ASSESSED"},
            "concern": "MODEL_NOT_READY",
            "note": str(exc),
        }

    rects = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(max(24, min(h, w) // 20), max(24, min(h, w) // 20)),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )

    faces: List[Dict[str, Any]] = []
    for (x, y, fw, fh) in rects:
        pad = int(0.08 * max(fw, fh))
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(w, x + fw + pad)
        y1 = min(h, y + fh + pad)
        crop = gray[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        quality = _face_quality(crop, h, w, (x, y, fw, fh))
        emb = _embedding(crop)
        faces.append(
            {
                "bbox": {"x": int(x), "y": int(y), "w": int(fw), "h": int(fh)},
                "crop_size": {"w": int(x1 - x0), "h": int(y1 - y0)},
                "quality": quality,
                "embedding_dim": int(emb.shape[0]),
                "embedding_id": _embedding_id(emb),
                "embedding_fingerprint": [round(float(v), 5) for v in emb[:12].tolist()],
                "source_role": source_role,
                "_embedding": emb,
            }
        )

    faces.sort(key=lambda f: f["bbox"]["w"] * f["bbox"]["h"], reverse=True)

    if not faces:
        status = "NOT_DETECTED"
    elif len(faces) == 1:
        status = "DETECTED"
    else:
        status = "MULTIPLE_DETECTED"

    return {
        "status": status,
        "face_count": len(faces),
        "faces": faces,
        "image_size": {"width": int(w), "height": int(h)},
        "image_sha16": hashlib.sha256(image_bytes).hexdigest()[:16],
        "source_role": source_role,
        "detector": "opencv_haar_frontalface_default",
        "verification": {
            "status": "NOT_ASSESSED",
            "result": "NOT_AVAILABLE",
            "reason": "Detection only — no 1:1 comparison performed.",
        },
        # Never run live PAD inside detect_faces — that would recurse (PAD → detect_faces).
        "liveness": (
            assess_passive_liveness(image_bytes, source="document")
            if source_role in {"document", "unknown"}
            else {
                "status": "NOT_ASSESSED",
                "decision": "NOT_ASSESSED",
                "method": "passive_heuristic_v1",
                "note": "PAD is assessed on the camera capture path, not inside detect_faces.",
            }
        ),
        "camera": {
            "status": "NOT_USED",
            "note": "Live camera capture available via /api/camera/* (SIH Phase 12). Not used on document-only analyze.",
        },
        "concern": "DETECTION_ONLY",
        "calibrated": False,
        "note": (
            "Face detection / quality are review aids only. "
            "DETECTED ≠ MATCH. Detection does not set authenticity_estimate."
        ),
    }


def _public_faces(faces: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for f in faces:
        pub = {k: v for k, v in f.items() if k != "_embedding"}
        out.append(pub)
    return out


def _face_summary(block: Dict[str, Any]) -> Dict[str, Any]:
    faces = block.get("faces") or []
    primary = faces[0] if faces else None
    return {
        "detected": block.get("status") == "DETECTED",
        "status": block.get("status"),
        "face_count": block.get("face_count", 0),
        "image_sha16": block.get("image_sha16"),
        "source_role": block.get("source_role"),
        "primary_bbox": (primary or {}).get("bbox"),
        "primary_quality": (primary or {}).get("quality"),
        "embedding_id": (primary or {}).get("embedding_id"),
        "embedding_dim": (primary or {}).get("embedding_dim"),
        "embedding_fingerprint": (primary or {}).get("embedding_fingerprint"),
    }


def compare_faces(
    document_bytes: bytes,
    reference_bytes: Optional[bytes] = None,
) -> Dict[str, Any]:
    """
    Detect face(s) on document. If live/reference provided, compute 1:1 similarity.

    DETECTED ≠ MATCH. PASS must never be inferred from detection alone.
    """
    doc = detect_faces(document_bytes, source_role="document")
    if doc.get("status") == "ERROR":
        doc["faces"] = _public_faces(doc.get("faces") or [])
        doc["document_face"] = _face_summary(doc)
        doc["face_verification"] = "ERROR"
        return doc

    result = dict(doc)
    result["faces"] = _public_faces(doc.get("faces") or [])
    result["document_face"] = _face_summary({**doc, "faces": doc.get("faces") or []})

    if reference_bytes is None:
        result["face_verification"] = "NOT_AVAILABLE"
        result["live_face"] = None
        result["verification"] = {
            "status": "NOT_ASSESSED",
            "result": "NOT_AVAILABLE",
            "reason": "No live/reference image provided for 1:1 compare.",
        }
        return result

    # Guard: identical image bytes ⇒ self-comparison (never a real 1:1)
    same_bytes = hashlib.sha256(document_bytes).digest() == hashlib.sha256(reference_bytes).digest()

    live = detect_faces(reference_bytes, source_role="live")
    result["live_face"] = _face_summary(live)
    result["reference"] = {
        "status": live.get("status"),
        "face_count": live.get("face_count"),
        "faces": _public_faces(live.get("faces") or []),
        "source_role": "live",
        "image_sha16": live.get("image_sha16"),
    }

    doc_count = int(doc.get("face_count") or 0)
    live_count = int(live.get("face_count") or 0)

    if doc_count == 0 or live_count == 0:
        result["verification"] = {
            "status": "NOT_ASSESSED",
            "result": "NOT_AVAILABLE",
            "reason": "Need exactly one detectable face on both document and live capture.",
            "document_face_status": doc.get("status"),
            "live_face_status": live.get("status"),
            "calibration": "NOT_ASSESSED",
            "calibrated": False,
        }
        result["face_verification"] = "NOT_AVAILABLE"
        result["concern"] = "DETECTION_ONLY"
        result["verification_debug"] = {
            "document_face": result["document_face"],
            "live_face": result["live_face"],
            "embedding_sources": "DIFFERENT" if not same_bytes else "SAME_IMAGE_BYTES",
            "similarity": None,
            "metric": METRIC_NAME,
            "threshold_match_min": MATCH_SIMILARITY_MIN,
            "threshold_no_match_max": NO_MATCH_SIMILARITY_MAX,
            "calibration": "UNCALIBRATED",
            "verification": "NOT_AVAILABLE",
        }
        return result

    if doc_count > 1 or live_count > 1:
        result["verification"] = {
            "status": "NOT_ASSESSED",
            "result": "INCONCLUSIVE",
            "reason": "Multiple faces detected — normal 1:1 verification not performed.",
            "document_face_count": doc_count,
            "live_face_count": live_count,
            "calibration": "UNCALIBRATED",
            "calibrated": False,
        }
        result["face_verification"] = "INCONCLUSIVE"
        result["concern"] = "MULTIPLE_FACES"
        return result

    a = doc["faces"][0]["_embedding"]
    b = live["faces"][0]["_embedding"]
    emb_id_a = doc["faces"][0].get("embedding_id")
    emb_id_b = live["faces"][0].get("embedding_id")
    # Exact vector equality only — rounded fingerprint ids can collide on near-copies
    identical_emb = bool(np.array_equal(a, b))

    raw_cos = _cosine(a, b)
    sim = _clip01((raw_cos + 1.0) / 2.0)

    # Self-compare / cached-embedding guard (same image bytes or exact same vector)
    if same_bytes or identical_emb:
        match_result = "INCONCLUSIVE"
        technical_decision = "INCONCLUSIVE_SELF_COMPARE_GUARD"
        match_label = "REVIEW"
        officer_note = (
            "Document and live embeddings appear identical or came from the same image bytes — "
            "refusing MATCH (possible self-comparison / cache bug)."
        )
    elif sim >= MATCH_SIMILARITY_MIN:
        # Configured band for this prototype metric — still UNCALIBRATED
        # Extra guard: refuse MATCH if either crop quality is POOR
        doc_q = ((doc.get("faces") or [{}])[0].get("quality") or {}).get("label")
        live_q = ((live.get("faces") or [{}])[0].get("quality") or {}).get("label")
        if doc_q == "POOR" or live_q == "POOR":
            match_result = "INCONCLUSIVE"
            technical_decision = "INCONCLUSIVE_LOW_QUALITY"
            match_label = "REVIEW"
            officer_note = (
                f"Similarity {sim:.3f} is high but face quality is POOR — refusing MATCH. "
                "Detection alone is never MATCH."
            )
        else:
            match_result = "MATCH"
            technical_decision = "INCONCLUSIVE_HIGH_SIMILARITY"
            match_label = "MATCH"
            officer_note = (
                f"Prototype fingerprint similarity {sim:.3f} ≥ {MATCH_SIMILARITY_MIN} → MATCH band. "
                "Calibration=UNCALIBRATED — not legal identity proof. DETECTED ≠ MATCH."
            )
    elif sim <= NO_MATCH_SIMILARITY_MAX:
        match_result = "NO_MATCH"
        technical_decision = "INCONCLUSIVE_LOW_SIMILARITY"
        match_label = "NO_MATCH"
        officer_note = (
            f"Prototype fingerprint similarity {sim:.3f} ≤ {NO_MATCH_SIMILARITY_MAX} → NO_MATCH. "
            "Document face and live face are not similar under the configured rule."
        )
    else:
        match_result = "INCONCLUSIVE"
        technical_decision = "INCONCLUSIVE"
        match_label = "REVIEW"
        officer_note = (
            f"Similarity {sim:.3f} is between NO_MATCH and MATCH bands — "
            "no validated biometric decision."
        )

    debug = {
        "document_face": result["document_face"],
        "live_face": result["live_face"],
        "document_embedding": "GENERATED",
        "live_embedding": "GENERATED",
        "document_embedding_id": emb_id_a,
        "live_embedding_id": emb_id_b,
        "embedding_sources": "SAME_IMAGE_BYTES" if same_bytes else ("IDENTICAL_VECTOR" if identical_emb else "DIFFERENT"),
        "embedding_method": EMBEDDING_METHOD,
        "embedding_dim": EMBEDDING_DIM,
        "raw_cosine": round(raw_cos, 6),
        "similarity": round(sim, 4),
        "metric": METRIC_NAME,
        "threshold_match_min": MATCH_SIMILARITY_MIN,
        "threshold_no_match_max": NO_MATCH_SIMILARITY_MAX,
        "threshold_source": "configured_prototype_bands_v2",
        "calibration": "UNCALIBRATED",
        "verification": match_result,
        "note": (
            "Similarity is not an identity probability. "
            "NO VALIDATED FACE VERIFICATION THRESHOLD (biometric calibration absent)."
        ),
    }

    result_block = {
        "status": "COMPARED_UNCALIBRATED",
        "result": match_result,
        "decision": technical_decision,
        "similarity": round(sim, 4),
        "calibrated": False,
        "calibration": "UNCALIBRATED",
        "age_robustness": "NOT_VALIDATED",
        "match_label": match_label,
        "metric": METRIC_NAME,
        "threshold_match_min": MATCH_SIMILARITY_MIN,
        "threshold_no_match_max": NO_MATCH_SIMILARITY_MAX,
        "embedding_method": EMBEDDING_METHOD,
        "document_embedding_id": emb_id_a,
        "live_embedding_id": emb_id_b,
        "embedding_sources": debug["embedding_sources"],
        "threshold_note": (
            f"Prototype bands: ≥{MATCH_SIMILARITY_MIN} → MATCH, ≤{NO_MATCH_SIMILARITY_MAX} → NO_MATCH. "
            "Calibration=UNCALIBRATED — not a legal identity match. "
            "Similarity ≠ P(same person)."
        ),
        "concern": "UNCALIBRATED_SIMILARITY",
        "officer_note": officer_note,
        "validated_threshold": False,
        "validated_threshold_note": "NO VALIDATED FACE VERIFICATION THRESHOLD",
    }
    result["verification"] = result_block
    result["verification_debug"] = debug
    result["face_verification"] = match_result  # MATCH | NO_MATCH | INCONCLUSIVE — never DETECTED
    result["concern"] = "UNCALIBRATED_SIMILARITY"
    result["note"] = (
        "1:1 face compare uses prototype fingerprint bands (UNCALIBRATED). "
        "DETECTED ≠ MATCH. Liveness/PAD ≠ identity. authenticity_estimate unchanged."
    )
    return result


def analyze_document_face(image_bytes: bytes) -> Dict[str, Any]:
    """Pipeline helper — detection only (no reference in analyze flow)."""
    result = detect_faces(image_bytes, source_role="document")
    result["faces"] = _public_faces(result.get("faces") or [])
    result["document_face"] = _face_summary({**result, "faces": result.get("faces") or []})
    # Never promote detection status into a verification MATCH
    result["face_verification"] = "NOT_AVAILABLE"
    result["verification"] = {
        "status": "NOT_ASSESSED",
        "result": "NOT_AVAILABLE",
        "reason": "Document analyze is detection-only. Live capture required for 1:1 MATCH/NO_MATCH.",
    }
    return result
