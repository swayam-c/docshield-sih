"""Image quality analysis — evidence suitability, not authenticity."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Tuple

import cv2
import numpy as np


# overall_quality below this → INCONCLUSIVE (insufficient evidence quality)
QUALITY_INCONCLUSIVE_THRESHOLD = 0.45


@dataclass
class QualityReport:
    resolution: str
    blur: str
    brightness: str
    contrast: str
    noise: str
    rotation: str
    perspective: str
    crop: str
    document_visibility: str
    compression: str
    overall_quality: float
    # SIH-facing percentage scores (0–100), derived from measured metrics
    scores: Dict[str, float]
    metrics: Dict[str, float]
    insufficient_for_analysis: bool
    reason: str | None = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)

        def _finite(x: Any, default: float = 0.0) -> float:
            try:
                v = float(x)
            except (TypeError, ValueError):
                return default
            if v != v or v in (float("inf"), float("-inf")):  # NaN/Inf
                return default
            return v

        data["overall_quality"] = _finite(data.get("overall_quality"), 0.0)
        data["scores"] = {
            k: _finite(v, 0.0) for k, v in (data.get("scores") or {}).items()
        }
        data["metrics"] = {
            k: _finite(v, 0.0) for k, v in (data.get("metrics") or {}).items()
        }
        data["note"] = (
            "overall_quality and scores measure evidence/image suitability for analysis, "
            "NOT document authenticity probability."
        )
        return data


def _label_from_thresholds(
    value: float,
    *,
    good: float,
    fair: float,
    labels: Tuple[str, str, str] = ("GOOD", "FAIR", "POOR"),
    higher_is_better: bool = True,
) -> str:
    if higher_is_better:
        if value >= good:
            return labels[0]
        if value >= fair:
            return labels[1]
        return labels[2]
    if value <= good:
        return labels[0]
    if value <= fair:
        return labels[1]
    return labels[2]


def _decode_bgr(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Unable to decode image for quality analysis.")
    return img


def _estimate_skew_degrees(gray: np.ndarray) -> float:
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=max(80, gray.shape[0] // 8))
    if lines is None:
        return 0.0
    angles = []
    for rho_theta in lines[:80]:
        rho, theta = rho_theta[0]
        deg = (theta * 180.0 / np.pi) - 90.0
        if abs(deg) < 45:
            angles.append(deg)
    if not angles:
        return 0.0
    return float(np.median(angles))


def _document_quad_score(gray: np.ndarray) -> Tuple[float, float]:
    h, w = gray.shape[:2]
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thr = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        edges = cv2.Canny(gray, 40, 120)
        density = float(np.mean(edges > 0))
        vis = min(1.0, density * 8.0)
        return vis, 0.55

    page = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(page))
    frame = float(h * w)
    coverage = area / frame if frame else 0.0

    peri = cv2.arcLength(page, True)
    approx = cv2.approxPolyDP(page, 0.02 * peri, True)
    perspective = 0.55
    if len(approx) == 4 and coverage > 0.15:
        pts = approx.reshape(4, 2).astype(np.float32)
        rect = cv2.minAreaRect(pts)
        (_, _), (rw, rh), _ = rect
        if rw > 1 and rh > 1:
            ratio = max(rw, rh) / min(rw, rh)
            rectangularity = 1.0 - min(abs(ratio - 1.586) / 2.0, 1.0)
            perspective = 0.45 + 0.55 * rectangularity
        x, y, bw, bh = cv2.boundingRect(approx)
        box_area = float(bw * bh) or 1.0
        fill = area / box_area
        perspective = float(np.clip(0.4 * perspective + 0.6 * fill, 0.0, 1.0))
    elif coverage > 0.35:
        perspective = 0.65
    else:
        perspective = 0.45

    visibility = float(np.clip(coverage * 1.25, 0.0, 1.0))
    return visibility, perspective


def _crop_score(gray: np.ndarray, visibility: float) -> Tuple[str, float]:
    h, w = gray.shape[:2]
    border = max(2, min(h, w) // 40)
    edges = cv2.Canny(gray, 40, 120)
    top = float(np.mean(edges[:border, :] > 0))
    bottom = float(np.mean(edges[-border:, :] > 0))
    left = float(np.mean(edges[:, :border] > 0))
    right = float(np.mean(edges[:, -border:] > 0))
    border_activity = (top + bottom + left + right) / 4.0
    if visibility < 0.35:
        return "SEVERE", 0.25
    if border_activity > 0.22 and visibility < 0.55:
        return "PARTIAL", 0.55
    return "GOOD", 0.9


def _compression_score(gray: np.ndarray) -> Tuple[str, float, float]:
    """
    Estimate compression / blockiness via 8x8 grid high-frequency energy drop.
    Higher score = cleaner (fewer blocking artifacts).
    """
    h, w = gray.shape[:2]
    # Work on a moderate crop for speed
    sample = gray
    if h > 512 or w > 512:
        sample = cv2.resize(gray, (min(w, 512), min(h, 512)), interpolation=cv2.INTER_AREA)
    sh, sw = sample.shape[:2]
    sh8, sw8 = sh - (sh % 8), sw - (sw % 8)
    if sh8 < 16 or sw8 < 16:
        return "UNKNOWN", 0.6, 0.0
    sample = sample[:sh8, :sw8].astype(np.float32)
    # Horizontal adjacent 8-boundary differences vs interior
    boundary = []
    interior = []
    for x in range(8, sw8, 8):
        boundary.append(np.mean(np.abs(sample[:, x] - sample[:, x - 1])))
        if x + 3 < sw8:
            interior.append(np.mean(np.abs(sample[:, x + 2] - sample[:, x + 1])))
    for y in range(8, sh8, 8):
        boundary.append(np.mean(np.abs(sample[y, :] - sample[y - 1, :])))
        if y + 3 < sh8:
            interior.append(np.mean(np.abs(sample[y + 2, :] - sample[y + 1, :])))
    b = float(np.mean(boundary)) if boundary else 0.0
    i = float(np.mean(interior)) if interior else 1.0
    blockiness = b / (i + 1e-6)
    # blockiness ~1 is natural; >>1 suggests grid artifacts
    score = float(np.clip(1.0 - max(0.0, blockiness - 1.0) / 2.5, 0.0, 1.0))
    if score >= 0.75:
        label = "LOW"
    elif score >= 0.45:
        label = "MEDIUM"
    else:
        label = "HIGH"
    return label, score, round(blockiness, 4)


def analyze_image_quality(image_bytes: bytes) -> QualityReport:
    """
    Compute image/evidence quality metrics.

    overall_quality / scores are NOT authenticity probability.
    """
    bgr = _decode_bgr(image_bytes)
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    megapixels = (w * h) / 1_000_000.0
    min_side = min(w, h)
    # Continuous resolution score for % display
    res_score = float(
        np.clip(0.45 * (min_side / 900.0) + 0.55 * (megapixels / 1.2), 0.0, 1.0)
    )
    if min_side >= 900 and megapixels >= 0.8:
        resolution = "GOOD"
    elif min_side >= 500 and megapixels >= 0.25:
        resolution = "FAIR"
    else:
        resolution = "POOR"

    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    blur_score = float(np.clip(lap_var / 250.0, 0.0, 1.0))
    if lap_var >= 120:
        blur = "LOW"
    elif lap_var >= 50:
        blur = "MEDIUM"
    else:
        blur = "HIGH"

    mean_br = float(np.mean(gray))
    std_br = float(np.std(gray))
    brightness_score = float(np.clip(1.0 - abs(mean_br - 127.0) / 127.0, 0.0, 1.0))
    if 70 <= mean_br <= 185:
        brightness = "GOOD"
    elif 45 <= mean_br <= 210:
        brightness = "FAIR"
    else:
        brightness = "POOR"

    contrast_score = float(np.clip(std_br / 65.0, 0.0, 1.0))
    contrast = _label_from_thresholds(std_br, good=45, fair=25, labels=("GOOD", "FAIR", "POOR"))
    lighting_score = float(np.clip(0.6 * brightness_score + 0.4 * contrast_score, 0.0, 1.0))

    den = cv2.GaussianBlur(gray, (3, 3), 0)
    residual = cv2.absdiff(gray, den)
    noise_level = float(np.mean(residual))
    noise_score = float(np.clip(1.0 - noise_level / 18.0, 0.0, 1.0))
    if noise_level <= 4.5:
        noise = "LOW"
    elif noise_level <= 9.0:
        noise = "MEDIUM"
    else:
        noise = "HIGH"

    skew = abs(_estimate_skew_degrees(gray))
    rotation_score = float(np.clip(1.0 - skew / 20.0, 0.0, 1.0))
    if skew <= 3.0:
        rotation = "GOOD"
    elif skew <= 10.0:
        rotation = "MILD"
    else:
        rotation = "SEVERE"

    visibility, perspective_score = _document_quad_score(gray)
    if perspective_score >= 0.75:
        perspective = "GOOD"
    elif perspective_score >= 0.5:
        perspective = "FAIR"
    else:
        perspective = "POOR"

    if visibility >= 0.7:
        document_visibility = "GOOD"
    elif visibility >= 0.4:
        document_visibility = "FAIR"
    else:
        document_visibility = "POOR"

    crop_label, crop_score = _crop_score(gray, visibility)
    compression_label, compression_score, blockiness = _compression_score(gray)

    overall = float(
        np.clip(
            0.16 * res_score
            + 0.20 * blur_score
            + 0.10 * lighting_score
            + 0.08 * noise_score
            + 0.08 * rotation_score
            + 0.12 * perspective_score
            + 0.06 * crop_score
            + 0.08 * visibility
            + 0.12 * compression_score,
            0.0,
            1.0,
        )
    )
    overall = round(overall, 4)

    scores = {
        "resolution": round(res_score * 100, 1),
        "sharpness": round(blur_score * 100, 1),
        "lighting": round(lighting_score * 100, 1),
        "perspective": round(perspective_score * 100, 1),
        "noise": round(noise_score * 100, 1),
        "rotation": round(rotation_score * 100, 1),
        "compression": round(compression_score * 100, 1),
        "visibility": round(visibility * 100, 1),
        "overall_quality": round(overall * 100, 1),
    }

    insufficient = overall < QUALITY_INCONCLUSIVE_THRESHOLD or blur == "HIGH" or resolution == "POOR"
    reason = None
    if insufficient:
        reasons = []
        if blur == "HIGH":
            reasons.append("Image is too blurry for reliable analysis")
        if resolution == "POOR":
            reasons.append("Resolution is too low")
        if brightness == "POOR":
            reasons.append("Brightness is unsuitable")
        if document_visibility == "POOR":
            reasons.append("Document is not clearly visible")
        if overall < QUALITY_INCONCLUSIVE_THRESHOLD and not reasons:
            reasons.append("Overall evidence quality is insufficient")
        reason = (
            "Image quality is insufficient for reliable verification. "
            + "; ".join(reasons)
            + ". Upload a clearer full-document image."
        )

    return QualityReport(
        resolution=resolution,
        blur=blur,
        brightness=brightness,
        contrast=contrast,
        noise=noise,
        rotation=rotation,
        perspective=perspective,
        crop=crop_label,
        document_visibility=document_visibility,
        compression=compression_label,
        overall_quality=overall,
        scores=scores,
        metrics={
            "width": float(w),
            "height": float(h),
            "megapixels": round(megapixels, 4),
            "laplacian_variance": round(lap_var, 3),
            "mean_brightness": round(mean_br, 3),
            "contrast_std": round(std_br, 3),
            "noise_residual_mean": round(noise_level, 3),
            "skew_degrees": round(skew, 3),
            "perspective_score": round(perspective_score, 4),
            "visibility_score": round(visibility, 4),
            "crop_score": round(crop_score, 4),
            "compression_blockiness": blockiness,
            "compression_score": round(compression_score, 4),
        },
        insufficient_for_analysis=insufficient,
        reason=reason,
    )
