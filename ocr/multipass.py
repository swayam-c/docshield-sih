"""Multi-pass OCR preprocessing variants — pick best by OCR confidence evidence."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

from ocr.preprocess import decode_bgr, _deskew


def build_ocr_variants(image_bytes: bytes, max_side: int = 1800) -> Dict[str, Any]:
    """Build named grayscale variants for multi-pass OCR."""
    bgr = decode_bgr(image_bytes)
    h, w = bgr.shape[:2]
    scale = 1.0
    longest = max(h, w)
    if longest > max_side:
        scale = max_side / float(longest)
        bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    elif longest < 900:
        scale = 900 / float(longest)
        bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

    gray0 = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray, skew = _deskew(gray0)
    den = cv2.fastNlMeansDenoising(gray, None, h=8, templateWindowSize=7, searchWindowSize=21)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast = clahe.apply(den)

    # Sharpen
    blur = cv2.GaussianBlur(contrast, (0, 0), 1.2)
    sharpen = cv2.addWeighted(contrast, 1.5, blur, -0.5, 0)

    binary = cv2.adaptiveThreshold(
        contrast, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11
    )

    # Upscale for small text
    up = cv2.resize(contrast, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)

    variants = {
        "original_gray": gray,
        "contrast": contrast,
        "sharpen": sharpen,
        "adaptive_threshold": binary,
        "upscale": up,
    }
    return {
        "bgr": bgr,
        "variants": variants,
        "scale": scale,
        "skew_degrees": round(float(skew), 3),
        "width": int(contrast.shape[1]),
        "height": int(contrast.shape[0]),
        # keep legacy keys used by existing callers
        "gray": contrast,
        "binary": binary,
    }


def score_pass(boxes: List[Any]) -> float:
    """Evidence score: mean confidence × log(1+count). Prefer more confident text."""
    if not boxes:
        return 0.0
    confs = [float(getattr(b, "confidence", b.get("confidence", 0) if isinstance(b, dict) else 0)) for b in boxes]
    mean_c = float(np.mean(confs)) if confs else 0.0
    return mean_c * float(np.log1p(len(boxes)))
