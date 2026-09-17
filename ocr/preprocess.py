"""OCR preprocessing — enhance readability without claiming authenticity gains."""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def decode_bgr(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Unable to decode image for OCR.")
    return img


def _deskew(gray: np.ndarray) -> Tuple[np.ndarray, float]:
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=max(60, gray.shape[0] // 10))
    angle = 0.0
    if lines is not None:
        angles = []
        for item in lines[:60]:
            _, theta = item[0]
            deg = (theta * 180.0 / np.pi) - 90.0
            if abs(deg) < 20:
                angles.append(deg)
        if angles:
            angle = float(np.median(angles))
    if abs(angle) < 0.4:
        return gray, 0.0
    h, w = gray.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rotated = cv2.warpAffine(
        gray, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )
    return rotated, angle


def preprocess_for_ocr(image_bytes: bytes, max_side: int = 1800) -> dict:
    """
    Returns preprocessed grayscale image and metadata.
    """
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

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray, skew = _deskew(gray)
    gray = cv2.fastNlMeansDenoising(gray, None, h=8, templateWindowSize=7, searchWindowSize=21)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    # Light adaptive threshold variant kept as alternate channel for OCR
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11
    )

    return {
        "bgr": bgr,
        "gray": gray,
        "binary": binary,
        "scale": scale,
        "skew_degrees": round(skew, 3),
        "width": int(gray.shape[1]),
        "height": int(gray.shape[0]),
    }
