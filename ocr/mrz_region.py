"""MRZ region detection on passport images — crop bottom band and OCR.

Never fabricates MRZ. Returns NOT_DETECTED when lines cannot be read.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ocr.mrz import analyze_mrz, extract_mrz_candidates
from ocr.preprocess import decode_bgr, preprocess_for_ocr


def _ocr_mrz_band(gray: np.ndarray) -> Tuple[str, List[Dict[str, Any]]]:
    """Run Tesseract on an MRZ-like band with restrictive charset."""
    try:
        import pytesseract
        from pytesseract import Output
    except ImportError:
        return "", []

    # Upscale thin MRZ bands
    h, w = gray.shape[:2]
    if h < 80:
        scale = 120 / max(h, 1)
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

    # Binary helps OCR on OCR-B style fonts
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    config = r"--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"
    boxes: List[Dict[str, Any]] = []
    texts: List[str] = []
    for img in (binary, gray):
        try:
            data = pytesseract.image_to_data(img, output_type=Output.DICT, config=config)
        except Exception:
            continue
        n = len(data["text"])
        line_map: Dict[int, List[str]] = {}
        for i in range(n):
            raw = (data["text"][i] or "").strip()
            if not raw:
                continue
            try:
                conf = float(data["conf"][i])
            except (TypeError, ValueError):
                conf = -1
            if conf < 0:
                continue
            ln = int(data.get("line_num", [0])[i])
            line_map.setdefault(ln, []).append(raw)
            boxes.append(
                {
                    "text": raw,
                    "confidence": round(conf / 100.0, 4),
                    "bbox": [
                        int(data["left"][i]),
                        int(data["top"][i]),
                        int(data["width"][i]),
                        int(data["height"][i]),
                    ],
                }
            )
        for ln in sorted(line_map):
            texts.append("".join(line_map[ln]))
        if texts:
            break

    # Also full string
    try:
        raw_full = pytesseract.image_to_string(binary, config=config)
        if raw_full.strip():
            texts.append(raw_full)
    except Exception:
        pass

    return "\n".join(texts), boxes


def detect_and_parse_mrz_from_image(
    image_bytes: bytes,
    document_type: str,
    supports_mrz: bool,
    fields_payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    1) Try MRZ from full-image OCR text (caller may pass via fields)
    2) Crop bottom ~28% of page and OCR with MRZ whitelist
    3) Parse TD3/TD2/TD1 — never invent
    """
    if not supports_mrz:
        return analyze_mrz("", document_type, False, fields_payload=fields_payload)

    # Bottom-band crop
    try:
        bgr = decode_bgr(image_bytes)
    except ValueError:
        return {
            "status": "NOT_ASSESSED",
            "message": "Could not decode image for MRZ region detection.",
            "checklist": [],
        }

    h, w = bgr.shape[:2]
    y0 = int(h * 0.68)
    band = bgr[y0:h, 0:w]
    gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)

    # Also try a slightly higher band (some scans crop bottom)
    y0b = int(h * 0.55)
    band2 = bgr[y0b:h, 0:w]
    gray2 = cv2.cvtColor(band2, cv2.COLOR_BGR2GRAY)

    combined_text = ""
    all_boxes: List[Dict[str, Any]] = []
    for g in (gray, gray2):
        t, boxes = _ocr_mrz_band(g)
        if t:
            combined_text += "\n" + t
            all_boxes.extend(boxes)

    # Merge with any preprocess full-page OCR text already in fields? handled by caller
    candidates = extract_mrz_candidates(combined_text)
    result = analyze_mrz(
        combined_text,
        document_type if document_type != "UNKNOWN" else "PASSPORT",
        True,
        fields_payload=fields_payload,
    )

    # Enrich with region metadata
    result["region"] = {
        "method": "bottom_band_crop",
        "y_start_ratio": 0.68,
        "band_height_px": int(h - y0),
        "candidate_line_count": len(candidates),
        "ocr_box_count": len(all_boxes),
    }
    if result.get("status") == "PARSED":
        result["status"] = "DETECTED"  # SIH passport pipeline naming
        result["format_valid"] = bool(result.get("structure_valid"))
        result["check_digits_valid"] = bool(result.get("check_digits_ok"))
        cons = result.get("consistency") or {}
        result["name_consistency"] = (cons.get("name") or {}).get("status", "NOT_ASSESSED")
        result["dob_consistency"] = (cons.get("date_of_birth") or {}).get("status", "NOT_ASSESSED")
        result["document_number_consistency"] = (
            cons.get("document_number") or {}
        ).get("status", "NOT_ASSESSED")
    elif result.get("status") == "NOT_DETECTED":
        result["format_valid"] = None
        result["check_digits_valid"] = None
        result["name_consistency"] = "NOT_ASSESSED"
        result["dob_consistency"] = "NOT_ASSESSED"
        result["document_number_consistency"] = "NOT_ASSESSED"

    return result
