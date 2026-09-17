"""Text visual / typography consistency + text-region splicing cues.

Review signals only — never proof of fraud.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ocr.preprocess import decode_bgr


def _bbox_xywh(box: Dict[str, Any]) -> Optional[Tuple[int, int, int, int]]:
    bb = box.get("bbox")
    if not bb or len(bb) < 4:
        return None
    return int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])


def _detect_text_like_regions(gray: np.ndarray, max_regions: int = 24) -> List[Dict[str, Any]]:
    """Find ink-like connected components when OCR boxes are unavailable.

    These are visual geometry probes for typography review — not OCR transcripts.
    """
    h, w = gray.shape[:2]
    # Adaptive threshold → connected components that look like text lines/glyphs
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    bw = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 11
    )
    # Prefer horizontal text-ish shapes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, w // 80), 2))
    merged = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=1)
    n, _, stats, _ = cv2.connectedComponentsWithStats(merged, connectivity=8)
    candidates: List[Tuple[float, Dict[str, Any]]] = []
    min_area = max(40, int(0.00015 * h * w))
    max_area = int(0.18 * h * w)
    for i in range(1, n):
        x, y, bw_, bh_, area = (int(v) for v in stats[i])
        if area < min_area or area > max_area:
            continue
        if bw_ < 8 or bh_ < 6:
            continue
        aspect = bw_ / float(max(1, bh_))
        if aspect < 0.8 or aspect > 28.0:
            continue
        # Prefer mid-band document content (avoid borders)
        cy = (y + bh_ / 2.0) / float(max(1, h))
        if cy < 0.04 or cy > 0.96:
            continue
        score = area * min(aspect, 8.0)
        candidates.append(
            (
                score,
                {
                    "text": f"visual_region_{len(candidates) + 1}",
                    "confidence": None,
                    "bbox": [x, y, bw_, bh_],
                    "source": "visual_region_fallback",
                    "status": "VISUAL_PROBE",
                },
            )
        )
    candidates.sort(key=lambda t: t[0], reverse=True)
    out: List[Dict[str, Any]] = []
    for _, box in candidates[:max_regions]:
        box["text"] = f"visual_region_{len(out) + 1}"
        out.append(box)
    return out


def analyze_text_visual(
    image_bytes: bytes,
    text_boxes: Optional[List[Dict[str, Any]]] = None,
    fields: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    text_boxes = list(text_boxes or [])
    fields = fields or {}
    region_source = "ocr"

    try:
        bgr = decode_bgr(image_bytes)
    except ValueError as exc:
        return {
            "status": "ERROR",
            "error": str(exc),
            "typography": {"status": "NOT_ASSESSED"},
            "text_splicing": {"status": "NOT_ASSESSED"},
            "ocr_visual_crosscheck": [],
            "concern": "NOT_ASSESSED",
        }

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    doc_h = float(max(1, h))
    doc_w = float(max(1, w))

    if len(text_boxes) < 4:
        had_ocr = bool(text_boxes)
        visual = _detect_text_like_regions(gray)
        if visual:
            existing = {tuple(b.get("bbox") or []) for b in text_boxes}
            for vb in visual:
                key = tuple(vb.get("bbox") or [])
                if key in existing:
                    continue
                text_boxes.append(vb)
                existing.add(key)
            region_source = "ocr+visual" if had_ocr else "visual_region_fallback"

    regions: List[Dict[str, Any]] = []
    heights: List[float] = []
    for box in text_boxes:
        xywh = _bbox_xywh(box)
        if not xywh:
            continue
        x, y, bw, bh = xywh
        if bw < 2 or bh < 2:
            continue
        rel_h = bh / doc_h
        rel_w = bw / doc_w
        heights.append(rel_h)
        crop = gray[max(0, y) : min(h, y + bh), max(0, x) : min(w, x + bw)]
        sharp = float(cv2.Laplacian(crop, cv2.CV_64F).var()) if crop.size else 0.0
        contrast = float(np.std(crop)) if crop.size else 0.0
        regions.append(
            {
                "text": box.get("text"),
                "bbox": [x, y, bw, bh],
                "relative_height": round(rel_h, 5),
                "relative_width": round(rel_w, 5),
                "aspect_ratio": round(bw / float(max(1, bh)), 4),
                "local_sharpness": round(sharp, 2),
                "local_contrast": round(contrast, 2),
                "baseline_y": y + bh,
                "left_x": x,
            }
        )

    typography: Dict[str, Any] = {"status": "NOT_ASSESSED", "anomalies": []}
    if len(heights) >= 3:
        med = float(np.median(heights))
        mad = float(np.median(np.abs(np.asarray(heights) - med))) + 1e-9
        anomalies = []
        for r in regions:
            if abs(r["relative_height"] - med) / mad > 6.0 and r["relative_height"] > med * 2.5:
                anomalies.append(
                    {
                        "text": r["text"],
                        "status": "ANOMALOUS",
                        "reason": (
                            "Relative character-region height differs substantially "
                            "from comparable text regions."
                        ),
                        "relative_height": r["relative_height"],
                        "median_relative_height": round(med, 5),
                    }
                )
        # Alignment: check left-x clustering variance
        lefts = [r["left_x"] / doc_w for r in regions]
        align_std = float(np.std(lefts)) if lefts else 0.0
        note = (
            "Typography consistency is a review signal, not font-family ID or fraud proof."
        )
        if region_source == "visual_region_fallback":
            note = (
                "Measured from visual ink-region probes (OCR boxes unavailable). "
                "Not claimed font-family identification or OCR transcript."
            )
        elif region_source == "ocr+visual":
            note = (
                "Measured from OCR regions supplemented with visual ink probes. "
                "Not claimed font-family identification."
            )
        typography = {
            "status": "ANALYZED",
            "region_count": len(regions),
            "region_source": region_source,
            "median_relative_height": round(med, 5),
            "left_alignment_std": round(align_std, 4),
            "mean_local_sharpness": round(float(np.mean([r["local_sharpness"] for r in regions])), 2) if regions else None,
            "mean_local_contrast": round(float(np.mean([r["local_contrast"] for r in regions])), 2) if regions else None,
            "mean_relative_width": round(float(np.mean([r["relative_width"] for r in regions])), 5) if regions else None,
            "regions": regions[:24],
            "anomalies": anomalies[:12],
            "consistency_label": "ANOMALY_SIGNAL" if anomalies else "CONSISTENT",
            "note": note,
        }
    elif regions:
        lefts = [r["left_x"] / doc_w for r in regions]
        typography = {
            "status": "UNCERTAIN",
            "region_count": len(regions),
            "region_source": region_source,
            "median_relative_height": round(float(np.median(heights)), 5) if heights else None,
            "left_alignment_std": round(float(np.std(lefts)), 4) if lefts else None,
            "mean_local_sharpness": round(float(np.mean([r["local_sharpness"] for r in regions])), 2),
            "mean_local_contrast": round(float(np.mean([r["local_contrast"] for r in regions])), 2),
            "mean_relative_width": round(float(np.mean([r["relative_width"] for r in regions])), 5),
            "regions": regions[:24],
            "anomalies": [],
            "consistency_label": "INSUFFICIENT_REGIONS",
            "note": "Too few text regions for robust relative size analysis.",
        }

    # Text-region splicing cue: residual energy vs global
    splicing_flags: List[Dict[str, Any]] = []
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    residual = cv2.absdiff(gray, blur).astype(np.float32)
    global_mean = float(np.mean(residual)) + 1e-6
    for r in regions[:40]:
        x, y, bw, bh = r["bbox"]
        patch = residual[max(0, y) : min(h, y + bh), max(0, x) : min(w, x + bw)]
        if patch.size < 16:
            continue
        local = float(np.mean(patch))
        ratio = local / global_mean
        status = "NO_CLEAR_ANOMALY"
        if ratio > 2.2 or ratio < 0.35:
            status = "ANOMALY_DETECTED"
        elif ratio > 1.7 or ratio < 0.5:
            status = "UNCERTAIN"
        if status != "NO_CLEAR_ANOMALY":
            splicing_flags.append(
                {
                    "region": "text",
                    "evidence_type": "local_residual_vs_global",
                    "score": round(min(1.0, abs(np.log(ratio + 1e-6)) / 2.0), 4),
                    "bbox": r["bbox"],
                    "text": r["text"],
                    "status": status,
                }
            )

    if not regions:
        text_splicing = {"status": "NOT_ASSESSED", "regions": [], "note": "No OCR boxes for text splicing."}
    else:
        text_splicing = {
            "status": "ANALYZED",
            "regions": splicing_flags[:20],
            "anomaly_count": sum(1 for s in splicing_flags if s["status"] == "ANOMALY_DETECTED"),
            "note": "Text-region residual cues are heuristic review signals only.",
        }

    # OCR text ↔ image region cross-check for extracted fields
    cross: List[Dict[str, Any]] = []
    for field in (fields.get("fields") or [])[:20]:
        name = field.get("name") or field.get("field")
        bbox = field.get("bbox") or (field.get("source_region") and [
            field["source_region"].get("x"),
            field["source_region"].get("y"),
            field["source_region"].get("w"),
            field["source_region"].get("h"),
        ])
        if not bbox or any(v is None for v in bbox):
            cross.append(
                {
                    "field": name,
                    "ocr_value_masked": field.get("masked_value") or field.get("value"),
                    "image_region": "NOT_FOUND",
                    "status": "NOT_FOUND",
                }
            )
            continue
        x, y, bw, bh = [int(v) for v in bbox[:4]]
        crop = gray[max(0, y) : min(h, y + max(1, bh)), max(0, x) : min(w, x + max(1, bw))]
        if crop.size < 16:
            status = "UNCERTAIN"
            visual = "INVALID"
        else:
            sharp = float(cv2.Laplacian(crop, cv2.CV_64F).var())
            visual = "VALID" if sharp > 20 else "WEAK"
            status = "CONSISTENT" if visual == "VALID" else "UNCERTAIN"
        cross.append(
            {
                "field": name,
                "ocr_value_masked": field.get("masked_value") or field.get("value"),
                "image_region": "FOUND",
                "visual_region": visual,
                "bbox": [x, y, bw, bh],
                "status": status,
            }
        )

    return {
        "status": "ANALYZED",
        "typography": typography,
        "text_splicing": text_splicing,
        "ocr_visual_crosscheck": cross,
        "sample_regions": regions[:15],
        "concern": "HEURISTIC_REVIEW_ONLY",
        "note": "Text visual / typography / splicing cues are not legal authenticity proof.",
    }
