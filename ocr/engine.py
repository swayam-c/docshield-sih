"""OCR engine — Tesseract primary with multi-pass preprocessing; mock fallback."""

from __future__ import annotations

import os
import re
import shutil
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from ocr.multipass import build_ocr_variants, score_pass
from ocr.preprocess import preprocess_for_ocr


@dataclass
class TextBox:
    text: str
    confidence: float
    bbox: List[int]  # x, y, w, h
    line_num: int = 0
    block_num: int = 0
    source_pass: str = "unknown"
    status: str = "EXTRACTED"

    def to_dict(self) -> Dict[str, Any]:
        x, y, w, h = self.bbox if len(self.bbox) >= 4 else (0, 0, 0, 0)
        return {
            "text": self.text,
            "confidence": self.confidence,
            "bbox": self.bbox,
            "width": int(w),
            "height": int(h),
            "center": [int(x + w / 2), int(y + h / 2)],
            "line_num": self.line_num,
            "block_num": self.block_num,
            "source_pass": self.source_pass,
            "status": self.status,
        }


def tesseract_available() -> bool:
    return shutil.which("tesseract") is not None


def _run_tesseract(gray: np.ndarray, psm: int = 6, source_pass: str = "gray") -> List[TextBox]:
    import pytesseract
    from pytesseract import Output

    config = f"--oem 3 --psm {psm}"
    data = pytesseract.image_to_data(gray, output_type=Output.DICT, config=config)
    boxes: List[TextBox] = []
    n = len(data["text"])
    for i in range(n):
        raw = (data["text"][i] or "").strip()
        if not raw:
            continue
        try:
            conf = float(data["conf"][i])
        except (TypeError, ValueError):
            conf = -1.0
        if conf < 0:
            continue
        boxes.append(
            TextBox(
                text=raw,
                confidence=round(conf / 100.0, 4),
                bbox=[
                    int(data["left"][i]),
                    int(data["top"][i]),
                    int(data["width"][i]),
                    int(data["height"][i]),
                ],
                line_num=int(data.get("line_num", [0])[i]),
                block_num=int(data.get("block_num", [0])[i]),
                source_pass=f"{source_pass}_psm{psm}",
                status="EXTRACTED",
            )
        )
    return boxes


def _merge_box_sets(primary: List[TextBox], secondary: List[TextBox]) -> List[TextBox]:
    seen = {(b.text.upper(), tuple(b.bbox)) for b in primary}
    merged = list(primary)
    for b in secondary:
        key = (b.text.upper(), tuple(b.bbox))
        if key in seen:
            continue
        if any(p.text.upper() == b.text.upper() for p in primary):
            continue
        merged.append(b)
        seen.add(key)
    return merged


def reconstruct_full_text(boxes: List[TextBox]) -> str:
    if not boxes:
        return ""
    ordered = sorted(
        boxes, key=lambda b: (b.block_num, b.line_num, b.bbox[0] if b.bbox else 0)
    )
    lines: List[str] = []
    current_key: Optional[tuple] = None
    current_tokens: List[str] = []
    for b in ordered:
        key = (b.block_num, b.line_num)
        if current_key is None:
            current_key = key
        if key != current_key:
            lines.append(" ".join(current_tokens))
            current_tokens = []
            current_key = key
        current_tokens.append(b.text)
    if current_tokens:
        lines.append(" ".join(current_tokens))
    return "\n".join(lines)


def run_ocr(
    image_bytes: bytes,
    *,
    engine: Optional[str] = None,
) -> Dict[str, Any]:
    chosen = (engine or os.getenv("DOCSHIELD_OCR_ENGINE", "auto")).lower()
    # Prefer multi-pass variants; fall back to legacy preprocess if needed
    try:
        pre = build_ocr_variants(image_bytes)
        multipass = True
    except Exception:
        pre = preprocess_for_ocr(image_bytes)
        multipass = False

    status = "OK"
    backend = "tesseract"
    boxes: List[TextBox] = []
    error = None
    psm_used = [6]
    passes_tried: List[Dict[str, Any]] = []
    best_pass = "contrast"

    use_mock = chosen == "mock" or (chosen == "auto" and not tesseract_available())
    if use_mock:
        backend = "mock"
        status = "NOT_ASSESSED" if chosen == "auto" and not tesseract_available() else "OK"
        boxes = []
        if chosen == "auto" and not tesseract_available():
            error = "Tesseract binary not found; OCR not assessed."
            status = "NOT_ASSESSED"
    else:
        try:
            variants = pre.get("variants") or {
                "gray": pre["gray"],
                "binary": pre.get("binary", pre["gray"]),
            }
            best_score = -1.0
            best_boxes: List[TextBox] = []
            for name, img in variants.items():
                try:
                    cand = _run_tesseract(img, psm=6, source_pass=name)
                    if len(cand) < 6:
                        alt = _run_tesseract(img, psm=4, source_pass=name)
                        if alt:
                            cand = _merge_box_sets(cand, alt)
                            if 4 not in psm_used:
                                psm_used.append(4)
                    sc = score_pass(cand)
                    passes_tried.append(
                        {
                            "pass": name,
                            "box_count": len(cand),
                            "mean_confidence": round(
                                float(np.mean([b.confidence for b in cand])), 4
                            )
                            if cand
                            else None,
                            "score": round(sc, 4),
                        }
                    )
                    if sc > best_score:
                        best_score = sc
                        best_boxes = cand
                        best_pass = name
                except Exception as pass_exc:  # noqa: BLE001
                    passes_tried.append(
                        {"pass": name, "status": "ERROR", "error": repr(pass_exc)}
                    )
            boxes = best_boxes
            if not boxes:
                # last-ditch sparse page mode
                gray = pre.get("gray")
                if gray is not None:
                    boxes = _run_tesseract(gray, psm=11, source_pass="sparse")
                    psm_used = [11]
                    best_pass = "sparse_psm11"
        except Exception as exc:  # noqa: BLE001 — surface as NOT_ASSESSED
            status = "NOT_ASSESSED"
            backend = "tesseract"
            error = f"OCR engine failed: {exc}"
            boxes = []

    full_text = reconstruct_full_text(boxes)
    mean_conf = (
        round(float(np.mean([b.confidence for b in boxes])), 4) if boxes else None
    )

    return {
        "status": status,
        "engine": backend,
        "mean_ocr_confidence": mean_conf,
        "text_box_count": len(boxes),
        "text_boxes": [b.to_dict() for b in boxes],
        "full_text": full_text,
        "preprocess": {
            "scale": pre["scale"],
            "skew_degrees": pre["skew_degrees"],
            "width": pre["width"],
            "height": pre["height"],
            "psm_used": psm_used,
            "multipass": multipass,
            "best_pass": best_pass,
            "passes_tried": passes_tried,
        },
        "error": error,
        "note": "OCR confidence is transcription confidence, NOT authenticity confidence.",
    }


def normalize_ocr_text(text: str) -> str:
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()
