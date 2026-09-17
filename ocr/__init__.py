"""DOCSHIELD OCR package — Phase 4."""

from ocr.engine import run_ocr, tesseract_available
from ocr.fields import extract_fields
from ocr.masking import mask_value
from ocr.mrz import analyze_mrz

__all__ = [
    "run_ocr",
    "tesseract_available",
    "extract_fields",
    "mask_value",
    "analyze_mrz",
]
