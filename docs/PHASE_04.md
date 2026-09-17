# Phase 4 — OCR + field extraction + validation

## Summary

Implemented Tesseract OCR with preprocessing, profile-aware field extraction, PII masking, date checks, and basic ICAO-style MRZ parse/check digits.

## Working

- Preprocess → OCR → text boxes + OCR confidence
- Fields: name, DOB, document number, issue/expiry, address (when detected)
- Masked values in API/UI (`XXXXXX1234` style)
- `full_text` omitted from stored/public evidence
- MRZ: `NOT_APPLICABLE` / `NOT_DETECTED` / `PARSED` (no fabrication)
- Expired date flagged without implying the document is fake
- `POST /api/ocr` endpoint

## Explicit non-claims

- OCR confidence ≠ authenticity
- Authenticity estimate still `null`
- MRZ check-digit failure ≠ automatic fraud
