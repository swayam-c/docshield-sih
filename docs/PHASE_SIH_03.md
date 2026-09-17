# SIH Phase 3 — OCR + field extraction

## Delivered

### OCR engine
- Line-aware `full_text` reconstruction (newlines between Tesseract lines)
- Multi-PSM fallback when the first pass yields few tokens
- Preprocess metadata includes `psm_used`

### Profile-driven field intelligence
- Enriched extractors for PASSPORT, VISA, NATIONAL_ID, PERMIT, PAN, AADHAAR, DL
- New fields where cues exist: nationality, sex/gender, father_name, visa_number,
  visa_type, entries, duration_of_stay, permit_type
- Every field payload includes:
  - `field`, masked `value`, `ocr_confidence`, `bounding_box`, `source_region`, `validation`
- `field_completeness` vs profile expected fields (photo/signature/MRZ excluded from OCR %)
- Date-ordering consistency checks (issue≥DOB, expiry≥issue) — **not authenticity**
- Validation summary (pass / warning / fail counts)

### UI
- Completeness %, missing expected fields, source-region hint, OCR confidence

## Honesty
- OCR confidence ≠ authenticity
- Missing fields ≠ fraud
- Expired date ≠ fake (existing date engine preserved)
- No fabricated MRZ (MRZ hardening is SIH Phase 4)

## Tests
- `tests/test_sih_phase3_ocr.py` (+ existing Phase 4 OCR suite)
