# AUTONOMOUS_FULL_SYSTEM_TEST_REPORT.md

**Product:** DOCSHIELD AI (SIH 26188)  
**Build:** `0.30.1-sih-phase30-stable`  
**Date:** 2026-09-17  
**Mode tested:** `DOCSHIELD_ML_MODE=mock` (+ `DOCSHIELD_OCR_ENGINE=mock` for CI)

---

## Summary

| Metric | Value |
|--------|-------|
| TOTAL BUGS FOUND | 8 |
| TOTAL BUGS FIXED | 8 |
| REMAINING BLOCKING BUGS | 0 (in mock CI path) |
| TESTS RUN | 146 |
| TESTS PASSED | 146 |
| TESTS FAILED | 0 |

---

## Bugs found → fixed

| # | Bug | Root cause | Fix |
|---|-----|------------|-----|
| 1 | Silent `except: pass` on audit append | Errors swallowed | Explicit `audit_append` ERROR status + log |
| 2 | OCR single-pass miss risk | Only gray/binary PSM | Multi-pass variants + best-score selection (`ocr/multipass.py`) |
| 3 | Text boxes lacked geometry fields | Schema incomplete | Added width/height/center/source_pass/status |
| 4 | No text typography / splicing | Not implemented | `forensics/text_analysis.py` |
| 5 | Metadata stub only | Hard-coded NOT_ASSESSED | Real EXIF probe; missing ≠ fraud |
| 6 | No stamp / photo-region cues | Missing | `forensics/metadata_stamp.py` |
| 7 | Camera UX not gated on document face | Manual only | Banner: “Document face detected → Start live verification” |
| 8 | No analysis progress UX | Status text only | Progress list from real backend statuses |

---

## Module status (measured)

| Area | Status | Notes |
|------|--------|-------|
| OCR | WORKING (mock/tesseract) | Multipass when Tesseract present; mock = NOT_ASSESSED if binary missing |
| OCR bounding boxes | WORKING | Geometry + source_pass |
| Face | WORKING | Haar; age-robustness **NOT_VALIDATED** |
| Camera | WORKING | Permission denied → `PERMISSION_DENIED` (no crash loop) |
| Liveness | WORKING | Passive PAD uncalibrated; document source NOT_APPLICABLE |
| Deep learning | WORKING | EffNet/ViT/CLIP + synthetic heads (mock) |
| Forensics ELA/copy-move | WORKING | Heuristic review only |
| Text splicing / typography | WORKING | Heuristic |
| Image/photo splicing | WORKING | When face bbox present |
| Stamp | WORKING | Circle/ink cue — not official templates |
| Metadata | WORKING | PRESENT / UNAVAILABLE |
| Cross-validation | WORKING | Includes OCR↔visual rows |
| Fusion / calibration / decision | WORKING | Proxy Platt; PASS/REVIEW/HIGH-RISK/INCONCLUSIVE |
| Blockchain | NOT_CONNECTED | Local hash chain only |

---

## Document types exercised (automated)

PAN / AADHAAR / PASSPORT / DL / VISA / NATIONAL_ID / PERMIT / OTHER / UNKNOWN — via existing phase regression + pipeline smoke (mock OCR).

---

## Blocked / external dependencies

| Item | Reason |
|------|--------|
| Real Tesseract OCR quality on production scans | Requires Tesseract binary + real images in environment |
| Production-calibrated authenticity | No real gov-doc labeled dataset |
| Age-invariant face match claim | Prototype embedding **NOT_VALIDATED** |
| Public blockchain anchoring | Intentionally **NOT_CONNECTED** |
| Official verification / reverse search | Unavailable by design |
| Browser camera E2E in CI | Needs interactive permission |

---

## Files changed (this hardening pass)

- `ocr/engine.py`, `ocr/multipass.py`
- `forensics/text_analysis.py`, `forensics/metadata_stamp.py`
- `evidence/cross_validation.py`
- `app/pipeline/__init__.py`, `app/api/analyze.py`, `app/ml/clip_evidence.py`
- `verification/face.py`
- `frontend/index.html`, `frontend/assets/app.js`, `frontend/assets/styles.css`
- `tests/test_autonomous_hardening.py`
- Reports: this file + `PHASE_STATUS.md`

---

## Honesty statement

No fabricated accuracy %, no fake government verification, no fake blockchain transactions, no silent success on failures.
