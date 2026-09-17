# PHASE 1–8 AUDIT

**Date:** 2026-09-17  
**Scope:** SIH Phases 1–8 only (Phase 9+ untouched)  
**Method:** Code path trace from `POST /api/analyze` → pipeline → UI  
**Post-fix:** See `PHASE_1_8_FIX_REPORT.md`

## Execution path (traced)

```text
UPLOAD (validation/rate-limit/PDF)
 → IMAGE QUALITY (OpenCV)
 → DOCUMENT CLASSIFICATION (logreg+heuristic)
 → OCR (Tesseract preprocess)
 → FIELD EXTRACTION + VALIDATION (profiles)
 → MRZ (if supports_mrz)
 → EffNet features + authenticity head
 → ViT features + authenticity head
 → CLIP features + type alignment
 → FUSION (logistic multimodal)
 → assessment (authenticity_estimate = null)
 → screening_result (central object + phase_status)
```

---

## Per-phase status (after fixes)

| Phase | Title | Implemented? | Files | Models | Inputs | Outputs | Known errors | Dependencies | Status |
|-------|-------|--------------|-------|--------|--------|---------|--------------|--------------|--------|
| 1 | Secure input + quality | Yes | `app/core/validation.py`, `retention.py`, `rate_limit.py`, `pipeline/quality.py` | none | image/PDF | quality %, INCONCLUSIVE gate | NaN sanitized in `to_dict` | OpenCV, pypdfium2 | **COMPLETE** |
| 2 | Document classification | Yes | `pipeline/classifier.py`, `document_profiles/*` | `models/document_classifier/*.npz` | image+filename | label, aliases, confidence | Weak priors on hard images | numpy, OpenCV | **COMPLETE** (heuristic limits remain) |
| 3 | OCR + fields | Yes | `ocr/engine.py`, `fields.py`, `passport_fields.py`, `field_patterns.py` | Tesseract | image | masked fields + inventory | OCR quality depends on scan | pytesseract | **COMPLETE** |
| 4 | MRZ | Yes | `ocr/mrz.py` | none | OCR text | NOT_APPLICABLE / NOT_DETECTED / PARSED | No separate MRZ crop (full-page OCR) | — | **COMPLETE** |
| 5 | EffNet head | Yes | `ml/efficientnet.py`, `efficientnet_head.py` | `models/efficientnet/authenticity_head.npz` | embedding | UNCALIBRATED_SYNTHETIC scores | Not calibrated | torch | **COMPLETE** |
| 6 | ViT head | Yes | `ml/vit.py`, `vit_head.py` | `models/vit/authenticity_head.npz` | embedding | UNCALIBRATED_SYNTHETIC scores | Not calibrated | torch | **COMPLETE** |
| 7 | CLIP support | Yes | `ml/clip_encoder.py`, `clip_evidence.py` | open_clip / mock | image | type alignment NOT authenticity | position_ids warning cosmetic | open_clip | **COMPLETE** |
| 8 | Fusion | Yes | `evidence/fusion.py` | `models/fusion/fusion_logreg.npz` | multimodal vector | fused uncalibrated | authenticity_estimate null | numpy | **COMPLETE** |

---

## Critical bugs (found → fixed)

1. **0% OCR + PASS** — confidence never invented; PASS blocked at low OCR conf; statuses EXTRACTED/UNCERTAIN/MISSING.
2. **Completeness** — applicable OCR fields only; photo/signature/mrz → NOT_APPLICABLE.
3. **Father name regex** — tightened; profile-gated.
4. **MRZ** — NOT_APPLICABLE vs NOT_DETECTED correctly separated.
5. **Passport Surname/Given** — restored via `ocr/passport_fields.py`.
6. **EffNet/ViT UI** — backbone vs head statuses separated honestly.
7. **Quality NaN** — sanitized in `QualityReport.to_dict`.
8. **Module isolation** — OCR/fields/MRZ/vision failures isolated.
9. **Central object** — `screening_result` + `phase_status` (9_plus=NOT_TOUCHED).

## Models

| Model | Role | Scoring? |
|-------|------|----------|
| Doc classifier logreg | type | yes (type only) |
| EffNet-B0 ImageNet | features | no |
| EffNet authenticity head | auth/tamp | yes uncalibrated |
| ViT-B/16 ImageNet | features | no |
| ViT authenticity head | auth/tamp | yes uncalibrated |
| CLIP ViT-B/32 | support alignment | not authenticity |
| Fusion logreg | multimodal | yes uncalibrated |

## Phase 9+

**NOT TOUCHED** — tampering localization, face, blockchain, reverse search, watchlist remain stubs.
