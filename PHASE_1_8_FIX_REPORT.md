# PHASE 1–8 FIX REPORT

**Date:** 2026-09-17  
**Scope:** Bug fix & validation for SIH Phases 1–8 only  
**Phase 9+:** NOT TOUCHED

## Summary table

| Phase | Component | Bug | Root Cause | Fix | Test | Status |
| ----- | --------- | --- | ---------- | --- | ---- | ------ |
| 1 | Image quality | NaN/Inf risk to UI | OpenCV edge metrics | Sanitize floats in `QualityReport.to_dict` | `test_sih_phase1_*`, smoke low-quality | FIXED |
| 2 | Classifier aliases | Missing `document_type` keys | `to_dict` only used asdict | Added aliases in `ClassificationResult.to_dict` | `test_classifier_to_dict_aliases` | FIXED |
| 3 | OCR 0% + PASS | Conf fallback + PASS validation | Invented/zero conf still PASS | `_reconcile_validation` + EXTRACTED/UNCERTAIN | `test_zero_ocr_*`, honesty tests | FIXED |
| 3 | Completeness 0% / wrong missing | Counted non-OCR / wrong profile fields | Denominator included N/A fields | Applicable OCR-only completeness + inventory | `test_completeness_*` | FIXED |
| 3 | Passport fields missing | Surname/Given not matched | Generic Name: regex only | Restored `ocr/passport_fields.py` | passport honesty + regression | FIXED |
| 3 | Father name false positive | Loose `father` regex | Pattern matched Name | Tightened FATHER_NAME_RE; profile-gated | PAN regression | FIXED |
| 4 | MRZ NOT_DETECTED on Aadhaar | UI/module confusion | Wrong status when N/A | `supports_mrz=False` → NOT_APPLICABLE | `test_aadhaar_mrz_not_applicable` | FIXED |
| 4 | MRZ on OCR fail | Stuck NOT_ASSESSED | Soft status | MRZ-supporting + no OCR → NOT_DETECTED | pipeline | FIXED |
| 5 | EffNet UI contradiction | Backbone NOT_SCORED vs head scores | UI mixed roles | UI labels backbone vs head separately; keep UNCALIBRATED_SYNTHETIC | smoke + phase5 tests | FIXED |
| 6 | ViT UI contradiction | Same as EffNet | Same | Same pattern for ViT | phase6 + smoke | FIXED |
| 7 | CLIP misread as authenticity | Concern wording | Features ≠ authenticity | Keep NOT_AUTHENTICITY / supporting only | phase7 tests | VERIFIED |
| 8 | Fusion honesty | Risk of treating as authenticity_estimate | UI wording | Keep null authenticity_estimate; fusion concern preserved | phase8 + smoke | VERIFIED |
| * | Pipeline crash | Module exception abort | No isolation | try/except per OCR/fields/MRZ/vision + log | smoke | FIXED |
| * | Central result | Split UI calculations | No single object | `screening_result` + `phase_status` | smoke | FIXED |
| * | Frontend fields | Showed validation PASS | Used `validation` not status | Inventory + status; block 0%+PASS display | manual/UI | FIXED |

## MODELS LOADED

- Document classifier (sklearn/numpy logreg checkpoint)
- EfficientNet-B0 backbone (+ authenticity head npz)
- ViT-B/16 backbone (+ authenticity head npz)
- CLIP ViT-B/32 (open_clip or mock)
- Fusion logistic regression npz

## MODELS USED

All of the above in Phases 1–8 pipeline when image quality is sufficient.

## MODELS FEATURE-ONLY

- EfficientNet backbone
- ViT backbone
- CLIP image encoder (supporting type alignment only)

## MODELS SCORING

- EfficientNet authenticity head → SCORED · UNCALIBRATED_SYNTHETIC
- ViT authenticity head → SCORED · UNCALIBRATED_SYNTHETIC
- Fusion LR → FUSED · UNCALIBRATED_SYNTHETIC
- Document type classifier → type confidence only

## OCR STATUS

Working with Tesseract (or mock). Field statuses: EXTRACTED / UNCERTAIN / MISSING / NOT_APPLICABLE. Completeness = EXTRACTED / applicable OCR fields.

## MRZ STATUS

NOT_APPLICABLE when profile does not support MRZ; NOT_DETECTED / PARSED / PARTIAL when supported. Never fabricated.

## DOCUMENT CLASSIFICATION STATUS

Working with filename priors + visual features + logreg. Misclassification on ambiguous images remains a known limitation.

## FORENSICS STATUS

Phase 1–8 AI forensics = feature extraction + uncalibrated heads + CLIP alignment. Tampering localization = NOT_ASSESSED (Phase 9+).

## KNOWN LIMITATIONS

1. Authenticity heads and fusion are **uncalibrated synthetic** — not legal authenticity.
2. `authenticity_estimate` remains **null** until a later calibration phase.
3. MRZ depends on full-page OCR text (no dedicated MRZ crop pipeline in Phase 1–8).
4. Document classifier can mis-label difficult scans; fields then follow the wrong profile.
5. CLIP `position_ids` unexpected-key warnings are cosmetic for this loading path.
6. `np.random` appears only in **training/mock embedding** generators — not live screening scores.

## Regression

```text
DOCSHIELD_ML_MODE=mock pytest tests/ → PASS (incl. phase1–8 regression + smoke)
```
