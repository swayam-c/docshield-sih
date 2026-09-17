# SIH Phase 2 — Document classification

## Delivered

### Taxonomy & profiles
- SIH core classes: `PASSPORT`, `VISA`, `NATIONAL_ID`, `DRIVING_LICENCE`, `PERMIT`, `OTHER`, `UNKNOWN`
- Indian subtypes retained: `PAN`, `AADHAAR` (map to SIH category `NATIONAL_ID`)
- New profiles: `document_profiles/visa.py`, `national_id.py`, `permit.py`, `unknown.py`
- API returns `label`, `sih_category`, `confidence`, `alternatives`, `method`, `model_version`, `profile`

### Trainable classifier + fallback
- Feature vector: aspect, card/tall cues, color fractions, edge bands, photo-left bias, filename priors
- Multinomial logistic regression trained on **synthetic feature centroids only** (authorized synthetic proxy — not real government document dumps)
- Weights: `models/document_classifier/doc_type_logreg.npz` + metadata JSON
- Runtime blend: `0.55 × logreg + 0.45 × heuristic_v2`
- Low-confidence / low-margin gate → `UNKNOWN`
- Heuristic remains available if model load/train fails
- Method string reports honesty: `logreg_synthetic_v1+heuristic_v2` (optional `+unknown_gate`)

### UI
- Assessment panel shows SIH category and classifier method / model version

## Honesty constraints
- No claimed government-API verification
- No authenticity score from this phase
- Classifier confidence ≠ authenticity

## Tests
- Extended `tests/test_phase3_quality_classify.py`
- New `tests/test_sih_phase2_classify.py`
