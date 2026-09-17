# SIH Phase 8 — Feature fusion

## Delivered

### Architecture
Transparent **logistic regression** over a fixed multimodal vector:

- EfficientNet fingerprint + auth/tamp head probs
- ViT fingerprint + auth/tamp head probs
- CLIP fingerprint + prototype alignment probs
- Evidence quality, OCR confidence, field completeness, doc-type confidence
- MRZ checklist pass rate (when parsed)

Weights: `models/fusion/fusion_logreg.npz`  
Version: `sih-phase8-synthetic-v1` (synthetic multimodal proxy only)

### Outputs (`evidence.fusion`)
```text
status: FUSED
authentic_probability / tampered_probability
predicted_label
head_agreement (vs EffNet/ViT heads)
calibrated: false
concern: UNCALIBRATED_SYNTHETIC
```

### Honesty
- Fusion ≠ calibrated `authenticity_estimate` (still **null**)
- Not trained on real government documents
- Not a legal authenticity verdict

### UI / health
- Frontend shows fusion auth/tamp %
- `fusion=working_synthetic_logreg_v1`
- Version `0.12.0-sih-phase8`

## Tests
- `tests/test_sih_phase8_fusion.py`
