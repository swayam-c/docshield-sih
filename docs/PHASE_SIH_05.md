# SIH Phase 5 — EfficientNet authenticity head

## Delivered

### Model
- Frozen EfficientNet-B0 ImageNet embeddings (1280-d) preserved
- Task-specific logistic head: **authentic** vs **tampered**
- Weights: `models/efficientnet/authenticity_head.npz`
- Trained on **synthetic embedding proxy** only (authorized synthetic data — not real gov docs)

### Outputs (per image)
```text
authentic_probability
tampered_probability
predicted_label
model_version
calibrated: false
concern: UNCALIBRATED_SYNTHETIC
```

### Pipeline / UI
- Head scores attached under `evidence.ai_forensics.efficientnet.authenticity_head`
- Assessment `authenticity_estimate` remains **null** (awaits fusion + calibration)
- Frontend shows EffNet head auth/tamp %
- Health: `efficientnet_authenticity_head=working_synthetic_logreg_v1`

## Honesty
- Not a legal authenticity verdict
- Not calibrated
- Not trained on real government identity documents
- ViT head = SIH Phase 6; fusion later

## Tests
- `tests/test_sih_phase5_efficientnet_head.py`
