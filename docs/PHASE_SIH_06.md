# SIH Phase 6 — ViT authenticity head

## Delivered

### Model
- Frozen ViT-B/16 ImageNet embeddings (768-d) preserved
- Task-specific logistic head: **authentic** vs **tampered**
- Weights: `models/vit/authenticity_head.npz`
- Trained on **synthetic embedding proxy** only

### Outputs
```text
authentic_probability
tampered_probability
predicted_label
model_version
calibrated: false
concern: UNCALIBRATED_SYNTHETIC
```

### Pipeline / UI
- Scores under `evidence.ai_forensics.vit.authenticity_head`
- `authenticity_estimate` remains **null**
- Frontend shows ViT head auth/tamp %
- Health: `vit_authenticity_head=working_synthetic_logreg_v1`

## Honesty
- Uncalibrated · synthetic only · not legal authenticity
- CLIP role = SIH Phase 7; fusion later

## Tests
- `tests/test_sih_phase6_vit_head.py`
