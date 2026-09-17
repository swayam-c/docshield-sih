# MODEL_EVALUATION.md

## Status
No real government-document labeled dataset is shipped in this repository.

## What exists
- Synthetic proxy training for EffNet/ViT authenticity heads, fusion logistic, and Platt scaler
- Unit/integration pytest suite (`DOCSHIELD_ML_MODE=mock`)
- Heuristic tampering / PAD / OCR modules (not benchmarked for accuracy %)

## What must NOT be claimed
- Production accuracy percentages
- ROC/AUC on official ID corpora (not measured here)
- Unseen-attack robustness guarantees

## Unseen-attack / future eval (Phase 26 scaffold)
Place authorized evaluation sets under `dataset/` and record honest metrics here when available.
Until then, report metrics as **NOT MEASURED**.

## Performance notes (Phase 27)
- Prefer `DOCSHIELD_ML_MODE=mock` for CI / fast demo
- Real weights: first request may download torch/open-clip checkpoints
- Analyze is synchronous; suitable for SIH prototype, not high-QPS production
