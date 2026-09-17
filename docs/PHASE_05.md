# Phase 5 — EfficientNet + ViT + CLIP feature extraction

## Summary

Added lazy-cached pretrained backbone feature extractors:

- **EfficientNet-B0** (torchvision, local features)
- **ViT-B/16** (torchvision, global features)
- **CLIP ViT-B/32** (open_clip, supporting features)

## Important honesty

Loading pretrained ImageNet/CLIP weights does **not** create fraud detectors.
Each module returns `concern: NOT_SCORED` until a dedicated fraud head + fusion exist (Phase 6+).

## Runtime

- GPU used when available; CPU fallback otherwise
- Models loaded once per process (`app/ml/registry.py`)
- `DOCSHIELD_ML_MODE=mock` for tests (no weight download)
- Compact embedding fingerprints stored in evidence (not full vectors)

## API

- `POST /api/features`
- `/api/analyze` includes `evidence.ai_forensics`
