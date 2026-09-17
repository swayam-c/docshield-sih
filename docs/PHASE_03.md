# Phase 3 — Image quality + document classification

## Summary

Implemented a real OpenCV image-quality engine and a transparent heuristic document-type classifier, wired into `POST /api/analyze`.

## Working

- Resolution / blur / brightness / contrast / noise / rotation / perspective / crop / visibility
- `overall_quality` as **evidence quality** (explicitly not authenticity)
- `INCONCLUSIVE` when quality is insufficient
- Document types: PAN, AADHAAR, PASSPORT, DRIVING_LICENCE, OTHER
- Modular `document_profiles/`
- UI shows quality breakdown, type + confidence, reasons, pipeline progress

## Not claimed

- Calibrated authenticity estimate (still `null`)
- OCR / forensics / fusion
- Deep-learned document classifier (method: `heuristic_v1_geometry_color_layout`)
