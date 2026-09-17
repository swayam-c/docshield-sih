# SIH Phase 11 — Face detection / extraction

## Delivered

### Module
`verification/face.py` — OpenCV Haar frontal-face detection:

| Capability | Status |
|------------|--------|
| Detect face(s) on document image | Working |
| Face quality (size/sharpness/brightness/contrast) | Working |
| Lightweight fingerprint embedding | Working (uncalibrated) |
| Optional 1:1 compare vs reference upload | Working · `COMPARED_UNCALIBRATED` |
| Liveness / PAD | NOT_ASSESSED (later) |
| Live camera | NOT_ASSESSED (later) |

### Outputs (`evidence.face`)
```text
status: DETECTED | MULTIPLE_DETECTED | NOT_DETECTED | ERROR
face_count, faces[{bbox, quality, embedding_fingerprint}]
verification.status: NOT_ASSESSED | COMPARED_UNCALIBRATED
liveness.status: NOT_ASSESSED
camera.status: NOT_ASSESSED
concern: DETECTION_ONLY | UNCALIBRATED_SIMILARITY
```

### Honesty
- Detection ≠ identity match
- Similarity decisions are always `INCONCLUSIVE*` — never legal PASS/FAIL match
- Does **not** set `authenticity_estimate`
- No public face search

### API / UI / health
- `POST /api/face/detect`
- `POST /api/face/compare` (document `file` + optional `reference`)
- Frontend **Face detection** panel
- Health: `face_detection=working_opencv_haar_v1`
- Version `0.15.0-sih-phase11`

## Tests
- `tests/test_sih_phase11_face.py`
