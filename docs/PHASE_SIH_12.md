# SIH Phase 12 — Live camera capture

## Delivered

### Module
`verification/camera.py` — ephemeral in-memory camera sessions:

| Capability | Status |
|------------|--------|
| Start / end session | Working |
| Attach document reference | Working |
| Capture frame → face detect | Working |
| Optional uncalibrated selfie↔document compare | Working |
| Persist frames to disk | No (ephemeral) |
| Liveness / PAD | NOT_ASSESSED (Phase 13) |

### API
```text
POST /api/camera/session/start
POST /api/camera/session/end?session_id=
POST /api/camera/session/attach-document?session_id=
POST /api/camera/capture?session_id=&compare_to_document=
GET  /api/camera/status
```

### UI
Live camera panel: Start → getUserMedia → Capture & analyze → Stop.

### Honesty
- Capture ≠ identity proof
- Liveness remains **NOT_ASSESSED**
- Document-only analyze marks `face.camera = NOT_USED`
- Does **not** set `authenticity_estimate`

### Health / version
- `live_camera=working_ephemeral_session_v1`
- Version `0.16.0-sih-phase12`

## Tests
- `tests/test_sih_phase12_camera.py`
