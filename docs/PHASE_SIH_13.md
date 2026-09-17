# SIH Phase 13 — Passive liveness / passive PAD

## Delivered

### Module
`verification/liveness.py` — uncalibrated passive presentation-attack cues:

| Signal | Role |
|--------|------|
| Texture flatness | Print / screen flatness proxy |
| Moiré-like FFT energy | Screen capture cue |
| Specular glare | Display reflection cue |
| Color cast | Unnatural balance |
| Frame motion | Static replay cue (multi-capture session) |

### Source rules
| Source | Liveness status |
|--------|-----------------|
| `document` (ID photo) | `NOT_APPLICABLE` |
| `live_camera` | `PASSIVE_ASSESSED` / `NOT_ASSESSED` |

### Decisions (never legal verdicts)
```text
PASSIVE_OK_UNCALIBRATED
INCONCLUSIVE
SUSPECT_PRESENTATION_ATTACK
NOT_APPLICABLE
```

### API / UI
- `POST /api/face/liveness?source=live_camera|document`
- Camera capture now runs passive PAD automatically
- UI shows liveness status / decision / spoof-lean score

### Honesty
- `concern: UNCALIBRATED_PASSIVE_PAD`
- `calibrated: false`
- Challenge-response (blink/turn) **not** implemented
- Does **not** set `authenticity_estimate`

### Health / version
- `face_liveness=working_passive_heuristic_v1`
- Version `0.17.0-sih-phase13`

## Tests
- `tests/test_sih_phase13_liveness.py`
