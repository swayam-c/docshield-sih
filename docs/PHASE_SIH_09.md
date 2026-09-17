# SIH Phase 9 — Tampering heuristics

## Delivered

### Module
`forensics/tampering.py` — classical visual inconsistency cues:

| Signal | Role |
|--------|------|
| ELA (JPEG re-encode residual) | Compression / edit residual energy |
| Compression blockiness | 8×8 residual variance |
| Noise inconsistency | Tile Laplacian std variation |
| Edge inconsistency | Top/bottom edge density gap |
| Local anomaly | Residual energy outlier tiles |
| Copy-move indicator | Template self-similarity (cheap) |

### Outputs (`evidence.tampering`)
```text
status: ANALYZED | ERROR
severity: LOW_SIGNAL | MILD_REVIEW | ELEVATED_REVIEW
review_score: 0–1 (heuristic aggregate — NOT authenticity)
flags: [...]
signals: { ela_inconsistency, compression_..., ... }
concern: HEURISTIC_REVIEW_ONLY
calibrated: false
localization.status: NOT_ASSESSED  ← Phase 10
```

### Honesty
- Flags are **review signals only** — never proof of fraud/authenticity
- Does **not** set `authenticity_estimate` (still null)
- Localization / heatmaps explicitly deferred to **SIH Phase 10**

### API / UI / health
- `POST /api/tamper` — standalone heuristics
- Analyze pipeline always runs tampering (isolated try/except)
- Frontend **Tampering heuristics** panel
- Health: `tampering=working_heuristic_ela_noise_copymove_v1`
- Version `0.13.0-sih-phase9`

## Tests
- `tests/test_sih_phase9_tampering.py`
