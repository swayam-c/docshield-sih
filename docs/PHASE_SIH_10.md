# SIH Phase 10 — Tampering localization / heatmaps

## Delivered

### Module
`forensics/localization.py` — spatial anomaly map from:

| Component | Weight |
|-----------|--------|
| ELA residual | 0.45 |
| High-frequency residual | 0.30 |
| Laplacian noise magnitude | 0.25 |

Outputs a 24×24 review grid, hotspot boxes, and a downscaled PNG overlay (base64).

### Outputs (`evidence.tampering.localization`)
```text
status: LOCALIZED | ERROR | NOT_ASSESSED
grid: 24×24 float scores [0,1]
hotspots: [{x,y,w,h,score}, ...]
peak_score / mean_score / high_anomaly_coverage
overlay_png_base64 + overlay_mime
concern: HEURISTIC_LOCALIZATION_ONLY
calibrated: false
```

### Honesty
- Heatmap ≠ proof of tampering
- Does **not** set `authenticity_estimate`
- Scalar Phase-9 signals remain available even if localization fails

### API / UI / health
- Analyze + `/api/tamper` include localization block
- Frontend shows hotspot stats + overlay image
- Health: `tampering_localization=working_ela_hf_noise_heatmap_v1`
- Version `0.14.0-sih-phase10`

## Tests
- `tests/test_sih_phase10_localization.py`
