# SIH Phase 15 — Probability calibration

## Delivered

### Module
`evidence/calibration.py` — **Platt scaling** on fusion `authentic_probability`:

| Item | Detail |
|------|--------|
| Method | 1-D logistic on `logit(raw_fusion_auth)` |
| Training | Synthetic multimodal proxy scores (same family as fusion) |
| Weights | `models/calibration/platt_scaler.npz` |
| Version | `sih-phase15-synthetic-platt-v1` |

### Outputs (`evidence.calibration`)
```text
status: CALIBRATED | SKIPPED | NOT_ASSESSED | ERROR
raw_authentic_probability
calibrated_authentic_probability
authenticity_estimate   ← same as calibrated when CALIBRATED
proxy_calibrated: true
production_calibrated: false
concern: SYNTHETIC_PROXY_CALIBRATED
```

### Pipeline rules
- Sets `assessment.authenticity_estimate` when fusion is `FUSED` and case is not `INCONCLUSIVE`
- Leaves estimate **null** when evidence quality is insufficient / INCONCLUSIVE
- Does **not** implement PASS / REVIEW / HIGH-RISK (decision engine = later phase)

### Honesty
- Proxy calibration ≠ production calibration on real government documents
- Not a legal authenticity verdict
- UI labels estimate as **proxy**

### Health / version
- `calibration=working_synthetic_platt_v1`
- Version `0.19.0-sih-phase15`

## Tests
- `tests/test_sih_phase15_calibration.py`
