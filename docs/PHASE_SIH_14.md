# SIH Phase 14 — Cross-validation engine

## Delivered

### Module
`evidence/cross_validation.py` — rule-based consistency aggregator across:

| Signal | Role |
|--------|------|
| Image quality | FAIL if insufficient; WARNING if marginal |
| Document type | WARNING on low confidence / UNKNOWN |
| OCR + field completeness | Status + missing-field review |
| MRZ ↔ visible | N/A when profile does not support MRZ |
| Face photo expectation | PASS / WARNING vs profile photo field |
| Passive liveness | N/A on document; WARNING on suspect PAD |
| CLIP type alignment | ALIGNED / PARTIAL / DIVERGENT |
| EffNet ↔ ViT heads | WARNING when |Δ| auth > 0.20 (uncalibrated) |
| Tampering + fusion | Review severity / FUSED presence |

### Output shape
```text
status: COMPLETED
overall: PASS | WARNING | FAIL
severity: info | review | elevated
pass_rate: float | null
checks: [{item, status, severity, detail}, ...]
concern: CROSS_VALIDATION_REVIEW_ONLY
calibrated: false
```

### Pipeline / UI
- Wired after fusion in `app/pipeline`
- `evidence.cross_validation` + `evidence.consistency.cross_validation`
- `phase_status.14_cross_validation`
- Frontend Cross-validation panel (review counts + check list)

### Honesty
- Does **not** set `authenticity_estimate`
- Does **not** claim legal authenticity, gov APIs, or reverse search
- Aggregate is for human review only

### Health / version
- `cross_validation=working_rule_engine_v1`
- Version `0.18.0-sih-phase14`

## Tests
- `tests/test_sih_phase14_cross_validation.py`
