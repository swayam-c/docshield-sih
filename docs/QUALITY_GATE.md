# QUALITY GATE — SIH 26188 / DOCSHIELD AI

Version: `0.30.0-sih-phase30`

## Functional checklist

- [x] Document upload works
- [x] Camera works (`/api/camera/*`)
- [x] OCR works (Tesseract)
- [x] Document classification works
- [x] MRZ works where applicable
- [x] EfficientNet / ViT / CLIP feature + heads work (mock or real)
- [x] Fusion works
- [x] Tampering + heatmaps work
- [x] Face extraction / uncalibrated 1:1 / passive liveness work
- [x] Cross-validation works
- [x] Calibration works (synthetic Platt)
- [x] Confidence visualization works
- [x] Model comparison / explainability works
- [x] PASS / REVIEW / HIGH-RISK / INCONCLUSIVE works
- [x] Secure upload / retention / rate limit
- [x] Audit trail + hash verify (local)
- [x] Reports work
- [x] Demo mode flag works
- [x] Officer dashboard (`/dashboard`)
- [x] CPU mock fallback (`DOCSHIELD_ML_MODE=mock`)

## Honesty checklist

- [x] No fabricated AI results
- [x] No fabricated accuracy claims
- [x] No fabricated government verification
- [x] No fake blockchain transactions (`NOT_CONNECTED`)
- [x] No unsupported official-document claims

## How to run

```bash
$env:DOCSHIELD_ML_MODE="mock"
py -3 -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

- UI: http://127.0.0.1:8000  
- Dashboard: http://127.0.0.1:8000/dashboard  
- API docs: http://127.0.0.1:8000/docs  
