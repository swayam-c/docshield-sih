# DOCSHIELD AI — Demo pack

## Demo mode
`DOCSHIELD_DEMO_MODE=true` (default) adds demo banners.  
Screening outputs remain **proxy / heuristic** — never presented as official verification.

## Suggested demo flow
1. Open http://127.0.0.1:8000
2. Upload a clear document image (or synthetic sample)
3. Review quality, OCR, MRZ, vision, tampering heatmap, face, cross-val, calibration
4. Note decision: PASS / REVIEW / HIGH-RISK / INCONCLUSIVE
5. Open `/dashboard` for case queue + report + audit chain verify

## Labels required in every demo
- DEMO DATA / SYNTHETIC PROXY SIGNALS
- Blockchain: **NOT CONNECTED**
- Not a legal authenticity certificate

## Sample assets
Add clearly labeled synthetic images under `demo/samples/` if desired.
Do not include real PII identity documents in the public repo.
