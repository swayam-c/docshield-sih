# SIH Phase 1 — Secure input + image quality

## Delivered

### Secure input
- Magic-byte / file-signature checks (PNG/JPEG/WEBP/PDF)
- Rejection of executable/script polyglots (`MZ`, HTML, PHP, ELF)
- EXIF orientation correction
- Optional PDF first-page render via `pypdfium2`
- Rate limiting (`DOCSHIELD_RATE_LIMIT_PER_MINUTE`)
- Retention policy: default **delete after analyze** (`DOCSHIELD_RETAIN_UPLOADS=false`)
- TTL purge when retention enabled
- APIs: `POST /api/document/upload`, `POST /api/document/analyze`, `GET /api/system/status`

### Image quality
- Measured scores exposed as percentages:
  - Resolution, Sharpness, Lighting, Perspective, Noise, Compression, Overall
- Compression / blockiness estimate added
- `overall_quality` remains **evidence suitability**, not authenticity
- Insufficient quality → `INCONCLUSIVE` with clear reason

## Config

See `.env.example` for `ALLOW_PDF`, `RETAIN_UPLOADS`, `UPLOAD_TTL_SECONDS`, `RATE_LIMIT_PER_MINUTE`.
