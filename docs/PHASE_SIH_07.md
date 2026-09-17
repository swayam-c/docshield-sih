# SIH Phase 7 — CLIP feature integration

## Delivered

### Role
CLIP ViT-B/32 is integrated as **supporting evidence only**:
- Image embedding (512-d) preserved
- Text–image **document-type alignment** vs SIH prototypes (passport/visa/ID/DL/permit/…)
- Agreement vs classifier label: `ALIGNED` / `PARTIAL` / `DIVERGENT` / `NOT_ASSESSED`
- `fusion_ready` prototype probability vector for Phase 8

### Explicit non-claims
- CLIP similarity **≠** genuineness / authenticity
- `concern: NOT_AUTHENTICITY`
- `authenticity_estimate` remains **null**

### Mock vs real
- `DOCSHIELD_ML_MODE=mock`: deterministic prototype cosines
- Real open_clip: encode prototype prompts + image cosine

### UI / health
- Frontend shows CLIP top label / probability / agreement
- `clip_supporting_evidence=working_type_alignment_v1`
- Version `0.11.0-sih-phase7`

## Tests
- `tests/test_sih_phase7_clip.py`
