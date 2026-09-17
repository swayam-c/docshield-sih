# SIH Phase 4 — MRZ processing

## Delivered

### Formats
- **TD3** (2×44) — passports
- **TD2** (2×36) — visas / some travel docs
- **TD1** (3×30) — many national ID cards
- Incomplete MRZ-like text → `PARTIAL` (never fabricated)

### Checklist (SIH UI)
| Item | Meaning |
|------|---------|
| Format valid | Structure length/layout OK |
| Check digits valid | ICAO 9303 check digits |
| Name consistency | MRZ ↔ visible OCR name |
| DOB consistency | MRZ ↔ visible DOB |
| Document number consistency | MRZ ↔ visible doc/visa number |

Statuses: `PASS` / `FAIL` / `NOT_ASSESSED`

### Profiles
- MRZ enabled for: `PASSPORT`, `VISA`, `NATIONAL_ID`
- Others remain `NOT_APPLICABLE`

### Privacy
- MRZ fields returned **masked**
- Internal `compare_values` from OCR are stripped before public API/evidence

## Honesty
- Check-digit FAIL ≠ automatic fraud
- Consistency FAIL = review signal only
- No MRZ invented when lines are missing

## Tests
- `tests/test_sih_phase4_mrz.py`
