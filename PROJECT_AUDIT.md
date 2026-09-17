# PROJECT_AUDIT.md

**SIH Problem Statement:** 26188 — AI-Based Fake Identity & Document Screening System  
**Organization:** Ministry of Home Affairs / Sashastra Seema Bal (SSB), Police II Division  
**Theme:** Blockchain & Cybersecurity  
**Audit date:** 2026-09-17  
**Workspace:** `D:\SIH Hackathon\sih`  
**Product name in repo:** DOCSHIELD AI  
**Current app version:** `0.4.0-phase5`  
**Audit scope:** Phase 0 only — inspect & document. **No implementation changes in this phase.**

---

## 1. Current architecture

```text
Browser (frontend/) ──► FastAPI (app/main.py)
                           │
           ┌───────────────┼───────────────┐
           ▼               ▼               ▼
      /api/analyze    /api/ocr|quality   /api/cases|health
           │          /classify|features
           ▼
    run_pipeline()  (app/pipeline/__init__.py)
           │
    ┌──────┼──────┬──────────┬────────────┐
    ▼      ▼      ▼          ▼            ▼
 Quality Type   OCR+Fields  MRZ     Vision features
 (OpenCV) (heur.) (Tesseract) (TD3)  EffNet/ViT/CLIP
           │
           ▼
    SQLite cases + evidence_json (app/db)
```

**Honest capability today:** secure upload → image quality → heuristic document type → OCR/fields/MRZ → pretrained **feature extraction** (not fraud classification) → case record.  
**Decision states partially used:** `PENDING`, `INCONCLUSIVE`. Full `PASS` / `REVIEW` / `HIGH-RISK` decision engine is **not** implemented.

---

## 2. Current files (application surface)

| Area | Paths |
|------|--------|
| Backend entry | `app/main.py`, `app/config.py` |
| API | `app/api/analyze.py`, `cases.py`, `health.py`, `schemas.py` |
| Core | `app/core/validation.py`, `cases.py`, `security.py` |
| DB | `app/db/models.py` (`CaseRecord`, `AuditEvent`) |
| Pipeline | `app/pipeline/quality.py`, `classifier.py`, `__init__.py` |
| ML features | `app/ml/*` (EfficientNet, ViT, CLIP extractors + registry) |
| OCR | `ocr/*` (preprocess, engine, fields, masking, dates, mrz) |
| Profiles | `document_profiles/{pan,aadhaar,passport,driving_licence,generic}.py` |
| Frontend | `frontend/index.html`, `assets/app.js`, `assets/styles.css` |
| Stubs only | `forensics/`, `evidence/`, `verification/`, `audit/`, `reports/` |
| Tests | `tests/test_*.py` (~34 collected) |
| Docs | `README.md`, `docs/PHASE_02..05.md`, this audit |
| Data | `data/docshield.db`, `data/uploads/`, `data/temp/` |
| Empty ML weight dirs | `models/{efficientnet,vit,clip,fusion,calibration}/` |

---

## 3. Existing models

| Model | Implementation | Role today | Fraud / authenticity head? |
|-------|----------------|------------|----------------------------|
| EfficientNet-B0 | `torchvision` ImageNet weights, classifier → Identity | Local embedding (1280-d) | **No** — `concern: NOT_SCORED` |
| ViT-B/16 | `torchvision` ImageNet weights, heads → Identity | Global embedding (768-d) | **No** |
| CLIP ViT-B/32 | `open_clip` (`openai` pretrained) | Supporting embedding (512-d) | **No** |
| Fusion classifier | — | Stub (`NOT_IMPLEMENTED`) | Missing |
| Calibration | — | Missing | Missing |
| Face embedding / liveness | — | Missing | Missing |
| Document-type CNN | — | Heuristic only (`heuristic_v1`) | N/A |

**Principle compliance:** Code correctly states pretrained backbones are **not** fraud detectors. Task-specific authenticity heads + labeled training are **missing** (required by SIH prompt Modules 6–10).

**Runtime notes:**
- Torch `2.14.0+cpu`, CUDA **false** on this machine (CPU fallback works).
- First-run weight download is large (~EfficientNet small + ViT ~330MB + CLIP); can timeout on slow networks; then caches locally.
- `DOCSHIELD_ML_MODE=mock` used in tests to avoid downloads.
- open_clip may emit QuickGELU config warning — compatibility cosmetic; verify before changing.

---

## 4. Existing dependencies

From `requirements.txt`:

- Web: `fastapi`, `uvicorn`, `python-multipart`, `pydantic`, `pydantic-settings`
- Data: `sqlalchemy`, `httpx`, `python-dotenv`
- Vision/OCR: `Pillow`, `numpy`, `opencv-python-headless`, `pytesseract`
- ML: `torch`, `torchvision`, `open-clip-torch`
- Test: `pytest`, `pytest-asyncio`

**System requirement:** Tesseract OCR binary on PATH (verified present on this host).

**Not present (needed later):** face libs (e.g. insightface / deepface / face_recognition), PDF stack, report PDF libs, blockchain SDK, rate-limit middleware, auth framework, calibration libs (scikit-learn), dataset tooling.

---

## 5. Existing UI

- Single-page professional shell: brand **DOCSHIELD AI**, upload drag/drop, preview, Analyze/Remove/Clear.
- Results: case ID, type, authenticity (null), confidence, evidence quality, decision, quality breakdown, reasons, masked fields, MRZ status, vision feature status.
- Phase badge: Phase 5 · Vision Features.
- **Missing vs SIH:** officer multi-page dashboard, live camera screen, heatmaps, gauges/charts, face verification panel, SIH demo gallery, report export UI, audit integrity UI.

---

## 6. Existing APIs

| Endpoint | Status |
|----------|--------|
| `GET /api/health` | Working — component readiness map |
| `GET /api` | Working |
| `POST /api/analyze` | Working — full current pipeline |
| `POST /api/quality` | Working |
| `POST /api/classify` | Working |
| `POST /api/ocr` | Working — masked fields; no raw `full_text` in response |
| `POST /api/features` | Working — EffNet/ViT/CLIP summaries |
| `GET /api/case/{id}`, `GET /api/cases` | Working |
| `GET /api/report/{id}` | Stub `NOT_IMPLEMENTED` |
| `POST /api/tamper` | Stub |
| `POST /api/face/compare` | Stub → `NOT_ASSESSED` |
| `POST /api/reverse-search` | Honest `SEARCH_UNAVAILABLE` |

**Missing vs SIH §41:** upload-only, face extract/verify/liveness, tampering analyze, cross-validation, screening finalize, system status board, camera endpoints.

---

## 7. Existing database

- **Engine:** SQLite (`DOCSHIELD_DB_URL`, default `data/docshield.db`)
- **Tables:** `cases`, `audit_events` (schema exists; hash-chain population not wired in pipeline)
- **Cases store:** type, decision, authenticity/confidence/evidence strings, filenames, `evidence_json`
- **Uploads:** opaque UUID filenames under `data/uploads/` (raw images currently retained — retention policy incomplete vs “do not permanently store unless required”)

---

## 8. Existing working functionality

| Capability | Evidence |
|------------|----------|
| Secure image upload validation | Extension/MIME/size/dims/corruption (`app/core/validation.py`) |
| Case IDs `DS-YYYY-NNNNNN` | Working |
| Image quality metrics + `overall_quality` ≠ authenticity | Working; can force `INCONCLUSIVE` |
| Heuristic document classification | PAN/AADHAAR/PASSPORT/DRIVING_LICENCE/OTHER |
| Modular document profiles | Field keys only; no invented gov specs |
| Tesseract OCR + preprocess | Working |
| Field extraction + masking | Working |
| Date checks (expired ≠ fake) | Working |
| Basic TD3 MRZ parse/check digits | Working; never fabricates |
| EffNet/ViT/CLIP feature extraction + caching | Working; `NOT_SCORED` |
| Honest unavailable stubs | Official verify / reverse search / face |
| pytest suite | ~34 tests collected |
| Graceful module failure language | Generally followed |

---

## 9. Existing broken / incomplete / partial functionality

| Item | Status | Notes |
|------|--------|-------|
| Document classifier as **trained** model | Done (SIH P2) | Synthetic logreg + heuristic fallback; not real gov dumps |
| Authenticity probabilities from EffNet/ViT | Done (P5/P6) | Uncalibrated synthetic heads; fusion P8 |
| Feature fusion + trainable head | **Done (SIH P8)** | Synthetic multimodal logistic fusion; uncalibrated |
| Probability calibration | Missing | Authenticity always `null` |
| Tampering / ELA / copy-move / heatmaps | Missing | Package stub |
| Photo/text/stamp manipulation modules | Missing | — |
| Face extract / 1:1 verify / age-robust | Missing | API stub |
| Live camera + liveness | Missing | — |
| Cross-validation engine | Missing | — |
| Final decision PASS/REVIEW/HIGH-RISK | Partial | Mostly PENDING/INCONCLUSIVE |
| Officer dashboard / analytics charts | Missing | Single analyze page |
| PDF input | Missing | PNG/JPG/JPEG/WEBP only |
| Report PDF/HTML | Stub | — |
| Audit hash-chain verification | Schema only | Not producing/verifying chain on analyze |
| Blockchain layer | Missing | Must stay DEMO/LOCAL/NOT CONNECTED until real |
| Dataset + leakage-safe splits | Missing | `dataset/` README only |
| Model evaluation metrics docs | Missing | No measured Accuracy/AUC to display |
| Rate limiting / full auth | Partial | Optional API key only |
| Upload retention / auto-delete | Incomplete | Files kept on disk |

**Not broken — intentionally honest:** no fabricated gov verification, reverse search, accuracy %, or blockchain txs.

---

## 10. Missing requirements (SIH 26188 mapping)

| SIH module | Gap |
|------------|-----|
| Secure input | PDF support; retention controls |
| Quality engine | Mostly present; UI % presentation can be refined |
| Document classifier | **Done (SIH P2)** — synthetic logreg + heuristic fallback; VISA/NATIONAL_ID/PERMIT/UNKNOWN |
| OCR intelligence | **Done (SIH P3)** — visa/permit/passport fields + completeness; MRZ compare still Phase 4 |
| MRZ | **Done (SIH P4)** — TD1/TD2/TD3 + checklist vs visible fields |
| DL authenticity heads | EffNet **Done (P5)** · ViT **Done (P6)** · CLIP supporting **Done (P7)** · fusion **Done (P8)** |
| CLIP integration | **Done (SIH P7)** — supporting type alignment; fusion still Phase 8 |
| Feature fusion | **Done (SIH P8)** — LR over EffNet/ViT/CLIP/OCR/quality/MRZ |
| Tampering + localization | Full forensics package + heatmaps |
| Human identity verification | Face pipeline + live camera + liveness + quality |
| Cross-validation | OCR↔MRZ↔face↔models |
| Calibration / confidence / evidence quality | Separate engines; calibrate with validation data |
| Decision engine | Four-state with explainability |
| Dashboard / camera UI | Multi-view professional UI |
| Dataset / leakage prevention / unseen attacks | Tooling + eval |
| Cybersecurity hardening | Rate limit, authz, retention |
| Audit / blockchain-ready hash chain | Implement + label blockchain status honestly |
| Reports / demo mode / SIH mapping docs | Phase 28–29 |

---

## 11. Duplicate functionality

- Prior “DOCSHIELD master prompt” phases (1–5) vs this SIH 26188 phase numbering differ; **preserve code**, remap phases in planning (do not dual-implement).
- `run_phase3_pipeline` alias duplicates `run_pipeline` (harmless compatibility).
- Health “phase” string vs README phase table may drift — keep single source of truth later.
- `AuditEvent` table vs future `audit/` package — avoid two audit writers.

**No large duplicated ML stacks found.**

---

## 12. Model compatibility issues

| Issue | Risk | Recommendation |
|-------|------|----------------|
| ImageNet EffNet/ViT ≠ document fraud | High if misused | Keep `NOT_SCORED` until trained heads exist |
| CLIP similarity ≠ genuineness | High | Use only as supporting features in fusion |
| open_clip QuickGELU warning | Low | Confirm outputs; avoid drive-by model swaps |
| CPU-only host | Medium latency | Keep caching; optional GPU path already coded |
| First-run downloads | Medium | Document; optional offline weight bundle for demo |
| Python 3.13 | Medium for some CV wheels | Pin tested versions; prefer venv |

---

## 13. Security issues

| Issue | Severity | Notes |
|-------|----------|-------|
| Uploaded identity images retained in `data/uploads/` | Medium | Add retention / purge / encrypt-at-rest policy |
| Optional API key often empty in demo | Medium | OK for local demo; harden for deployment |
| No rate limiting | Medium | Add for public endpoints |
| CORS `*` when debug | Low–Med | Tighten for non-dev |
| `.env` present locally | Info | Ensure gitignored (is) |
| PII in OCR path | Mitigated | Masking in API evidence; ensure logs stay clean |
| No session isolation for camera (N/A yet) | — | Required when camera lands |

---

## 14. Performance issues

- Sequential EffNet → ViT → CLIP on CPU is slow for live demo (seconds–tens of seconds cold).
- Large ViT weight download blocked earlier smoke tests until cached.
- No async job queue — analyze is synchronous request/response.
- No batching; acceptable for SIH prototype if progress UI added.
- OCR + three backbones every request — consider skip/lazy toggles for demo speed.

---

## 15. Recommended architecture changes

**Preserve (do not rewrite):**
- FastAPI app layout, upload validation, quality engine, OCR/masking/MRZ, ML registry/caching, case store, honest UNAVAILABLE stubs, frontend shell.

**Repair / extend:**
- Expand document taxonomy (VISA, NATIONAL_ID/PERMIT, UNKNOWN) via profiles.
- Replace heuristic classifier with trainable classifier when dataset ready; keep heuristic as fallback.
- Add task-specific authenticity heads on EffNet/ViT (+ fusion MLP/LR).
- Implement `forensics/` (ELA, noise, compression, copy-move, localization/heatmaps).
- Implement face verification package (1:1, quality, liveness) — not public face search.
- Wire evidence fusion + calibration + four-state decision.
- Implement audit hash-chain; label blockchain `DEMO/LOCAL/NOT CONNECTED` until real network.
- Grow UI: dashboard, camera, heatmaps, charts — reuse existing visual language.

**Replace only if broken:** nothing currently requires discard of working modules.

**Do not add:** fabricated accuracy, fake gov APIs, fake blockchain txs, uncontrolled identity web search.

---

## 16. Exact implementation plan (map to SIH phases)

Align **new SIH phase numbers** to **existing code**:

| SIH Phase | Action | Preserve / build |
|-----------|--------|------------------|
| **0** | This audit | Done — **STOP here** |
| 1 Secure input + quality | Extend PDF; retention | Preserve validation + quality |
| 2 Document classification | Train/fallback classifier; add VISA/ID/PERMIT | **Done** — synthetic logreg + SIH taxonomy |
| 3 OCR + fields | Enrich profiles | **Done** — profile-driven fields + completeness |
| 4 MRZ | Harden + UI checklist | **Done** — TD1/TD2/TD3 + consistency checklist |
| 5 EffNet classifier head | Train authenticity head | **Done** — synthetic logreg on EffNet embeddings |
| 6 ViT classifier head | Same | **Done** — synthetic logreg on ViT embeddings |
| 7 CLIP integration | Fusion features | **Done** — type alignment + fusion_ready (not authenticity) |
| 8 Feature fusion | LR/MLP + versions | **Done** — synthetic multimodal logistic fusion |
| 9–10 Tampering + localization | New `forensics/` | Empty package ready |
| 11–15 Face + camera + liveness | New `verification/face*` + UI | Stubs exist |
| 16 Cross-validation | New engine | — |
| 17–19 Calibration + confidence + decision | New | Partial scores today |
| 20–22 Dashboard, heatmaps, reports | Frontend + `reports/` | Shell exists |
| 23–24 Audit + blockchain-ready | Wire `audit/` hash chain | Schema ready |
| 25–27 Eval, unseen attacks, perf | `dataset/`, `MODEL_EVALUATION.md` | — |
| 28–30 Demo, docs, quality gate | `demo/`, README, checklist | Partial README |

**Rule for next coding session:** begin SIH Phase 1 only after this audit is accepted; implement one phase at a time; measure before displaying any % authenticity/accuracy.

---

## Component status matrix (audit table)

| Component | Current File(s) | Status | Working / Partial / Broken / Missing | Purpose | Problems | Recommended Upgrade | Priority |
|-----------|-----------------|--------|--------------------------------------|---------|----------|---------------------|----------|
| Secure upload | `app/core/validation.py` | Implemented | Working | Intake safety | No PDF; retention | PDF optional; purge policy | P0 |
| Image quality | `app/pipeline/quality.py` | Implemented | Working | Evidence suitability | — | Keep; expose % UI | P0 |
| Doc classifier | `app/pipeline/classifier.py` | SIH P2 | Working | Type label | Synthetic train only | Retrain on authorized data when available | P1 |
| OCR | `ocr/engine.py` | Tesseract | Working | Text | Quality-dependent | Keep | P0 |
| Fields + mask | `ocr/fields.py`, `masking.py` | Implemented | Working | Field IQ | Limited patterns | Profile-driven expand | P0 |
| MRZ | `ocr/mrz.py` | SIH P4 | Working | Travel docs | OCR-dependent | Keep honest NOT_* | P1 |
| EffNet features | `app/ml/efficientnet.py` + head | SIH P5 | Working | Local feats + uncalibrated head | Synthetic only | Retrain on authorized data | P1 |
| ViT features | `app/ml/vit.py` + head | SIH P6 | Working | Global feats + uncalibrated head | Synthetic only | Retrain on authorized data | P1 |
| CLIP features | `app/ml/clip_encoder.py` + evidence | SIH P7 | Working | Support feats | Misuse risk mitigated in API notes | Feed Phase 8 fusion | P1 |
| Fusion | — | Stub | Missing | Authenticity | — | LR/MLP + calib | P0 |
| Tampering | `forensics/` | Empty | Missing | Forensics | — | ELA+…+heatmap | P0 |
| Face verify | stub API | Stub | Missing | 1:1 identity | — | Embeddings+thresholds from eval | P0 |
| Live camera | — | — | Missing | Live verify | — | WebRTC/getUserMedia UI | P1 |
| Liveness | — | — | Missing | Anti-spoof | — | PAD model + INCONCLUSIVE | P1 |
| Cross-validation | — | — | Missing | Consistency | — | Rule+score engine | P0 |
| Calibration | — | — | Missing | Probabilities | Softmax misuse risk | Platt/Isotonic | P0 |
| Decision engine | pipeline rules | Partial | Partial | PASS/REVIEW/… | Mostly PENDING | Full four-state | P0 |
| Dashboard | `frontend/` | Analyze shell | Partial | Officer UX | No camera/heatmaps | Expand | P1 |
| Audit hash | DB schema | Partial | Partial | Tamper-evident | Not chained on write | Implement chain | P1 |
| Blockchain | — | — | Missing | Theme | Must not fake txs | DEMO/LOCAL label | P2 |
| Dataset/eval | `dataset/README` | Placeholder | Missing | Train/test | Leakage risk | Grouped splits | P0 |
| Reports | stub | Stub | Missing | Export | — | HTML/PDF masked | P1 |
| Demo mode | flag only | Partial | Partial | SIH demo | No DEMO-0x pack | Synthetic pack | P1 |

---

## Phase 0 completion

```text
PHASE COMPLETED: PHASE 0 — PROJECT AUDIT (SIH 26188)

Files Added/Updated:
  - PROJECT_AUDIT.md (this file; supersedes earlier empty-repo audit content)

Files Modified (code):
  - NONE (per Phase 0 stop rule)

Files Removed:
  - NONE

Models Used:
  - (audit only; runtime stack documented above)

Dependencies Added:
  - NONE

Tests Passed:
  - Not re-executed as part of Phase 0 stop gate (suite exists: ~34 tests)

Known Issues:
  - Authenticity/fusion/tampering/face/camera/calibration not built
  - Classifier heuristic; taxonomy incomplete vs Passport/Visa/ID/DL/Permit
  - Uploads retained on disk; no PDF
  - Blockchain not connected (correctly unclaimed)

Next Phase (when authorized):
  - SIH PHASE 1 — Secure input + image quality improvements
  - Do not start coding until this audit is accepted
```

---

**STOP.** No further project modifications until Phase 1 is explicitly authorized.
