"""SIH Phase 14 — cross-validation engine (review signals only).

Aggregates consistency across:
  OCR/fields · MRZ · face · CLIP type alignment · EffNet/ViT heads · tampering · quality

Does NOT produce authenticity_estimate. Does NOT fabricate official verification.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def _check(
    item: str,
    status: str,
    *,
    detail: str = "",
    severity: str = "info",
) -> Dict[str, Any]:
    return {
        "item": item,
        "status": status,  # PASS | FAIL | WARNING | NOT_ASSESSED | NOT_APPLICABLE
        "severity": severity,  # info | review | elevated
        "detail": detail,
    }


def run_cross_validation(
    *,
    quality: Optional[Dict[str, Any]] = None,
    document_type: Optional[Dict[str, Any]] = None,
    ocr: Optional[Dict[str, Any]] = None,
    fields: Optional[Dict[str, Any]] = None,
    mrz: Optional[Dict[str, Any]] = None,
    face: Optional[Dict[str, Any]] = None,
    vision: Optional[Dict[str, Any]] = None,
    fusion: Optional[Dict[str, Any]] = None,
    tampering: Optional[Dict[str, Any]] = None,
    text_visual: Optional[Dict[str, Any]] = None,
    profile_supports_mrz: bool = False,
    profile_expects_photo: bool = False,
) -> Dict[str, Any]:
    quality = quality or {}
    document_type = document_type or {}
    ocr = ocr or {}
    fields = fields or {}
    mrz = mrz or {}
    face = face or {}
    vision = vision or {}
    fusion = fusion or {}
    tampering = tampering or {}
    text_visual = text_visual or {}

    checks: List[Dict[str, Any]] = []

    # --- Image quality ---
    if quality.get("insufficient_for_analysis"):
        checks.append(
            _check(
                "image_quality",
                "FAIL",
                detail="Evidence quality insufficient for reliable screening.",
                severity="elevated",
            )
        )
    elif quality.get("overall_quality") is not None:
        oq = float(quality["overall_quality"])
        if oq < 0.45:
            checks.append(
                _check(
                    "image_quality",
                    "WARNING",
                    detail=f"Overall evidence quality={oq:.2f} is marginal.",
                    severity="review",
                )
            )
        else:
            checks.append(
                _check(
                    "image_quality",
                    "PASS",
                    detail=f"Overall evidence quality={oq:.2f}.",
                )
            )
    else:
        checks.append(_check("image_quality", "NOT_ASSESSED", detail="Quality missing."))

    # --- Document type confidence ---
    conf = document_type.get("confidence")
    label = document_type.get("label") or document_type.get("document_type_label") or "UNKNOWN"
    if conf is None:
        checks.append(_check("document_type", "NOT_ASSESSED", detail="No classifier confidence."))
    elif float(conf) < 0.35 or str(label).upper() in {"UNKNOWN", "OTHER"}:
        checks.append(
            _check(
                "document_type",
                "WARNING",
                detail=f"Type={label} confidence={conf} — uncertain routing.",
                severity="review",
            )
        )
    else:
        checks.append(
            _check(
                "document_type",
                "PASS",
                detail=f"Type={label} confidence={float(conf):.3f}.",
            )
        )

    # --- OCR / fields ---
    ocr_status = ocr.get("status")
    if ocr_status not in {"OK", "SKIPPED"}:
        checks.append(
            _check(
                "ocr",
                "FAIL" if ocr_status == "ERROR" else "WARNING",
                detail=f"OCR status={ocr_status}.",
                severity="review",
            )
        )
    else:
        checks.append(_check("ocr", "PASS", detail=f"OCR status={ocr_status}."))

    completeness = fields.get("field_completeness")
    if completeness is None:
        checks.append(_check("field_completeness", "NOT_ASSESSED", detail="No completeness."))
    elif float(completeness) < 0.35:
        checks.append(
            _check(
                "field_completeness",
                "WARNING",
                detail=f"Completeness={float(completeness):.1%} — many expected fields missing.",
                severity="review",
            )
        )
    else:
        checks.append(
            _check(
                "field_completeness",
                "PASS",
                detail=f"Completeness={float(completeness):.1%}.",
            )
        )

    for chk in fields.get("consistency_checks") or []:
        st = chk.get("status")
        if st in {"FAIL", "WARNING"}:
            checks.append(
                _check(
                    f"field_{chk.get('check')}",
                    st,
                    detail=chk.get("note") or "",
                    severity="review" if st == "WARNING" else "elevated",
                )
            )

    # --- MRZ ↔ visible ---
    mrz_status = mrz.get("status")
    if not profile_supports_mrz or mrz_status == "NOT_APPLICABLE":
        checks.append(
            _check(
                "mrz",
                "NOT_APPLICABLE",
                detail="MRZ not applicable for this document profile.",
            )
        )
    elif mrz_status == "NOT_DETECTED":
        checks.append(
            _check(
                "mrz",
                "WARNING",
                detail="MRZ expected but not detected.",
                severity="review",
            )
        )
    elif mrz_status in {"PARSED", "PARTIAL"}:
        checks.append(_check("mrz", "PASS" if mrz_status == "PARSED" else "WARNING", detail=f"MRZ {mrz_status}."))
        cons = mrz.get("consistency") or {}
        for key in ("name", "date_of_birth", "document_number"):
            row = cons.get(key) if isinstance(cons, dict) else None
            if not isinstance(row, dict):
                # checklist fallback
                continue
            st = row.get("status") or "NOT_ASSESSED"
            checks.append(
                _check(
                    f"mrz_visible_{key}",
                    st if st in {"PASS", "FAIL", "WARNING", "NOT_ASSESSED"} else "NOT_ASSESSED",
                    detail=row.get("detail") or row.get("note") or "",
                    severity="elevated" if st == "FAIL" else "review" if st == "WARNING" else "info",
                )
            )
        for row in mrz.get("checklist") or []:
            item = row.get("item") or ""
            if "consistency" not in item:
                continue
            st = row.get("status") or "NOT_ASSESSED"
            if st in {"PASS", "FAIL", "WARNING"}:
                checks.append(
                    _check(
                        f"mrz_checklist_{item}",
                        st,
                        detail=row.get("label") or "",
                        severity="elevated" if st == "FAIL" else "review" if st == "WARNING" else "info",
                    )
                )
    else:
        checks.append(_check("mrz", "NOT_ASSESSED", detail=f"MRZ status={mrz_status}."))

    # --- Face vs photo expectation ---
    face_status = face.get("status")
    if profile_expects_photo:
        if face_status == "DETECTED":
            checks.append(_check("face_photo", "PASS", detail="Face detected on document."))
        elif face_status == "MULTIPLE_DETECTED":
            checks.append(
                _check(
                    "face_photo",
                    "WARNING",
                    detail="Multiple faces detected.",
                    severity="review",
                )
            )
        elif face_status == "NOT_DETECTED":
            checks.append(
                _check(
                    "face_photo",
                    "WARNING",
                    detail="Profile expects a photo region but no face detected.",
                    severity="review",
                )
            )
        else:
            checks.append(_check("face_photo", "NOT_ASSESSED", detail=f"Face status={face_status}."))
    else:
        checks.append(
            _check(
                "face_photo",
                "NOT_APPLICABLE",
                detail="Photo face not required by this profile.",
            )
        )

    live = face.get("liveness") or {}
    if live.get("status") == "NOT_APPLICABLE":
        checks.append(_check("liveness", "NOT_APPLICABLE", detail="PAD N/A on document analyze."))
    elif live.get("decision") == "SUSPECT_PRESENTATION_ATTACK":
        checks.append(
            _check(
                "liveness",
                "WARNING",
                detail="Passive PAD flagged SUSPECT_PRESENTATION_ATTACK (uncalibrated).",
                severity="elevated",
            )
        )
    elif live.get("status") == "PASSIVE_ASSESSED":
        checks.append(
            _check(
                "liveness",
                "PASS" if live.get("decision") == "PASSIVE_OK_UNCALIBRATED" else "WARNING",
                detail=f"Passive PAD decision={live.get('decision')}.",
                severity="review" if live.get("decision") == "INCONCLUSIVE" else "info",
            )
        )
    else:
        checks.append(_check("liveness", "NOT_ASSESSED", detail=f"Liveness status={live.get('status')}."))

    # --- CLIP type alignment vs classifier ---
    clip = (vision.get("clip") or {}) if isinstance(vision, dict) else {}
    support = clip.get("supporting_evidence") or {}
    align = support.get("document_type_alignment") or {}
    agreement = align.get("agreement")
    if support.get("status") == "SUPPORTING_SCORED" and agreement:
        ag = str(agreement).upper()
        if ag == "ALIGNED":
            checks.append(
                _check(
                    "clip_type_alignment",
                    "PASS",
                    detail=f"CLIP top={align.get('top_label')} ALIGNED with classifier.",
                )
            )
        elif ag == "PARTIAL":
            checks.append(
                _check(
                    "clip_type_alignment",
                    "WARNING",
                    detail=f"CLIP top={align.get('top_label')} PARTIAL vs classifier={label}.",
                    severity="review",
                )
            )
        elif ag == "DIVERGENT":
            checks.append(
                _check(
                    "clip_type_alignment",
                    "WARNING",
                    detail=f"CLIP top={align.get('top_label')} DIVERGENT vs classifier={label}.",
                    severity="elevated",
                )
            )
        else:
            checks.append(
                _check(
                    "clip_type_alignment",
                    "NOT_ASSESSED",
                    detail=f"CLIP agreement={agreement}.",
                )
            )
    else:
        checks.append(_check("clip_type_alignment", "NOT_ASSESSED", detail="CLIP support not scored."))

    # --- EffNet / ViT head agreement ---
    enet_h = ((vision.get("efficientnet") or {}).get("authenticity_head") or {}) if isinstance(vision, dict) else {}
    vit_h = ((vision.get("vit") or {}).get("authenticity_head") or {}) if isinstance(vision, dict) else {}
    if enet_h.get("status") == "SCORED" and vit_h.get("status") == "SCORED":
        ea = float(enet_h.get("authentic_probability") or 0)
        va = float(vit_h.get("authentic_probability") or 0)
        delta = abs(ea - va)
        if delta <= 0.20:
            checks.append(
                _check(
                    "head_agreement",
                    "PASS",
                    detail=f"|EffNet−ViT| auth Δ={delta:.2f} (uncalibrated).",
                )
            )
        else:
            checks.append(
                _check(
                    "head_agreement",
                    "WARNING",
                    detail=f"Head disagreement Δ={delta:.2f} (EffNet={ea:.2f}, ViT={va:.2f}) — uncalibrated.",
                    severity="review",
                )
            )
    else:
        checks.append(_check("head_agreement", "NOT_ASSESSED", detail="One or both heads not scored."))

    # --- Tampering vs fusion ---
    tamp_sev = tampering.get("severity")
    if tampering.get("status") == "ANALYZED" and tamp_sev == "ELEVATED_REVIEW":
        checks.append(
            _check(
                "tampering_signals",
                "WARNING",
                detail=f"Tampering severity={tamp_sev} flags={tampering.get('flags') or []}.",
                severity="elevated",
            )
        )
    elif tampering.get("status") == "ANALYZED":
        checks.append(
            _check(
                "tampering_signals",
                "PASS",
                detail=f"Tampering severity={tamp_sev}.",
            )
        )
    else:
        checks.append(_check("tampering_signals", "NOT_ASSESSED", detail=f"status={tampering.get('status')}."))

    if fusion.get("status") == "FUSED":
        checks.append(
            _check(
                "fusion",
                "PASS",
                detail=f"Fusion {fusion.get('concern') or 'FUSED'} — not authenticity_estimate.",
            )
        )
    else:
        checks.append(_check("fusion", "NOT_ASSESSED", detail=f"Fusion status={fusion.get('status')}."))

    # OCR ↔ visual text region
    for row in (text_visual.get("ocr_visual_crosscheck") or [])[:8]:
        st = row.get("status") or "NOT_ASSESSED"
        mapped = {
            "CONSISTENT": "PASS",
            "INCONSISTENT": "FAIL",
            "UNCERTAIN": "WARNING",
            "NOT_FOUND": "WARNING",
        }.get(st, "NOT_ASSESSED")
        checks.append(
            _check(
                f"ocr_visual_{row.get('field')}",
                mapped,
                detail=f"OCR↔image region {st} · visual={row.get('visual_region')}",
                severity="review" if mapped == "WARNING" else "elevated" if mapped == "FAIL" else "info",
            )
        )

    typo = text_visual.get("typography") or {}
    if typo.get("anomalies"):
        checks.append(
            _check(
                "typography",
                "WARNING",
                detail=f"{len(typo['anomalies'])} relative text-size anomal(ies).",
                severity="review",
            )
        )

    # --- Aggregate ---
    counts = {"PASS": 0, "FAIL": 0, "WARNING": 0, "NOT_ASSESSED": 0, "NOT_APPLICABLE": 0}
    for c in checks:
        counts[c["status"]] = counts.get(c["status"], 0) + 1

    if counts["FAIL"] > 0:
        overall = "FAIL"
        severity = "elevated"
    elif counts["WARNING"] >= 3:
        overall = "WARNING"
        severity = "elevated"
    elif counts["WARNING"] > 0:
        overall = "WARNING"
        severity = "review"
    else:
        overall = "PASS"
        severity = "info"

    applicable = [c for c in checks if c["status"] not in {"NOT_APPLICABLE", "NOT_ASSESSED"}]
    pass_rate = (
        round(sum(1 for c in applicable if c["status"] == "PASS") / len(applicable), 4)
        if applicable
        else None
    )

    return {
        "status": "COMPLETED",
        "overall": overall,
        "severity": severity,
        "pass_rate": pass_rate,
        "counts": counts,
        "checks": checks,
        "calibrated": False,
        "concern": "CROSS_VALIDATION_REVIEW_ONLY",
        "note": (
            "Cross-validation aggregates module consistency for human review. "
            "It is not a legal authenticity verdict and does not set authenticity_estimate."
        ),
    }
