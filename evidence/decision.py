"""SIH Phase 17 — final screening + officer verdict fusion.

Backend screening labels: PASS | REVIEW | HIGH-RISK | INCONCLUSIVE
Officer-facing labels (assessment.officer_verdict): PASS — AUTHENTIC |
TAMPERING INDICATED | SUSPICIOUS / FAKE INDICATION | INCONCLUSIVE — REVIEW REQUIRED

Fuses DOCUMENT/IMAGE evidence with HUMAN IDENTITY evidence.
Calibration status is tracked separately from match/no-match results.
Screening assist only — not legal authenticity.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


def _identity_from_face(face: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Map face/camera verification → identity evidence status.

    DETECTED ≠ MATCH. Only verification.result == MATCH may yield identity MATCH.
    Liveness/PAD/face_count must never produce MATCH.
    """
    face = face or {}
    ver = face.get("verification") or {}
    live = face.get("liveness") or {}
    checks: List[Dict[str, str]] = []
    conflicts: List[str] = []
    cam = face.get("camera") or {}
    ref = face.get("reference") or {}
    doc_meta = face.get("document_face") or {}
    live_meta = face.get("live_face") or {}

    # Prefer explicit document_face metadata (survives live-frame merge)
    doc_face = (
        doc_meta.get("status")
        or face.get("document_face_status")
        or ("NOT_ASSESSED" if cam.get("status") == "CAPTURED" else None)
        or face.get("status")
        or "NOT_ASSESSED"
    )
    live_status = (
        live_meta.get("status")
        or ref.get("status")
        or (face.get("status") if cam.get("status") == "CAPTURED" else None)
        or "NOT_ASSESSED"
    )

    if doc_face == "NOT_DETECTED" and cam.get("status") != "CAPTURED":
        checks.append({"item": "Document face", "status": "FAIL", "detail": "NOT_DETECTED"})
        checks.append({"item": "Face comparison", "status": "NA", "detail": "NOT_AVAILABLE — no document face"})
        checks.append({"item": "Live face", "status": "NA", "detail": "NOT_ASSESSED"})
        return {
            "status": "FACE_NOT_DETECTED",
            "calibration": "NOT_ASSESSED",
            "result": "NOT_AVAILABLE",
            "similarity": None,
            "technical_status": "NOT_ASSESSED",
            "technical_decision": None,
            "checks": checks,
            "conflicts": ["Document face was not detected — identity cannot be verified."],
            "document_face": doc_meta or {"status": "NOT_DETECTED"},
            "live_face": live_meta or None,
        }

    if doc_face in {"DETECTED", "MULTIPLE_DETECTED"}:
        checks.append(
            {
                "item": "Document face",
                "status": "OK" if doc_face == "DETECTED" else "WARN",
                "detail": doc_face,
            }
        )
    else:
        checks.append({"item": "Document face", "status": "NA", "detail": str(doc_face)})

    live_count = live_meta.get("face_count")
    if live_count is None:
        live_count = ref.get("face_count")
    if live_count is None and cam.get("status") == "CAPTURED":
        live_count = face.get("face_count")

    if live_count is not None or cam.get("status") == "CAPTURED":
        checks.append(
            {
                "item": "Live face",
                "status": "OK" if live_count == 1 and live_status == "DETECTED" else "WARN",
                "detail": f"status={live_status} count={live_count}",
            }
        )
    else:
        checks.append({"item": "Live face", "status": "NA", "detail": "NOT_ASSESSED"})

    # ONLY explicit verification.result == MATCH — never DETECTED / liveness / face_count / match_label alone
    result = (ver.get("result") or "").upper() or None
    if result == "MATCH_CANDIDATE":
        result = "MATCH"
    if result == "NO_MATCH_CANDIDATE":
        result = "NO_MATCH"
    if result not in {"MATCH", "NO_MATCH", "INCONCLUSIVE", "NOT_AVAILABLE", "NOT_ASSESSED"}:
        # Do not promote HIGH_SIMILARITY or match_label without explicit result
        if not ver or ver.get("status") in {"NOT_ASSESSED", None}:
            result = "NOT_AVAILABLE"
        else:
            result = "INCONCLUSIVE"

    # Hard rule: detection / camera capture without a compare result is never MATCH
    if result == "MATCH" and ver.get("status") not in {"COMPARED_UNCALIBRATED", "COMPARED"}:
        result = "INCONCLUSIVE"
        conflicts.append("MATCH ignored — comparison status missing (detection is not verification).")

    if result == "MATCH" and ver.get("similarity") is None:
        result = "INCONCLUSIVE"
        conflicts.append("MATCH ignored — no similarity score from doc↔live compare.")

    if live_count is not None and int(live_count) > 1:
        result = "INCONCLUSIVE"
        conflicts.append("Multiple live faces detected — 1:1 verification not performed.")

    if cam.get("status") == "CAPTURED" and (
        live_count == 0
        or live_status == "NOT_DETECTED"
        or (
            result in {"NOT_AVAILABLE", "NOT_ASSESSED", None}
            and "detectable face" in (ver.get("reason") or "").lower()
        )
    ):
        if live_count == 0 or live_status == "NOT_DETECTED":
            result = "NO_MATCH"
            conflicts.append(
                "Live capture face was not detected or not comparable to the document face."
            )

    cal = ver.get("calibration") or (
        "UNCALIBRATED"
        if ver.get("calibrated") is False
        else "CALIBRATED"
        if ver.get("calibrated") is True
        else "NOT_ASSESSED"
    )

    if result == "MATCH":
        sources = (ver.get("embedding_sources") or "").upper()
        if sources in {"SAME_IMAGE_BYTES", "IDENTICAL_VECTOR"}:
            result = "INCONCLUSIVE"
            identity_status = "INCONCLUSIVE"
            conflicts.append("Self-comparison / identical embedding guard blocked MATCH.")
            checks.append(
                {
                    "item": "Face comparison",
                    "status": "WARN",
                    "detail": "MATCH blocked — identical embedding sources",
                }
            )
        else:
            checks.append(
                {
                    "item": "Face comparison",
                    "status": "OK",
                    "detail": f"MATCH · sim={ver.get('similarity')} · calibration={cal}",
                }
            )
            identity_status = "MATCH"
    elif result == "NO_MATCH":
        checks.append(
            {
                "item": "Face comparison",
                "status": "FAIL",
                "detail": f"NO_MATCH · sim={ver.get('similarity')} · calibration={cal}",
            }
        )
        identity_status = "NO_MATCH"
        if "sufficient face match" not in " ".join(conflicts).lower():
            conflicts.append(
                "The document face and presented person did not produce a sufficient face match."
            )
    elif result == "INCONCLUSIVE":
        checks.append(
            {
                "item": "Face comparison",
                "status": "WARN",
                "detail": f"INCONCLUSIVE · calibration={cal}",
            }
        )
        identity_status = "INCONCLUSIVE"
    elif result in {"NOT_AVAILABLE", "NOT_ASSESSED"} and cam.get("status") != "CAPTURED":
        if doc_face in {"DETECTED", "MULTIPLE_DETECTED"}:
            checks.append(
                {
                    "item": "Face comparison",
                    "status": "NA",
                    "detail": "NOT_AVAILABLE — awaiting live capture",
                }
            )
            identity_status = "AWAITING_LIVE"
            conflicts.append("Document face detected; live identity capture not completed.")
        else:
            checks.append({"item": "Face comparison", "status": "NA", "detail": "NOT_AVAILABLE"})
            identity_status = "NOT_ASSESSED"
    else:
        checks.append(
            {
                "item": "Face comparison",
                "status": "NA",
                "detail": result or "NOT_AVAILABLE",
            }
        )
        identity_status = (
            "NOT_ASSESSED"
            if result in {None, "NOT_AVAILABLE", "NOT_ASSESSED"}
            else "INCONCLUSIVE"
        )

    live_dec = (live.get("decision") or live.get("status") or "NOT_ASSESSED").upper()
    if "PASSIVE_OK" in live_dec or live_dec in {"PASS", "OK"}:
        checks.append({"item": "Liveness", "status": "OK", "detail": live.get("decision") or live_dec})
        checks.append(
            {
                "item": "Liveness note",
                "status": "INFO",
                "detail": "Liveness/PAD ≠ identity MATCH.",
            }
        )
    elif live_dec in {"NOT_APPLICABLE", "NOT_ASSESSED", "NOT_USED"}:
        checks.append({"item": "Liveness", "status": "NA", "detail": live_dec})
    elif "INCONCLUSIVE" in live_dec or "FAIL" in live_dec:
        checks.append({"item": "Liveness", "status": "WARN", "detail": live.get("decision") or live_dec})
    else:
        checks.append({"item": "Liveness", "status": "NA", "detail": live_dec})

    return {
        "status": identity_status,
        "calibration": cal
        if result not in {"NOT_AVAILABLE", "NOT_ASSESSED", None}
        else "NOT_ASSESSED",
        "result": result,
        "similarity": ver.get("similarity"),
        "technical_status": ver.get("status") or "NOT_ASSESSED",
        "technical_decision": ver.get("decision"),
        "checks": checks,
        "conflicts": conflicts,
        "document_face": doc_meta or {"status": doc_face},
        "live_face": live_meta or {"status": live_status, "face_count": live_count},
        "verification_debug": face.get("verification_debug"),
    }



def _document_from_signals(
    *,
    prior_decision: str,
    evidence_quality: float,
    authenticity_estimate: Optional[float],
    confidence: float,
    fusion: Dict[str, Any],
    cross_validation: Dict[str, Any],
    tampering: Dict[str, Any],
    calibration: Dict[str, Any],
    image_quality: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    checks: List[Dict[str, str]] = []
    conflicts: List[str] = []
    image_quality = image_quality or {}

    insufficient = bool(image_quality.get("insufficient_for_analysis")) or (
        prior_decision == "INCONCLUSIVE"
        and authenticity_estimate is None
        and evidence_quality < 0.50
    )

    eq_ok = evidence_quality >= 0.50
    checks.append(
        {
            "item": "Image quality",
            "status": "OK" if eq_ok and not insufficient else "WARN",
            "detail": f"evidence_quality={evidence_quality:.3f}",
        }
    )

    sev = tampering.get("severity") or "NOT_ASSESSED"
    review_score = tampering.get("review_score")
    flags = tampering.get("flags") or []
    if sev == "ELEVATED_REVIEW":
        checks.append(
            {
                "item": "Forensics",
                "status": "FAIL",
                "detail": f"{sev} · score={review_score} · flags={flags}",
            }
        )
    elif sev == "MILD_REVIEW":
        checks.append(
            {
                "item": "Forensics",
                "status": "WARN",
                "detail": f"{sev} · score={review_score}",
            }
        )
    elif sev in {"LOW_SIGNAL", "LOW", "CLEAR"}:
        checks.append({"item": "Forensics", "status": "OK", "detail": sev})
    else:
        checks.append({"item": "Forensics", "status": "NA", "detail": str(sev)})

    cv = cross_validation.get("overall")
    if cv == "PASS":
        checks.append({"item": "Cross-validation", "status": "OK", "detail": cv})
    elif cv == "WARNING":
        checks.append({"item": "Cross-validation", "status": "WARN", "detail": cv})
    elif cv == "FAIL":
        checks.append({"item": "Cross-validation", "status": "FAIL", "detail": cv})
        conflicts.append("Cross-validation FAIL.")
    else:
        checks.append({"item": "Cross-validation", "status": "NA", "detail": str(cv or "NOT_ASSESSED")})

    auth = authenticity_estimate
    cal_status = calibration.get("status")
    if auth is not None:
        checks.append(
            {
                "item": "Authenticity estimate",
                "status": "OK" if float(auth) >= 0.55 else "WARN",
                "detail": f"{float(auth):.3f} · calib={cal_status} · proxy={calibration.get('proxy_calibrated')}",
            }
        )
    else:
        checks.append(
            {
                "item": "Authenticity estimate",
                "status": "NA",
                "detail": f"NOT AVAILABLE · calib={cal_status}",
            }
        )

    fusion_auth = fusion.get("authentic_probability")
    fusion_tamp = fusion.get("tampered_probability")
    if fusion.get("status") == "FUSED" and fusion_auth is not None:
        checks.append(
            {
                "item": "Fusion / models",
                "status": "OK" if float(fusion_auth) >= 0.45 else "WARN",
                "detail": f"auth={fusion_auth} · tamp={fusion_tamp}",
            }
        )

    # --- document status ---
    if insufficient:
        return {
            "status": "INCONCLUSIVE",
            "checks": checks,
            "conflicts": conflicts
            + ["Insufficient image/OCR evidence for a reliable document conclusion."],
            "reason": "Insufficient reliable document/image evidence.",
        }

    elevated = sev == "ELEVATED_REVIEW"
    fails = int((cross_validation.get("counts") or {}).get("FAIL") or 0)
    # Prefer proxy/calibrated authenticity_estimate when present. Raw fusion alone
    # must not force SUSPICIOUS if the calibrated estimate supports authenticity.
    strong_fake_lean = (
        (auth is not None and float(auth) < 0.28)
        or (
            auth is None
            and fusion_tamp is not None
            and float(fusion_tamp) >= 0.72
            and (fusion_auth is None or float(fusion_auth) < 0.35)
        )
        or (cv == "FAIL" and fails >= 2)
    )

    if elevated:
        conflicts.append("Forensic analysis detected elevated tampering / anomaly signals.")
        return {
            "status": "TAMPERING_INDICATED",
            "checks": checks,
            "conflicts": conflicts,
            "reason": "Forensic analysis detected evidence requiring document review.",
        }

    if strong_fake_lean:
        conflicts.append("Model / consistency signals lean suspicious.")
        return {
            "status": "SUSPICIOUS",
            "checks": checks,
            "conflicts": conflicts,
            "reason": "Multiple indicators require further examination.",
        }

    # AUTHENTIC support: quality OK, no elevated forensics, no CV FAIL, and
    # proxy authenticity band (when estimate exists it must support authenticity)
    auth_ok = auth is not None and float(auth) >= 0.55
    mild_ok = sev in {"LOW_SIGNAL", "LOW", "MILD_REVIEW", "CLEAR", "NOT_ASSESSED", None}
    cv_ok = cv in {"PASS", "WARNING", None}
    conf_ok = confidence >= 0.45

    if eq_ok and mild_ok and cv_ok and fails == 0 and auth_ok:
        return {
            "status": "AUTHENTIC",
            "checks": checks,
            "conflicts": conflicts,
            "reason": "Document evidence is consistent; no significant tampering indicated.",
        }

    if (
        eq_ok
        and mild_ok
        and cv_ok
        and fails == 0
        and conf_ok
        and sev in {"LOW_SIGNAL", "LOW", "CLEAR"}
        and auth is None
    ):
        return {
            "status": "AUTHENTIC",
            "checks": checks,
            "conflicts": conflicts,
            "reason": "Document evidence is consistent; no significant tampering indicated.",
        }

    if sev == "MILD_REVIEW" or (auth is not None and float(auth) < 0.55):
        return {
            "status": "INCONCLUSIVE",
            "checks": checks,
            "conflicts": conflicts,
            "reason": "Document signals mixed or incomplete.",
        }

    return {
        "status": "INCONCLUSIVE",
        "checks": checks,
        "conflicts": conflicts,
        "reason": "Insufficient consistent document evidence for AUTHENTIC.",
    }


def _fuse(
    document: Dict[str, Any],
    identity: Dict[str, Any],
) -> Tuple[str, str, str, List[str], str]:
    """Returns (screening_decision, officer_code, officer_label, conflicts, summary).

    Officer rule:
      - Detected + real MATCH (doc↔live) + authentic document → PASS
      - Otherwise → INCONCLUSIVE
      - Tampering/suspicious → HIGH-RISK

    DETECTED alone is never PASS.
    """
    d = document.get("status")
    i = identity.get("status")
    conflicts = list(document.get("conflicts") or []) + list(identity.get("conflicts") or [])

    real_match = (
        d == "AUTHENTIC"
        and i == "MATCH"
        and identity.get("result") == "MATCH"
        and identity.get("similarity") is not None
    )
    if real_match:
        return (
            "PASS",
            "PASS_AUTHENTIC",
            "PASS — AUTHENTIC",
            conflicts,
            "Document looks authentic and live face MATCHED the document face (doc↔live compare).",
        )

    if d == "TAMPERING_INDICATED":
        if i == "MATCH":
            conflicts.append("Face MATCH does not cancel tampering evidence on the document.")
        return (
            "HIGH-RISK",
            "TAMPERING_INDICATED",
            "TAMPERING INDICATED",
            conflicts,
            "Forensic analysis detected evidence requiring document review.",
        )

    if d == "SUSPICIOUS":
        return (
            "HIGH-RISK",
            "SUSPICIOUS",
            "SUSPICIOUS / FAKE INDICATION",
            conflicts,
            "Multiple indicators require further examination.",
        )

    if i == "NO_MATCH":
        conflicts.append("Document face and live capture did not MATCH — result is INCONCLUSIVE.")
        summary = "Faces were detected but not a real match — INCONCLUSIVE."
    elif i == "FACE_NOT_DETECTED":
        conflicts.append("Document face was not detected.")
        summary = "Face not detected — INCONCLUSIVE (cannot PASS)."
    elif i == "AWAITING_LIVE":
        summary = "Live capture/match not completed — INCONCLUSIVE (cannot PASS)."
    elif i == "INCONCLUSIVE":
        conflicts.append("Identity comparison inconclusive.")
        summary = "Face comparison inconclusive — INCONCLUSIVE."
    elif d == "INCONCLUSIVE":
        summary = "Document evidence inconclusive — INCONCLUSIVE."
    elif i == "MATCH" and d != "AUTHENTIC":
        summary = "Face MATCHED but document is not AUTHENTIC — INCONCLUSIVE."
    else:
        summary = "Not a verified real face match — INCONCLUSIVE (DETECTED ≠ PASS)."

    return (
        "INCONCLUSIVE",
        "INCONCLUSIVE",
        "INCONCLUSIVE",
        conflicts,
        summary,
    )



def decide_screening(
    *,
    prior_decision: str,
    evidence_quality: float,
    authenticity_estimate: Optional[float],
    confidence: float,
    fusion: Optional[Dict[str, Any]] = None,
    cross_validation: Optional[Dict[str, Any]] = None,
    tampering: Optional[Dict[str, Any]] = None,
    calibration: Optional[Dict[str, Any]] = None,
    face: Optional[Dict[str, Any]] = None,
    image_quality: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    fusion = fusion or {}
    cross_validation = cross_validation or {}
    tampering = tampering or {}
    calibration = calibration or {}

    document = _document_from_signals(
        prior_decision=prior_decision,
        evidence_quality=float(evidence_quality or 0.0),
        authenticity_estimate=authenticity_estimate,
        confidence=float(confidence or 0.0),
        fusion=fusion,
        cross_validation=cross_validation,
        tampering=tampering,
        calibration=calibration,
        image_quality=image_quality,
    )
    identity = _identity_from_face(face)

    decision, officer_code, officer_label, conflicts, summary = _fuse(document, identity)

    actions = {
        "PASS": (
            "Routine review acceptable. Confirm with authorized procedures if required — "
            "PASS is not government verification."
        ),
        "HIGH-RISK": (
            "Prioritize human review. Treat as elevated screening risk — "
            "do not treat as a legal fraud finding."
        ),
        "REVIEW": (
            "Officer review required. Inspect document and identity evidence panels."
        ),
        "INCONCLUSIVE": (
            "Improve image quality / capture a clearer face match / resubmit if needed."
        ),
    }

    reasons: List[str] = [
        f"Document evidence: {document.get('status')}.",
        f"Identity evidence: {identity.get('status')}.",
        summary,
    ]
    for c in conflicts:
        if c not in reasons:
            reasons.append(c)

    # Preserve early quality INCONCLUSIVE when document truly insufficient and no identity match override
    if (
        prior_decision == "INCONCLUSIVE"
        and document.get("status") == "INCONCLUSIVE"
        and identity.get("status") in {"NOT_ASSESSED", "INCONCLUSIVE", "FACE_NOT_DETECTED", "AWAITING_LIVE"}
        and authenticity_estimate is None
        and evidence_quality < 0.45
    ):
        decision = "INCONCLUSIVE"
        officer_code = "INCONCLUSIVE"
        officer_label = "INCONCLUSIVE"
        summary = "Insufficient evidence for a reliable automated decision."
        reasons = [
            "Prior pipeline marked insufficient evidence/image quality.",
            summary,
        ]

    return {
        "status": "DECIDED",
        "decision": decision,
        "officer_verdict": {
            "code": officer_code,
            "label": officer_label,
            "summary": summary,
            "document_status": document.get("status"),
            "identity_status": identity.get("status"),
            "identity_calibration": identity.get("calibration"),
            "conflicts": conflicts,
            "document_checks": document.get("checks") or [],
            "identity_checks": identity.get("checks") or [],
        },
        "document_evidence": document,
        "identity_evidence": identity,
        "signals": {
            "prior_decision": prior_decision,
            "authenticity_estimate": authenticity_estimate,
            "confidence": confidence,
            "evidence_quality": evidence_quality,
            "cross_validation_overall": cross_validation.get("overall"),
            "tampering_severity": tampering.get("severity"),
            "calibration_status": calibration.get("status"),
            "fusion_status": fusion.get("status"),
            "identity_status": identity.get("status"),
            "identity_result": identity.get("result"),
            "identity_calibration": identity.get("calibration"),
        },
        "decision_reasons": reasons,
        "recommended_action": actions.get(decision, actions["REVIEW"]),
        "concern": "SCREENING_ASSIST_ONLY",
        "legal_authenticity": False,
        "note": (
            f"{decision} / officer={officer_label} is a screening triage label from "
            "document + identity evidence fusion. Not official authenticity or identity proof. "
            "UNCALIBRATED face similarity is shown separately from MATCH/NO_MATCH."
        ),
    }


def refine_with_identity(
    previous: Optional[Dict[str, Any]],
    *,
    face: Dict[str, Any],
) -> Dict[str, Any]:
    """Re-fuse officer verdict after live camera compare using prior document evidence."""
    previous = previous or {}
    document = previous.get("document_evidence")
    if not document:
        signals = previous.get("signals") or {}
        return decide_screening(
            prior_decision=str(signals.get("prior_decision") or previous.get("decision") or "PENDING"),
            evidence_quality=float(signals.get("evidence_quality") or 0.7),
            authenticity_estimate=signals.get("authenticity_estimate"),
            confidence=float(signals.get("confidence") or 0.6),
            fusion={"status": signals.get("fusion_status") or "NOT_ASSESSED"},
            cross_validation={"overall": signals.get("cross_validation_overall")},
            tampering={"severity": signals.get("tampering_severity")},
            calibration={"status": signals.get("calibration_status")},
            face=face,
        )

    identity = _identity_from_face(face)
    decision, officer_code, officer_label, conflicts, summary = _fuse(document, identity)
    actions = {
        "PASS": (
            "Routine review acceptable. Confirm with authorized procedures if required — "
            "PASS is not government verification."
        ),
        "HIGH-RISK": (
            "Prioritize human review. Treat as elevated screening risk — "
            "do not treat as a legal fraud finding."
        ),
        "REVIEW": "Officer review required. Inspect document and identity evidence panels.",
        "INCONCLUSIVE": "Improve image quality / capture a clearer face match / resubmit if needed.",
    }
    reasons = [
        f"Document evidence: {document.get('status')}.",
        f"Identity evidence: {identity.get('status')}.",
        summary,
    ] + list(conflicts)
    signals = dict(previous.get("signals") or {})
    signals["identity_status"] = identity.get("status")
    signals["identity_result"] = identity.get("result")
    signals["identity_calibration"] = identity.get("calibration")
    return {
        "status": "DECIDED",
        "decision": decision,
        "officer_verdict": {
            "code": officer_code,
            "label": officer_label,
            "summary": summary,
            "document_status": document.get("status"),
            "identity_status": identity.get("status"),
            "identity_calibration": identity.get("calibration"),
            "conflicts": conflicts,
            "document_checks": document.get("checks") or [],
            "identity_checks": identity.get("checks") or [],
        },
        "document_evidence": document,
        "identity_evidence": identity,
        "signals": signals,
        "decision_reasons": reasons,
        "recommended_action": actions.get(decision, actions["REVIEW"]),
        "concern": "SCREENING_ASSIST_ONLY",
        "legal_authenticity": False,
        "note": previous.get("note")
        or "Refined after live identity compare. Screening assist only.",
    }
