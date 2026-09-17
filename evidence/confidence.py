"""SIH Phase 16 — confidence / evidence support engine.

Produces a structured confidence breakdown for the current assessment.
This is support-for-assessment confidence, not authenticity accuracy.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def build_confidence_evidence(
    *,
    evidence_quality: float,
    doc_type_confidence: float,
    ocr_confidence: Optional[float],
    field_completeness: Optional[float],
    fusion: Optional[Dict[str, Any]] = None,
    cross_validation: Optional[Dict[str, Any]] = None,
    calibration: Optional[Dict[str, Any]] = None,
    tampering: Optional[Dict[str, Any]] = None,
    vision_ok: Optional[Dict[str, bool]] = None,
) -> Dict[str, Any]:
    fusion = fusion or {}
    cross_validation = cross_validation or {}
    calibration = calibration or {}
    tampering = tampering or {}
    vision_ok = vision_ok or {}

    factors: List[Dict[str, Any]] = [
        {
            "name": "evidence_quality",
            "weight": 0.28,
            "value": _clamp(evidence_quality),
            "note": "Image suitability for analysis",
        },
        {
            "name": "document_type",
            "weight": 0.14,
            "value": _clamp(doc_type_confidence),
            "note": "Classifier confidence",
        },
        {
            "name": "ocr",
            "weight": 0.14,
            "value": _clamp(ocr_confidence if ocr_confidence is not None else 0.35),
            "note": "Transcription confidence only",
        },
        {
            "name": "field_completeness",
            "weight": 0.10,
            "value": _clamp(field_completeness if field_completeness is not None else 0.35),
            "note": "Expected fields present",
        },
    ]

    vision_score = (
        0.34 * (1.0 if vision_ok.get("efficientnet") else 0.0)
        + 0.33 * (1.0 if vision_ok.get("vit") else 0.0)
        + 0.33 * (1.0 if vision_ok.get("clip") else 0.0)
    )
    factors.append(
        {
            "name": "vision_backbones",
            "weight": 0.12,
            "value": vision_score,
            "note": "EffNet / ViT / CLIP availability",
        }
    )

    fusion_v = 0.45
    if fusion.get("status") == "FUSED" and fusion.get("authentic_probability") is not None:
        # Higher fusion confidence when heads agree and probs are decisive
        auth = float(fusion.get("authentic_probability") or 0.5)
        fusion_v = 0.55 + 0.35 * abs(auth - 0.5) * 2
        if fusion.get("head_agreement") == "AGREE":
            fusion_v = min(1.0, fusion_v + 0.08)
    factors.append(
        {
            "name": "fusion",
            "weight": 0.10,
            "value": _clamp(fusion_v),
            "note": "Multimodal fusion presence / decisiveness",
        }
    )

    cv_rate = cross_validation.get("pass_rate")
    factors.append(
        {
            "name": "cross_validation",
            "weight": 0.08,
            "value": _clamp(cv_rate if cv_rate is not None else 0.4),
            "note": "Module consistency pass rate",
        }
    )

    cal_v = 0.35
    if calibration.get("status") == "CALIBRATED":
        cal_v = 0.75
    elif calibration.get("status") == "SKIPPED":
        cal_v = 0.25
    factors.append(
        {
            "name": "calibration",
            "weight": 0.04,
            "value": cal_v,
            "note": "Proxy Platt applied (not production)",
        }
    )

    # Soft penalty for elevated tampering (does not invent authenticity)
    tamp_pen = 0.0
    if tampering.get("severity") == "ELEVATED_REVIEW":
        tamp_pen = 0.08
    elif tampering.get("severity") in {"MODERATE_REVIEW", "MODERATE"}:
        tamp_pen = 0.04

    weighted = sum(f["weight"] * f["value"] for f in factors)
    overall = round(_clamp(weighted - tamp_pen), 4)

    return {
        "status": "SCORED",
        "overall_confidence": overall,
        "factors": factors,
        "tampering_penalty": tamp_pen,
        "concern": "ASSESSMENT_SUPPORT_ONLY",
        "note": (
            "Confidence measures support for the current screening assessment given "
            "available evidence. It is not accuracy %, not legal authenticity, and not "
            "a claim about government database matches."
        ),
    }
