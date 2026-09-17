"""Analysis pipeline — SIH: quality, classify, OCR/MRZ, EffNet/ViT heads, CLIP features."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict

from document_profiles import get_profile
from app.ml.extract import extract_vision_features
from app.pipeline.classifier import classify_document_type
from app.pipeline.quality import analyze_image_quality
from evidence.calibration import calibrate_fusion_score
from evidence.confidence import build_confidence_evidence
from evidence.cross_validation import run_cross_validation
from evidence.decision import decide_screening
from evidence.fusion import fuse_evidence
from forensics.tampering import analyze_tampering
from forensics.text_analysis import analyze_text_visual
from forensics.metadata_stamp import (
    analyze_metadata,
    analyze_photo_region_splicing,
    analyze_stamp_seal,
    build_forensic_map,
)
from ocr.engine import run_ocr
from ocr.fields import extract_fields
from ocr.mrz import analyze_mrz
from verification.face import analyze_document_face

logger = logging.getLogger("docshield.pipeline")


def run_phase3_pipeline(image_bytes: bytes, filename: str = "") -> Dict[str, Any]:
    """Backward-compatible alias."""
    return run_pipeline(image_bytes, filename=filename)


def run_pipeline(image_bytes: bytes, filename: str = "") -> Dict[str, Any]:
    quality = analyze_image_quality(image_bytes)
    doc_type = classify_document_type(image_bytes, filename=filename)
    profile = get_profile(doc_type.label)

    ocr_raw = {
        "status": "SKIPPED",
        "message": "OCR skipped due to insufficient image quality.",
        "text_boxes": [],
        "full_text": "",
        "mean_ocr_confidence": None,
        "note": "OCR confidence is transcription confidence, NOT authenticity confidence.",
    }
    fields_result: Dict[str, Any] = {
        "status": "NOT_ASSESSED",
        "fields": [],
        "field_inventory": [],
        "missing_expected_fields": [
            f for f in profile.expected_fields if f not in {"photo", "signature", "mrz"}
        ],
    }
    mrz_result: Dict[str, Any] = {
        "status": "NOT_APPLICABLE" if not profile.supports_mrz else "NOT_ASSESSED"
    }
    vision_features: Dict[str, Any] = {
        "status": "SKIPPED",
        "message": "Vision features skipped due to insufficient image quality.",
        "note": "Pretrained backbones are not fraud detectors.",
    }
    module_errors: Dict[str, str] = {}

    # Phase 9–10: tampering + localization
    try:
        tampering_result = analyze_tampering(image_bytes)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Tampering module failed")
        module_errors["tampering"] = repr(exc)
        tampering_result = {
            "status": "ERROR",
            "error": repr(exc),
            "flags": [],
            "concern": "NOT_ASSESSED",
            "note": "Tampering heuristics failed. Anomalies ≠ proof of fraud.",
            "localization": {
                "status": "NOT_ASSESSED",
                "note": "Localization unavailable after tampering error.",
            },
        }

    # Phase 11: face detection on document (no reference → no identity claim)
    try:
        face_result = analyze_document_face(image_bytes)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Face module failed")
        module_errors["face"] = repr(exc)
        face_result = {
            "status": "ERROR",
            "error": repr(exc),
            "face_count": 0,
            "faces": [],
            "verification": {"status": "NOT_ASSESSED"},
            "liveness": {"status": "NOT_ASSESSED"},
            "concern": "NOT_ASSESSED",
            "note": "Face detection failed.",
        }

    if not quality.insufficient_for_analysis:
        try:
            ocr_raw = run_ocr(image_bytes)
        except Exception as exc:  # noqa: BLE001 — isolate OCR crash
            logger.exception("OCR module failed")
            module_errors["ocr"] = repr(exc)
            ocr_raw = {
                "status": "ERROR",
                "error": repr(exc),
                "text_boxes": [],
                "full_text": "",
                "mean_ocr_confidence": None,
                "note": "OCR confidence is transcription confidence, NOT authenticity confidence.",
            }

        if ocr_raw.get("status") == "OK":
            try:
                fields_result = extract_fields(
                    ocr_raw.get("full_text", ""),
                    ocr_raw.get("text_boxes", []),
                    doc_type.label,
                )
                fields_result = {
                    k: v for k, v in fields_result.items() if k != "compare_values"
                }
            except Exception as exc:  # noqa: BLE001
                logger.exception("Field extraction failed")
                module_errors["fields"] = repr(exc)
                fields_result = {
                    "status": "ERROR",
                    "fields": [],
                    "field_inventory": [],
                    "message": repr(exc),
                }
            try:
                mrz_result = analyze_mrz(
                    ocr_raw.get("full_text", ""),
                    doc_type.label,
                    profile.supports_mrz,
                    fields_payload=fields_result,
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception("MRZ module failed")
                module_errors["mrz"] = repr(exc)
                mrz_result = {
                    "status": "ERROR" if profile.supports_mrz else "NOT_APPLICABLE",
                    "message": repr(exc),
                }
        else:
            fields_result = {
                "status": "ERROR" if ocr_raw.get("status") == "ERROR" else "NOT_ASSESSED",
                "fields": [],
                "field_inventory": [],
                "message": ocr_raw.get("error") or "OCR not assessed.",
            }
            mrz_result = {
                "status": "NOT_APPLICABLE"
                if not profile.supports_mrz
                else "NOT_DETECTED",
                "message": "MRZ not assessed because OCR was unavailable.",
            }

        try:
            vision_features = extract_vision_features(
                image_bytes, predicted_document_type=doc_type.label
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Vision / AI forensics failed")
            module_errors["ai_forensics"] = repr(exc)
            vision_features = {
                "status": "ERROR",
                "message": repr(exc),
                "note": "Pretrained backbones are not fraud detectors.",
            }

    ocr_evidence = {
        "status": ocr_raw.get("status"),
        "engine": ocr_raw.get("engine"),
        "mean_ocr_confidence": ocr_raw.get("mean_ocr_confidence"),
        "text_box_count": ocr_raw.get("text_box_count", len(ocr_raw.get("text_boxes", []))),
        "text_boxes": ocr_raw.get("text_boxes", []),
        "full_text_preview": (ocr_raw.get("full_text") or "")[:500],
        "preprocess": ocr_raw.get("preprocess"),
        "error": ocr_raw.get("error"),
        "note": ocr_raw.get("note"),
    }

    try:
        text_visual = analyze_text_visual(
            image_bytes,
            text_boxes=ocr_evidence.get("text_boxes") or [],
            fields=fields_result,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Text visual analysis failed")
        module_errors["text_visual"] = repr(exc)
        text_visual = {
            "status": "ERROR",
            "error": repr(exc),
            "typography": {"status": "NOT_ASSESSED"},
            "text_splicing": {"status": "NOT_ASSESSED"},
            "ocr_visual_crosscheck": [],
            "concern": "NOT_ASSESSED",
        }

    try:
        metadata_result = analyze_metadata(image_bytes)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Metadata analysis failed")
        module_errors["metadata"] = repr(exc)
        metadata_result = {
            "status": "ERROR",
            "error": repr(exc),
            "existance": "UNAVAILABLE",
            "note": "Metadata analysis failed — missing metadata is not fraud proof.",
        }

    try:
        stamp_result = analyze_stamp_seal(image_bytes)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Stamp analysis failed")
        module_errors["stamp"] = repr(exc)
        stamp_result = {"status": "ERROR", "error": repr(exc), "detection": "NOT_ASSESSED"}

    face0 = (face_result.get("faces") or [{}])[0]
    fb = face0.get("bbox")
    if isinstance(fb, dict):
        face_bbox = [fb.get("x"), fb.get("y"), fb.get("w"), fb.get("h")]
    else:
        face_bbox = fb
    try:
        photo_splice = analyze_photo_region_splicing(image_bytes, face_bbox=face_bbox)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Photo splicing analysis failed")
        module_errors["photo_splicing"] = repr(exc)
        photo_splice = {"status": "ERROR", "error": repr(exc)}

    if isinstance(tampering_result.get("signals"), dict):
        tampering_result["signals"]["metadata"] = metadata_result

    forensic_map = build_forensic_map(
        tampering=tampering_result,
        text_visual=text_visual,
        photo=photo_splice,
        stamp=stamp_result,
        metadata=metadata_result,
    )

    evidence = {
        "image_quality": quality.to_dict(),
        "document_type": doc_type.to_dict(),
        "ocr": ocr_evidence,
        "fields": fields_result,
        "mrz": mrz_result,
        "layout": {"status": "NOT_ASSESSED"},
        "ai_forensics": vision_features,
        "fusion": {"status": "PENDING"},
        "tampering": tampering_result,
        "text_visual": text_visual,
        "forensics": forensic_map,
        "metadata": metadata_result,
        "stamp": stamp_result,
        "photo_splicing": photo_splice,
        "ai_edit": {"status": "NOT_ASSESSED"},
        "face": face_result,
        "duplicate": {"status": "NOT_ASSESSED"},
        "official_verification": {"status": "UNAVAILABLE"},
        "watchlist": {"status": "NOT_ASSESSED"},
        "reverse_search": {"status": "SEARCH_UNAVAILABLE"},
        "consistency": {"status": "PENDING"},
        "cross_validation": {"status": "PENDING"},
        "calibration": {"status": "PENDING"},
    }

    fusion_result = fuse_evidence(
        vision_features=vision_features,
        evidence_quality=quality.overall_quality,
        ocr_confidence=ocr_raw.get("mean_ocr_confidence"),
        field_completeness=fields_result.get("field_completeness"),
        doc_type_confidence=doc_type.confidence,
        mrz=mrz_result,
    )
    evidence["fusion"] = fusion_result
    # Keep nested fusion status on ai_forensics in sync for UI consumers
    if isinstance(vision_features, dict):
        vision_features["fusion"] = {
            "status": fusion_result.get("status"),
            "message": "See evidence.fusion for multimodal fusion output.",
            "inputs_ready": fusion_result.get("modalities_present"),
        }

    try:
        cross_val = run_cross_validation(
            quality=quality.to_dict(),
            document_type=doc_type.to_dict(),
            ocr=ocr_evidence,
            fields=fields_result,
            mrz=mrz_result,
            face=face_result,
            vision=vision_features if isinstance(vision_features, dict) else {},
            fusion=fusion_result,
            tampering=tampering_result,
            text_visual=text_visual,
            profile_supports_mrz=profile.supports_mrz,
            profile_expects_photo="photo" in (profile.expected_fields or ()),
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Cross-validation failed")
        module_errors["cross_validation"] = repr(exc)
        cross_val = {
            "status": "ERROR",
            "overall": "NOT_ASSESSED",
            "error": repr(exc),
            "checks": [],
            "concern": "NOT_ASSESSED",
            "note": "Cross-validation failed; other modules still available.",
        }
    evidence["cross_validation"] = cross_val
    evidence["consistency"] = {
        "status": cross_val.get("overall") or "NOT_ASSESSED",
        "mrz_visible": mrz_result.get("consistency"),
        "cross_validation": {
            "overall": cross_val.get("overall"),
            "pass_rate": cross_val.get("pass_rate"),
            "severity": cross_val.get("severity"),
        },
        "note": (
            "Cross-validation + MRZ↔visible consistency are review signals, "
            "not authenticity proof."
        ),
    }

    evidence_quality = quality.overall_quality
    ocr_conf = ocr_raw.get("mean_ocr_confidence")
    field_conf = fields_result.get("mean_field_ocr_confidence")

    enet_ok = vision_features.get("efficientnet", {}).get("status") == "FEATURES_EXTRACTED"
    vit_ok = vision_features.get("vit", {}).get("status") == "FEATURES_EXTRACTED"
    clip_ok = vision_features.get("clip", {}).get("status") == "FEATURES_EXTRACTED"

    reasons = []
    if quality.insufficient_for_analysis:
        decision = "INCONCLUSIVE"
        reasons.append(
            quality.reason or "Image quality is insufficient for reliable analysis."
        )
        recommended_action = "Upload a clearer, well-lit, full-document image and retry."
        confidence = round(min(0.55, evidence_quality + 0.1), 4)
        message = (
            "INCONCLUSIVE: evidence/image quality is insufficient for reliable screening. "
            "Authenticity was not estimated."
        )
        if tampering_result.get("status") == "ANALYZED":
            reasons.append(
                f"Tampering heuristics still ran: {tampering_result.get('severity')} · "
                f"flags={tampering_result.get('flags') or ['none']} — review only."
            )
        if cross_val.get("status") == "COMPLETED":
            reasons.append(
                f"Cross-validation: overall={cross_val.get('overall')} — review only "
                "(not authenticity)."
            )
    elif ocr_raw.get("status") in {"NOT_ASSESSED", "ERROR"}:
        decision = "INCONCLUSIVE"
        reasons.append("OCR could not be assessed." if ocr_raw.get("status") != "ERROR" else "OCR module returned ERROR.")
        if ocr_raw.get("error"):
            reasons.append(str(ocr_raw["error"]))
        recommended_action = "Ensure Tesseract OCR is installed, or upload a clearer image."
        confidence = round(min(0.6, 0.5 * evidence_quality + 0.3 * doc_type.confidence), 4)
        message = "INCONCLUSIVE: OCR not assessed. Authenticity was not estimated."
    elif (
        ocr_conf is not None
        and ocr_conf < 0.35
        and fields_result.get("status") == "NO_FIELDS"
    ):
        decision = "INCONCLUSIVE"
        reasons.append("OCR confidence too low and no fields extracted.")
        recommended_action = "Upload a sharper full-document image."
        confidence = round(0.4 * evidence_quality + 0.2 * (ocr_conf or 0), 4)
        message = "INCONCLUSIVE: OCR quality insufficient for field extraction."
    else:
        decision = "PENDING"
        reasons.append(
            f"Document classified as {doc_type.label} "
            f"({doc_type.confidence * 100:.1f}% heuristic confidence)."
        )
        if fields_result.get("fields"):
            reasons.append(
                f"Extracted {len(fields_result['fields'])} masked field(s) via OCR."
            )
            completeness = fields_result.get("field_completeness")
            if completeness is not None:
                reasons.append(
                    f"Profile field completeness: {completeness * 100:.0f}% "
                    "(missing fields are not proof of fraud)."
                )
            missing = fields_result.get("missing_expected_fields") or []
            if missing:
                reasons.append(
                    "Missing expected OCR fields: " + ", ".join(missing[:6])
                    + ("…" if len(missing) > 6 else "")
                )
            for chk in fields_result.get("consistency_checks") or []:
                if chk.get("status") in {"FAIL", "WARNING"}:
                    reasons.append(f"Field check {chk.get('check')}: {chk.get('status')}")
        else:
            reasons.append("OCR ran but no high-confidence identity fields were extracted.")
        if mrz_result.get("status") == "PARSED":
            reasons.append("MRZ structure parsed; check-digit results available for review.")
            for row in mrz_result.get("checklist") or []:
                if row.get("status") == "FAIL":
                    reasons.append(
                        f"MRZ checklist: {row.get('label')} → FAIL (review signal, not automatic fraud)."
                    )
                elif row.get("status") == "PASS" and row.get("item") in {
                    "name_consistency",
                    "dob_consistency",
                    "document_number_consistency",
                }:
                    reasons.append(f"MRZ checklist: {row.get('label')} → PASS.")
        elif mrz_result.get("status") == "NOT_APPLICABLE":
            reasons.append("MRZ not applicable for this document type.")
        elif mrz_result.get("status") == "NOT_DETECTED":
            reasons.append("MRZ not detected on this image (not fabricated).")
        elif mrz_result.get("status") == "PARTIAL":
            reasons.append("MRZ-like text incomplete — marked PARTIAL / NOT fully assessed.")

        extracted = []
        if enet_ok:
            extracted.append("EfficientNet")
        if vit_ok:
            extracted.append("ViT")
        if clip_ok:
            extracted.append("CLIP")
        if extracted:
            reasons.append(
                f"Vision features extracted: {', '.join(extracted)}."
            )
        else:
            reasons.append("Vision backbone features were not assessed.")

        enet_head = (vision_features.get("efficientnet") or {}).get("authenticity_head") or {}
        if enet_head.get("status") == "SCORED":
            reasons.append(
                f"EfficientNet authenticity head (uncalibrated/synthetic): "
                f"authentic={enet_head.get('authentic_probability')} · "
                f"tampered={enet_head.get('tampered_probability')} — "
                "not a final authenticity verdict."
            )
        vit_head = (vision_features.get("vit") or {}).get("authenticity_head") or {}
        if vit_head.get("status") == "SCORED":
            reasons.append(
                f"ViT authenticity head (uncalibrated/synthetic): "
                f"authentic={vit_head.get('authentic_probability')} · "
                f"tampered={vit_head.get('tampered_probability')} — "
                "not a final authenticity verdict."
            )
        clip_support = (vision_features.get("clip") or {}).get("supporting_evidence") or {}
        if clip_support.get("status") == "SUPPORTING_SCORED":
            align = clip_support.get("document_type_alignment") or {}
            reasons.append(
                f"CLIP supporting type alignment: top={align.get('top_label')} "
                f"({(align.get('top_probability') or 0) * 100:.1f}%) · "
                f"vs classifier={align.get('agreement')} — NOT authenticity."
            )
        if fusion_result.get("status") == "FUSED":
            reasons.append(
                f"Multimodal fusion (uncalibrated/synthetic LR): "
                f"authentic={fusion_result.get('authentic_probability')} · "
                f"tampered={fusion_result.get('tampered_probability')} · "
                f"head_agreement={fusion_result.get('head_agreement')} — "
                "not authenticity_estimate."
            )
        if tampering_result.get("status") == "ANALYZED":
            flags = tampering_result.get("flags") or []
            severity = tampering_result.get("severity") or "LOW_SIGNAL"
            reasons.append(
                f"Tampering heuristics: {severity} · review_score="
                f"{tampering_result.get('review_score')} · "
                f"flags={flags or ['none']} — review signals only, not fraud proof."
            )
            loc = tampering_result.get("localization") or {}
            if loc.get("status") == "LOCALIZED":
                reasons.append(
                    f"Tampering localization: {loc.get('hotspot_count', 0)} hotspot(s) · "
                    f"peak={loc.get('peak_score')} · coverage={loc.get('high_anomaly_coverage')} — "
                    "heatmap is heuristic review only."
                )
        elif tampering_result.get("status") == "ERROR":
            reasons.append("Tampering module returned ERROR (logged); continuing without it.")
        if face_result.get("status") in {"DETECTED", "MULTIPLE_DETECTED"}:
            q0 = ((face_result.get("faces") or [{}])[0].get("quality") or {}).get("label")
            reasons.append(
                f"Face detection: {face_result.get('status')} · count={face_result.get('face_count')} · "
                f"quality={q0 or '—'} — detection only, not identity match."
            )
            live = face_result.get("liveness") or {}
            reasons.append(
                f"Liveness: {live.get('status')} · decision={live.get('decision')} — "
                f"{'PAD N/A on document photo' if live.get('status') == 'NOT_APPLICABLE' else 'passive PAD only on live camera'}."
            )
        elif face_result.get("status") == "NOT_DETECTED":
            reasons.append("Face detection: NOT_DETECTED on this image (not fabricated).")
            live = face_result.get("liveness") or {}
            if live.get("status") == "NOT_APPLICABLE":
                reasons.append("Liveness: NOT_APPLICABLE for document-only analyze.")
        elif face_result.get("status") == "ERROR":
            reasons.append("Face module returned ERROR (logged); continuing without it.")
        reasons.append(
            "Screening decision engine will assign PASS / REVIEW / HIGH-RISK after calibration."
        )
        if cross_val.get("status") == "COMPLETED":
            reasons.append(
                f"Cross-validation: overall={cross_val.get('overall')} · "
                f"pass_rate={cross_val.get('pass_rate')} · "
                f"warnings={((cross_val.get('counts') or {}).get('WARNING') or 0)} · "
                f"fails={((cross_val.get('counts') or {}).get('FAIL') or 0)} — review only."
            )
        recommended_action = (
            "Review OCR/MRZ, model heads, CLIP, fusion, tampering, face, liveness, "
            "cross-validation, and the proxy-calibrated authenticity estimate as signals "
            "only — not a legal verdict."
        )
        confidence = round(
            min(
                0.92,
                0.30 * evidence_quality
                + 0.20 * doc_type.confidence
                + 0.20 * (ocr_conf or 0.4)
                + 0.10 * (field_conf or 0.4)
                + 0.07 * (1.0 if enet_ok else 0.0)
                + 0.07 * (1.0 if vit_ok else 0.0)
                + 0.06 * (1.0 if clip_ok else 0.0),
            ),
            4,
        )
        message = (
            f"{doc_type.label} · OCR {ocr_raw.get('status')} · "
            f"vision {vision_features.get('status')} · "
            f"{len(fields_result.get('fields') or [])} field(s). "
            "Proxy-calibrated authenticity estimate available when fusion scored."
        )

    # --- Phase 15: Platt calibration of fusion → authenticity_estimate ---
    try:
        calibration = calibrate_fusion_score(
            fusion=fusion_result,
            evidence_quality_ok=not quality.insufficient_for_analysis,
            decision=decision,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Calibration failed")
        module_errors["calibration"] = repr(exc)
        calibration = {
            "status": "ERROR",
            "authenticity_estimate": None,
            "proxy_calibrated": False,
            "production_calibrated": False,
            "error": repr(exc),
            "concern": "CALIBRATION_ERROR",
            "note": "Calibration failed; authenticity_estimate left null.",
        }
    evidence["calibration"] = calibration
    authenticity_estimate = calibration.get("authenticity_estimate")
    if calibration.get("status") == "CALIBRATED" and authenticity_estimate is not None:
        reasons.append(
            f"Proxy-calibrated authenticity_estimate={float(authenticity_estimate):.1%} "
            f"(raw fusion={calibration.get('raw_authentic_probability')}) — "
            "synthetic Platt mapping only; not production / legal authenticity."
        )

    # --- Phase 16: confidence / evidence support engine ---
    confidence_engine = build_confidence_evidence(
        evidence_quality=evidence_quality,
        doc_type_confidence=doc_type.confidence,
        ocr_confidence=ocr_conf,
        field_completeness=fields_result.get("field_completeness"),
        fusion=fusion_result,
        cross_validation=cross_val,
        calibration=calibration,
        tampering=tampering_result,
        vision_ok={"efficientnet": enet_ok, "vit": vit_ok, "clip": clip_ok},
    )
    evidence["confidence_engine"] = confidence_engine
    confidence = float(confidence_engine.get("overall_confidence") or confidence)

    # --- Phase 17: final screening decision (document + identity fusion) ---
    decision_engine = decide_screening(
        prior_decision=decision,
        evidence_quality=evidence_quality,
        authenticity_estimate=authenticity_estimate,
        confidence=confidence,
        fusion=fusion_result,
        cross_validation=cross_val,
        tampering=tampering_result,
        calibration=calibration,
        face=face_result,
        image_quality=quality.to_dict() if hasattr(quality, "to_dict") else quality,
    )
    evidence["decision_engine"] = decision_engine
    decision = decision_engine.get("decision") or decision
    if decision_engine.get("recommended_action"):
        recommended_action = decision_engine["recommended_action"]
    for dr in decision_engine.get("decision_reasons") or []:
        if dr not in reasons:
            reasons.append(dr)
    reasons.append(
        f"Screening decision={decision} (assist only — not legal authenticity)."
    )

    # --- Phase 19: explainability / model comparison snapshot ---
    explainability = {
        "status": "READY",
        "model_comparison": {
            "efficientnet_auth": (
                (vision_features.get("efficientnet") or {})
                .get("authenticity_head", {})
                .get("authentic_probability")
            ),
            "vit_auth": (
                (vision_features.get("vit") or {})
                .get("authenticity_head", {})
                .get("authentic_probability")
            ),
            "fusion_auth_raw": fusion_result.get("authentic_probability"),
            "authenticity_estimate_proxy": authenticity_estimate,
            "clip_alignment": (
                ((vision_features.get("clip") or {}).get("supporting_evidence") or {})
                .get("document_type_alignment", {})
                .get("agreement")
            ),
        },
        "heatmap_available": (tampering_result.get("localization") or {}).get("status")
        == "LOCALIZED",
        "confidence_factors": confidence_engine.get("factors"),
        "note": (
            "Explainability surfaces relative model signals and heatmap availability. "
            "Not a causal proof of fraud or authenticity."
        ),
    }
    evidence["explainability"] = explainability

    enet_head = (vision_features.get("efficientnet") or {}).get("authenticity_head") or {}
    enet_concern = enet_head.get("concern") or vision_features.get("efficientnet", {}).get(
        "concern", "NOT_ASSESSED"
    )
    vit_head = (vision_features.get("vit") or {}).get("authenticity_head") or {}
    vit_concern = vit_head.get("concern") or vision_features.get("vit", {}).get(
        "concern", "NOT_ASSESSED"
    )
    clip_support = (vision_features.get("clip") or {}).get("supporting_evidence") or {}
    clip_align = clip_support.get("document_type_alignment") or {}
    clip_concern = clip_support.get("concern") or vision_features.get("clip", {}).get(
        "concern", "NOT_ASSESSED"
    )

    assessment = {
        "decision": decision,
        "officer_verdict": decision_engine.get("officer_verdict"),
        "authenticity_estimate": authenticity_estimate,
        "confidence": confidence,
        "evidence_quality": evidence_quality,
        "reasons": reasons,
        "recommended_action": recommended_action,
        "model_agreement": {
            "efficientnet": enet_concern,
            "efficientnet_authentic_probability": enet_head.get("authentic_probability"),
            "efficientnet_tampered_probability": enet_head.get("tampered_probability"),
            "vit": vit_concern,
            "vit_authentic_probability": vit_head.get("authentic_probability"),
            "vit_tampered_probability": vit_head.get("tampered_probability"),
            "clip": clip_concern,
            "clip_type_alignment": clip_align.get("agreement"),
            "clip_top_label": clip_align.get("top_label"),
            "fusion": fusion_result.get("concern") or fusion_result.get("status"),
            "fusion_authentic_probability": fusion_result.get("authentic_probability"),
            "fusion_tampered_probability": fusion_result.get("tampered_probability"),
            "fusion_head_agreement": fusion_result.get("head_agreement"),
            "ocr": fields_result.get("status") or ocr_raw.get("status", "NOT_ASSESSED"),
            "tampering": tampering_result.get("severity")
            or tampering_result.get("status", "NOT_ASSESSED"),
            "tampering_flags": tampering_result.get("flags") or [],
            "tampering_review_score": tampering_result.get("review_score"),
            "face": face_result.get("status"),
            "face_count": face_result.get("face_count"),
            "face_verification": (face_result.get("verification") or {}).get("status"),
            "liveness": (face_result.get("liveness") or {}).get("status"),
            "cross_validation": cross_val.get("overall") or cross_val.get("status"),
            "cross_validation_pass_rate": cross_val.get("pass_rate"),
            "calibration": calibration.get("status"),
            "calibration_raw": calibration.get("raw_authentic_probability"),
            "calibration_proxy": calibration.get("proxy_calibrated"),
            "confidence_engine": confidence_engine.get("overall_confidence"),
            "decision_engine": decision_engine.get("decision"),
            "note": (
                "Screening signals only. authenticity_estimate is proxy-calibrated "
                "(synthetic Platt). Decision is PASS/REVIEW/HIGH-RISK/INCONCLUSIVE for "
                "triage — NOT legal authenticity / NOT official verification."
            ),
        },
        "score_definitions": {
            "authenticity_estimate": (
                "Proxy-calibrated P(authentic) from fusion via synthetic Platt scaling. "
                "Not production-calibrated; not a legal authenticity verdict."
            ),
            "confidence": (
                "Structured support for the current screening assessment "
                "(evidence.confidence_engine) — not accuracy %."
            ),
            "evidence_quality": "Suitability of the submitted image/evidence for analysis.",
            "ocr_confidence": "Transcription confidence only — never authenticity.",
            "efficientnet_authentic_probability": (
                "Uncalibrated EfficientNet head P(authentic) on synthetic proxy — not legal authenticity."
            ),
            "vit_authentic_probability": (
                "Uncalibrated ViT head P(authentic) on synthetic proxy — not legal authenticity."
            ),
            "clip_type_alignment": (
                "CLIP text–image document-type alignment — supporting cue only, never authenticity."
            ),
            "fusion_authentic_probability": (
                "Uncalibrated multimodal fusion P(authentic) — raw input to Platt calibration."
            ),
            "tampering_review_score": (
                "Heuristic visual inconsistency aggregate (0–1) — review signal only, not authenticity."
            ),
        },
    }

    def _mark(ok: bool, label: str) -> str:
        return f"✓ {label}" if ok else f"○ {label}"

    pipeline = [
        "✓ Upload validation",
        "✓ Image quality",
        "✓ Document classification",
        "✓ OCR" if ocr_raw.get("status") in {"OK", "SKIPPED"} else "⚠ OCR",
        "✓ Field extraction" if fields_result.get("fields") else "○ Field extraction",
        "✓ MRZ"
        if mrz_result.get("status") in {"PARSED", "NOT_APPLICABLE", "NOT_DETECTED", "PARTIAL"}
        else "○ MRZ",
        _mark(enet_ok, "EfficientNet features"),
        _mark(
            enet_head.get("status") == "SCORED",
            "EfficientNet authenticity head",
        ),
        _mark(vit_ok, "ViT features"),
        _mark(
            vit_head.get("status") == "SCORED",
            "ViT authenticity head",
        ),
        _mark(clip_ok, "CLIP features"),
        _mark(
            clip_support.get("status") == "SUPPORTING_SCORED",
            "CLIP supporting evidence",
        ),
        _mark(fusion_result.get("status") == "FUSED", "Feature fusion"),
        _mark(tampering_result.get("status") == "ANALYZED", "Tampering heuristics"),
        _mark(
            (tampering_result.get("localization") or {}).get("status") == "LOCALIZED",
            "Tampering localization / heatmaps",
        ),
        _mark(
            face_result.get("status") in {"DETECTED", "MULTIPLE_DETECTED", "NOT_DETECTED"},
            "Face detection",
        ),
        "✓ Live camera module (use /api/camera — not used on document-only analyze)",
        _mark(
            (face_result.get("liveness") or {}).get("status")
            in {"NOT_APPLICABLE", "PASSIVE_ASSESSED", "NOT_ASSESSED"},
            "Face liveness / passive PAD",
        ),
        _mark(cross_val.get("status") == "COMPLETED", "Cross-validation"),
        _mark(calibration.get("status") == "CALIBRATED", "Probability calibration"),
        _mark(confidence_engine.get("status") == "SCORED", "Confidence / evidence engine"),
        _mark(decision_engine.get("status") == "DECIDED", "Decision engine"),
        _mark(explainability.get("status") == "READY", "Explainability / model comparison"),
        "✓ Report generation (GET /api/report/{case_id})",
        "✓ Audit hash chain (local / NOT CONNECTED)",
        "✓ Demo mode + quality gate docs",
    ]


    return {
        "evidence": evidence,
        "assessment": assessment,
        "pipeline": pipeline,
        "message": message,
        "document_type_label": doc_type.label,
        "evidence_json": json.dumps(evidence, separators=(",", ":")),
        "screening_result": {
            "document_type": evidence["document_type"],
            "image_quality": evidence["image_quality"],
            "ocr": evidence["ocr"],
            "document_validation": {
                "status": fields_result.get("status"),
                "fields": fields_result.get("fields"),
                "field_inventory": fields_result.get("field_inventory"),
                "field_completeness": fields_result.get("field_completeness"),
                "completeness_basis": fields_result.get("completeness_basis"),
                "consistency_checks": fields_result.get("consistency_checks"),
            },
            "mrz": evidence["mrz"],
            "ai_forensics": evidence["ai_forensics"],
            "efficientnet": (vision_features.get("efficientnet") if isinstance(vision_features, dict) else None),
            "vit": (vision_features.get("vit") if isinstance(vision_features, dict) else None),
            "clip": (vision_features.get("clip") if isinstance(vision_features, dict) else None),
            "tampering": evidence.get("tampering"),
            "fusion": evidence.get("fusion"),
            "phase_status": {
                "1_secure_quality": "COMPLETED",
                "2_classification": "COMPLETED",
                "3_ocr_fields": fields_result.get("status") or ocr_raw.get("status"),
                "4_mrz": mrz_result.get("status"),
                "5_efficientnet": (vision_features.get("efficientnet") or {}).get("status")
                if isinstance(vision_features, dict)
                else "NOT_ASSESSED",
                "6_vit": (vision_features.get("vit") or {}).get("status")
                if isinstance(vision_features, dict)
                else "NOT_ASSESSED",
                "7_clip": (vision_features.get("clip") or {}).get("status")
                if isinstance(vision_features, dict)
                else "NOT_ASSESSED",
                "8_fusion": fusion_result.get("status"),
                "9_tampering": tampering_result.get("status"),
                "10_localization": (tampering_result.get("localization") or {}).get("status"),
                "11_face": face_result.get("status"),
                "12_camera": (face_result.get("camera") or {}).get("status") or "NOT_USED",
                "13_liveness": (face_result.get("liveness") or {}).get("status"),
                "14_cross_validation": cross_val.get("status"),
                "15_calibration": calibration.get("status"),
                "16_confidence": confidence_engine.get("status"),
                "17_decision": decision_engine.get("decision"),
                "18_dashboard": "READY",
                "19_explainability": explainability.get("status"),
                "20_reports": "READY",
                "21_audit": "READY",
                "22_blockchain_ready": "LOCAL_NOT_CONNECTED",
                "23_plus_eval_demo_docs": "READY",
            },
            "face": face_result,
            "cross_validation": cross_val,
            "calibration": calibration,
            "confidence_engine": confidence_engine,
            "decision_engine": decision_engine,
            "explainability": explainability,
            "forensics": forensic_map,
            "text_visual": text_visual,
            "module_errors": module_errors or None,
        },
    }
