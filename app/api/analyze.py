"""Document upload + analysis APIs (SIH Phase 1 secure intake)."""

from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status
from sqlalchemy.orm import Session

from app.api.schemas import AnalyzeAcceptedResponse, UploadMeta
from app.config import Settings, get_settings
from app.core.cases import create_pending_case, update_case_analysis
from app.core.rate_limit import enforce_rate_limit
from app.core.retention import delete_upload, purge_expired_uploads, store_upload
from app.core.security import require_api_key_if_configured
from app.core.validation import UploadValidationError, validate_upload_file
from app.db import get_db
from app.pipeline import run_pipeline
from app.pipeline.classifier import classify_document_type
from app.pipeline.quality import analyze_image_quality
from document_profiles import get_profile
from ocr.engine import run_ocr
from ocr.fields import extract_fields
from ocr.mrz import analyze_mrz

router = APIRouter(prefix="/api", tags=["analyze"])


async def _secure_intake(
    request: Request,
    file: UploadFile,
    settings: Settings,
):
    enforce_rate_limit(request)
    purge_expired_uploads(settings)
    try:
        return await validate_upload_file(file, settings)
    except UploadValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": exc.message, "code": exc.code},
        ) from exc


@router.post(
    "/document/upload",
    dependencies=[Depends(require_api_key_if_configured)],
)
async def document_upload(
    request: Request,
    file: UploadFile = File(...),
    settings: Settings = Depends(get_settings),
):
    """Secure document intake only — validates and returns metadata (no full screening)."""
    validated = await _secure_intake(request, file, settings)
    retained = False
    stored_name = ""
    if settings.retain_uploads:
        stored_name = f"{uuid.uuid4().hex}{validated.extension}"
        store_upload(settings, stored_name, validated.content)
        retained = True
    return {
        "status": "ACCEPTED",
        "upload": UploadMeta(
            filename=validated.filename,
            mime_type=validated.mime_type,
            size_bytes=validated.size_bytes,
            width=validated.width,
            height=validated.height,
            source_format=validated.source_format,
            orientation_corrected=validated.orientation_corrected,
            retained=retained,
        ).model_dump(),
        "stored_filename": stored_name or None,
        "message": (
            "Document accepted after secure validation. "
            + ("File retained per policy." if retained else "Raw bytes not retained on disk.")
        ),
        "note": "DEMO DATA — NOT AN OFFICIAL GOVERNMENT VERIFICATION"
        if settings.demo_mode
        else None,
    }


@router.post(
    "/analyze",
    response_model=AnalyzeAcceptedResponse,
    dependencies=[Depends(require_api_key_if_configured)],
)
@router.post(
    "/document/analyze",
    response_model=AnalyzeAcceptedResponse,
    dependencies=[Depends(require_api_key_if_configured)],
)
async def analyze_document(
    request: Request,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
) -> AnalyzeAcceptedResponse:
    validated = await _secure_intake(request, file, settings)

    stored_name = f"{uuid.uuid4().hex}{validated.extension}"
    retained = False
    if settings.retain_uploads:
        store_upload(settings, stored_name, validated.content)
        retained = True
    else:
        # Still write briefly for case traceability then delete after analysis
        store_upload(settings, stored_name, validated.content)

    case = create_pending_case(
        db,
        original_filename=validated.filename,
        stored_filename=stored_name if retained else "",
        mime_type=validated.mime_type,
    )

    try:
        result = run_pipeline(validated.content, filename=validated.filename)
    except ValueError as exc:
        if not retained:
            delete_upload(settings, stored_name)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": str(exc), "code": "ANALYSIS_FAILED"},
        ) from exc

    if not retained:
        delete_upload(settings, stored_name)

    assessment = result["assessment"]
    update_case_analysis(
        db,
        case,
        document_type=result["document_type_label"],
        decision=assessment["decision"],
        evidence_quality=assessment.get("evidence_quality"),
        confidence=assessment.get("confidence"),
        authenticity_estimate=assessment.get("authenticity_estimate"),
        evidence_json=result["evidence_json"],
    )

    try:
        from audit.chain import append_audit_event

        append_audit_event(
            db,
            case_id=case.case_id,
            action="ANALYZE",
            actor="system",
            summary=(
                f"decision={assessment.get('decision')} "
                f"type={result['document_type_label']} "
                f"auth={assessment.get('authenticity_estimate')}"
            ),
            details={"decision": assessment.get("decision")},
        )
    except Exception as exc:  # noqa: BLE001 — surface audit failure explicitly
        logger = __import__("logging").getLogger("docshield.analyze")
        logger.exception("Audit append failed")
        result.setdefault("evidence", {})["audit_append"] = {
            "status": "ERROR",
            "error": repr(exc),
            "note": "Case analysis succeeded; audit chain append failed.",
        }

    return AnalyzeAcceptedResponse(
        case_id=case.case_id,
        status="ANALYZED" if assessment["decision"] != "INCONCLUSIVE" else "INCONCLUSIVE",
        message=result["message"],
        upload=UploadMeta(
            filename=validated.filename,
            mime_type=validated.mime_type,
            size_bytes=validated.size_bytes,
            width=validated.width,
            height=validated.height,
            source_format=validated.source_format,
            orientation_corrected=validated.orientation_corrected,
            retained=retained,
        ),
        document_type=result["evidence"]["document_type"],
        image_quality=result["evidence"]["image_quality"],
        assessment=assessment,
        evidence=result["evidence"],
        pipeline=result["pipeline"],
        screening_result=result.get("screening_result"),
    )


@router.post("/quality")
async def quality_only(
    request: Request,
    file: UploadFile = File(...),
    settings: Settings = Depends(get_settings),
):
    validated = await _secure_intake(request, file, settings)
    return analyze_image_quality(validated.content).to_dict()


@router.post("/classify")
async def classify_only(
    request: Request,
    file: UploadFile = File(...),
    settings: Settings = Depends(get_settings),
):
    validated = await _secure_intake(request, file, settings)
    return classify_document_type(validated.content, filename=validated.filename).to_dict()


@router.post("/ocr")
async def ocr_endpoint(
    request: Request,
    file: UploadFile = File(...),
    settings: Settings = Depends(get_settings),
):
    validated = await _secure_intake(request, file, settings)
    doc = classify_document_type(validated.content, filename=validated.filename)
    profile = get_profile(doc.label)
    ocr_raw = run_ocr(validated.content)
    if ocr_raw.get("status") != "OK":
        return {
            "status": ocr_raw.get("status", "NOT_ASSESSED"),
            "ocr": "NOT_ASSESSED",
            "document_type": doc.to_dict(),
            "error": ocr_raw.get("error"),
            "note": ocr_raw.get("note"),
        }

    fields = extract_fields(ocr_raw.get("full_text", ""), ocr_raw.get("text_boxes", []), doc.label)
    mrz = analyze_mrz(
        ocr_raw.get("full_text", ""),
        doc.label,
        profile.supports_mrz,
        fields_payload=fields,
    )
    fields_public = {k: v for k, v in fields.items() if k != "compare_values"}
    return {
        "status": "OK",
        "document_type": doc.to_dict(),
        "ocr": {
            "status": ocr_raw["status"],
            "engine": ocr_raw["engine"],
            "mean_ocr_confidence": ocr_raw["mean_ocr_confidence"],
            "text_box_count": ocr_raw["text_box_count"],
            "text_boxes": ocr_raw["text_boxes"],
            "preprocess": ocr_raw["preprocess"],
            "note": ocr_raw["note"],
        },
        "fields": fields_public,
        "mrz": mrz,
    }


@router.post("/features")
async def features_endpoint(
    request: Request,
    file: UploadFile = File(...),
    settings: Settings = Depends(get_settings),
):
    from app.ml.extract import extract_vision_features

    validated = await _secure_intake(request, file, settings)
    doc = classify_document_type(validated.content, filename=validated.filename)
    result = extract_vision_features(
        validated.content, predicted_document_type=doc.label
    )
    result["document_type"] = doc.to_dict()
    return result


@router.get("/system/status")
def system_status(settings: Settings = Depends(get_settings)):
    from app.ml.device import ml_mode, use_mock_backends
    from app.ml.registry import clip_available
    from ocr.engine import tesseract_available

    mock = use_mock_backends()
    return {
        "service": "DOCSHIELD AI",
        "demo_mode": settings.demo_mode,
        "demo_banner": "DEMO DATA — NOT AN OFFICIAL GOVERNMENT VERIFICATION"
        if settings.demo_mode
        else None,
        "secure_input": {
            "allowed_extensions": settings.allowed_extension_list,
            "allow_pdf": settings.allow_pdf,
            "max_upload_bytes": settings.max_upload_bytes,
            "retain_uploads": settings.retain_uploads,
            "upload_ttl_seconds": settings.upload_ttl_seconds,
            "rate_limit_per_minute": settings.rate_limit_per_minute,
        },
        "modules": {
            "OCR": "READY" if tesseract_available() else "UNAVAILABLE",
            "EfficientNet": "READY" if not mock else "READY (mock)",
            "ViT": "READY" if not mock else "READY (mock)",
            "CLIP": "READY" if (mock or clip_available()) else "UNAVAILABLE",
            "Face Model": "READY (Haar detect + uncalibrated compare)",
            "Liveness": "READY (passive heuristic PAD)",
            "Live camera": "READY (ephemeral session)",
            "Forensics": "READY (heuristic Phase 9)",
            "Tampering localization": "READY (heatmap Phase 10)",
            "Fusion Model": "READY (uncalibrated synthetic)",
            "Calibration": "NOT READY",
            "Database": "READY",
            "Audit": "PARTIAL",
            "ml_mode": ml_mode(),
        },
    }


@router.post("/tamper")
async def tamper_endpoint(
    request: Request,
    file: UploadFile = File(...),
    settings: Settings = Depends(get_settings),
):
    from forensics.tampering import analyze_tampering

    validated = await _secure_intake(request, file, settings)
    return analyze_tampering(validated.content)


@router.post("/face/compare")
async def face_compare(
    request: Request,
    file: UploadFile = File(...),
    reference: UploadFile | None = File(None),
    settings: Settings = Depends(get_settings),
):
    """Detect face on document image; optional reference for uncalibrated 1:1 compare."""
    from verification.face import compare_faces

    validated = await _secure_intake(request, file, settings)
    ref_bytes = None
    if reference is not None and reference.filename:
        ref_validated = await _secure_intake(request, reference, settings)
        ref_bytes = ref_validated.content
    return compare_faces(validated.content, reference_bytes=ref_bytes)


@router.post("/face/detect")
async def face_detect(
    request: Request,
    file: UploadFile = File(...),
    settings: Settings = Depends(get_settings),
):
    from verification.face import analyze_document_face

    validated = await _secure_intake(request, file, settings)
    return analyze_document_face(validated.content)


@router.post("/face/liveness")
async def face_liveness(
    request: Request,
    file: UploadFile = File(...),
    source: str = "live_camera",
    settings: Settings = Depends(get_settings),
):
    """Passive PAD heuristics. Use source=document for NOT_APPLICABLE on ID photos."""
    from verification.liveness import assess_passive_liveness

    validated = await _secure_intake(request, file, settings)
    src = (source or "live_camera").strip().lower()
    if src not in {"live_camera", "document", "unknown"}:
        src = "live_camera"
    return assess_passive_liveness(validated.content, source=src)


@router.post("/camera/session/start")
async def camera_session_start(case_id: str | None = None):
    from verification.camera import start_camera_session

    return start_camera_session(case_id=case_id)


@router.post("/camera/session/end")
async def camera_session_end(session_id: str):
    from verification.camera import end_camera_session

    return end_camera_session(session_id)


@router.post("/camera/session/attach-document")
async def camera_attach_document(
    request: Request,
    session_id: str,
    file: UploadFile = File(...),
    settings: Settings = Depends(get_settings),
):
    from verification.camera import attach_document_reference

    validated = await _secure_intake(request, file, settings)
    return attach_document_reference(session_id, validated.content)


@router.post("/officer/refine-verdict")
async def refine_officer_verdict(payload: dict):
    """Re-fuse officer verdict after identity capture using prior decision_engine + face."""
    from evidence.decision import refine_with_identity

    previous = payload.get("decision_engine") or {}
    face = payload.get("face") or {}
    refined = refine_with_identity(previous, face=face)
    return {
        "status": "REFINED",
        "decision": refined.get("decision"),
        "officer_verdict": refined.get("officer_verdict"),
        "decision_engine": refined,
        "note": "Screening assist only — not legal authenticity.",
    }


@router.post("/camera/capture")
async def camera_capture(
    request: Request,
    file: UploadFile = File(...),
    session_id: str | None = None,
    compare_to_document: bool = True,
    settings: Settings = Depends(get_settings),
):
    """Analyze a live camera frame (PNG/JPEG). Compares to attached document by default."""
    from verification.camera import analyze_camera_frame

    validated = await _secure_intake(request, file, settings)
    return analyze_camera_frame(
        validated.content,
        session_id=session_id,
        compare_to_document=compare_to_document,
    )


@router.get("/camera/status")
def camera_status():
    from verification.camera import camera_module_status

    return camera_module_status()


@router.post("/reverse-search")
async def reverse_stub():
    settings = get_settings()
    if not settings.reverse_search_url:
        return {
            "status": "SEARCH_UNAVAILABLE",
            "reverse_search": "SEARCH_UNAVAILABLE",
            "message": "No reverse-evidence provider configured.",
        }
    return {
        "status": "NOT_IMPLEMENTED",
        "reverse_search": "NOT_ASSESSED",
    }
