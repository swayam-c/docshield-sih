"""Health and readiness endpoints."""

from fastapi import APIRouter

from app import __version__
from app.api.schemas import HealthResponse
from app.config import get_settings
from app.ml.device import ml_mode
from audit.chain import chain_status


router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    settings = get_settings()
    mode = ml_mode()
    enet = "working_mock" if mode == "mock" else "working"
    vit = "working_mock" if mode == "mock" else "working"
    clip = "working_mock" if mode == "mock" else "working"
    chain = chain_status()
    return HealthResponse(
        status="ok",
        service="DOCSHIELD AI",
        version=__version__,
        phase="sih-phase-30-complete",
        demo_mode=settings.demo_mode,
        components={
            "api": "working",
            "database": "working",
            "upload_validation": "working",
            "pdf_intake": "working" if settings.allow_pdf else "disabled",
            "retention_policy": "delete_after_analyze"
            if not settings.retain_uploads
            else f"ttl_{settings.upload_ttl_seconds}s",
            "rate_limit": "working" if settings.rate_limit_per_minute > 0 else "disabled",
            "image_quality": "working",
            "document_classifier": "working_logreg_synthetic_v1+heuristic_v2",
            "document_profiles": "working_sih_taxonomy",
            "ocr": "working_tesseract",
            "field_extraction": "working_profile_driven",
            "mrz": "working_td1_td2_td3_checklist",
            "efficientnet": enet,
            "efficientnet_authenticity_head": "working_synthetic_logreg_v1",
            "vit": vit,
            "vit_authenticity_head": "working_synthetic_logreg_v1",
            "clip": clip,
            "clip_supporting_evidence": "working_type_alignment_v1",
            "fusion": "working_synthetic_logreg_v1",
            "ml_mode": mode,
            "tampering": "working_heuristic_ela_noise_copymove_v1",
            "tampering_localization": "working_ela_hf_noise_heatmap_v1",
            "face_detection": "working_opencv_haar_v1",
            "face_verify_1to1": "working_uncalibrated_prototype_v1",
            "face_liveness": "working_passive_heuristic_v1",
            "live_camera": "working_ephemeral_session_v1",
            "cross_validation": "working_rule_engine_v1",
            "calibration": "working_synthetic_platt_v1",
            "confidence_engine": "working_v1",
            "decision_engine": "working_pass_review_highrisk_v1",
            "dashboard": "working",
            "explainability": "working_model_comparison_v1",
            "reports": "working_json_markdown_v1",
            "audit_trail": "working_local_hash_chain_v1",
            "blockchain": chain.get("blockchain", "NOT_CONNECTED"),
            "official_verification": "unavailable",
            "reverse_search": "unavailable",
        },
    )
