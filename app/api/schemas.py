"""Shared API schemas."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    service: str = "DOCSHIELD AI"
    version: str
    phase: str
    demo_mode: bool
    components: Dict[str, str]


class UploadMeta(BaseModel):
    filename: str
    mime_type: str
    size_bytes: int
    width: int
    height: int
    source_format: str = "IMAGE"
    orientation_corrected: bool = False
    retained: bool = False


class AnalyzeAcceptedResponse(BaseModel):
    """Full screening response (SIH phases 1–30)."""

    case_id: str
    status: str = "ANALYZED"
    message: str
    upload: UploadMeta
    document_type: Dict[str, Any] = Field(default_factory=dict)
    image_quality: Dict[str, Any] = Field(default_factory=dict)
    assessment: Dict[str, Any] = Field(default_factory=dict)
    evidence: Dict[str, Any] = Field(default_factory=dict)
    pipeline: List[str] = Field(default_factory=list)
    screening_result: Optional[Dict[str, Any]] = None


class CaseSummary(BaseModel):
    case_id: str
    document_type: str
    decision: str
    authenticity_estimate: str
    confidence: str
    evidence_quality: str
    reviewer_status: str
    created_at: Optional[str] = None


class ErrorResponse(BaseModel):
    error: str
    code: str
    detail: Optional[str] = None
