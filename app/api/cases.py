"""Case retrieval, reports, and audit endpoints."""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.schemas import CaseSummary
from app.db import get_db
from app.db.models import CaseRecord
from audit.chain import chain_status, list_case_audit, verify_chain
from reports.generator import build_report_payload, render_markdown, save_report

router = APIRouter(prefix="/api", tags=["cases"])


@router.get("/case/{case_id}", response_model=CaseSummary)
def get_case(case_id: str, db: Session = Depends(get_db)) -> CaseSummary:
    case = db.scalar(select(CaseRecord).where(CaseRecord.case_id == case_id))
    if not case:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Case not found")
    return CaseSummary(
        case_id=case.case_id,
        document_type=case.document_type,
        decision=case.decision,
        authenticity_estimate=case.authenticity_estimate,
        confidence=case.confidence,
        evidence_quality=case.evidence_quality,
        reviewer_status=case.reviewer_status,
        created_at=case.created_at.isoformat() if case.created_at else None,
    )


@router.get("/cases", response_model=list[CaseSummary])
def list_cases(db: Session = Depends(get_db), limit: int = 50) -> list[CaseSummary]:
    rows = db.scalars(
        select(CaseRecord).order_by(CaseRecord.id.desc()).limit(min(limit, 200))
    ).all()
    return [
        CaseSummary(
            case_id=c.case_id,
            document_type=c.document_type,
            decision=c.decision,
            authenticity_estimate=c.authenticity_estimate,
            confidence=c.confidence,
            evidence_quality=c.evidence_quality,
            reviewer_status=c.reviewer_status,
            created_at=c.created_at.isoformat() if c.created_at else None,
        )
        for c in rows
    ]


@router.get("/case/{case_id}/detail")
def case_detail(case_id: str, db: Session = Depends(get_db)):
    case = db.scalar(select(CaseRecord).where(CaseRecord.case_id == case_id))
    if not case:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Case not found")
    try:
        evidence = json.loads(case.evidence_json or "{}")
    except json.JSONDecodeError:
        evidence = {}
    return {
        "case_id": case.case_id,
        "document_type": case.document_type,
        "decision": case.decision,
        "authenticity_estimate": case.authenticity_estimate,
        "confidence": case.confidence,
        "evidence_quality": case.evidence_quality,
        "reviewer_status": case.reviewer_status,
        "created_at": case.created_at.isoformat() if case.created_at else None,
        "original_filename": case.original_filename,
        "evidence": evidence,
        "audit": list_case_audit(db, case_id),
        "disclaimer": "Screening assist only — not legal authenticity / not official verification.",
    }


@router.get("/report/{case_id}")
def get_report(case_id: str, db: Session = Depends(get_db), persist: bool = True):
    case = db.scalar(select(CaseRecord).where(CaseRecord.case_id == case_id))
    if not case:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Case not found")
    try:
        evidence = json.loads(case.evidence_json or "{}")
    except json.JSONDecodeError:
        evidence = {}
    assessment = {
        "decision": case.decision,
        "authenticity_estimate": (
            float(case.authenticity_estimate)
            if case.authenticity_estimate not in {"", "N/A", None}
            else None
        ),
        "confidence": (
            float(case.confidence) if case.confidence not in {"", "N/A", None} else None
        ),
        "evidence_quality": (
            float(case.evidence_quality)
            if case.evidence_quality not in {"", "N/A", None}
            else None
        ),
        "reasons": (evidence.get("decision_engine") or {}).get("decision_reasons")
        or [],
        "recommended_action": (evidence.get("decision_engine") or {}).get(
            "recommended_action"
        ),
    }
    report = build_report_payload(
        case_id=case.case_id,
        document_type=case.document_type,
        decision=case.decision,
        assessment=assessment,
        evidence=evidence,
        screening_result={"phase_status": (evidence.get("phase_status") if False else None)},
    )
    # Prefer phase_status from nested screening if present in evidence blob
    paths = {}
    if persist:
        paths = save_report(report)
    return {
        "status": "READY",
        "report": report,
        "markdown": render_markdown(report),
        "saved": paths,
        "blockchain": "NOT_CONNECTED",
    }


@router.get("/audit/chain/verify")
def audit_verify(db: Session = Depends(get_db)):
    return verify_chain(db)


@router.get("/audit/chain/status")
def audit_status():
    return chain_status()


@router.get("/audit/case/{case_id}")
def audit_case(case_id: str, db: Session = Depends(get_db)):
    return {
        "case_id": case_id,
        "events": list_case_audit(db, case_id),
        "blockchain": "NOT_CONNECTED",
    }
