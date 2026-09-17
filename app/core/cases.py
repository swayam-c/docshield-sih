"""Case ID generation and case update helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.models import CaseRecord


def next_case_id(db: Session, year: int | None = None) -> str:
    y = year or datetime.now(timezone.utc).year
    prefix = f"DS-{y}-"
    count = db.scalar(
        select(func.count()).select_from(CaseRecord).where(CaseRecord.case_id.like(f"{prefix}%"))
    )
    seq = int(count or 0) + 1
    return f"{prefix}{seq:06d}"


def create_pending_case(
    db: Session,
    *,
    original_filename: str,
    stored_filename: str,
    mime_type: str,
) -> CaseRecord:
    case = CaseRecord(
        case_id=next_case_id(db),
        document_type="UNKNOWN",
        decision="PENDING",
        original_filename=original_filename,
        stored_filename=stored_filename,
        mime_type=mime_type,
        evidence_json="{}",
        reviewer_status="OPEN",
    )
    db.add(case)
    db.commit()
    db.refresh(case)
    return case


def update_case_analysis(
    db: Session,
    case: CaseRecord,
    *,
    document_type: str,
    decision: str,
    evidence_quality: Optional[float],
    confidence: Optional[float],
    evidence_json: str,
    authenticity_estimate: Optional[float] = None,
) -> CaseRecord:
    case.document_type = document_type
    case.decision = decision
    case.evidence_quality = (
        f"{evidence_quality:.4f}" if evidence_quality is not None else "N/A"
    )
    case.confidence = f"{confidence:.4f}" if confidence is not None else "N/A"
    case.authenticity_estimate = (
        f"{authenticity_estimate:.4f}" if authenticity_estimate is not None else "N/A"
    )
    case.evidence_json = evidence_json
    db.add(case)
    db.commit()
    db.refresh(case)
    return case
