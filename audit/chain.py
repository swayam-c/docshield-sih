"""SIH Phase 21–22 — local tamper-evident audit hash chain.

Labeled DEMO / LOCAL / NOT CONNECTED to any public blockchain.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import AuditEvent

GENESIS = "0" * 64


def _canonical(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def compute_entry_hash(*, prev_hash: str, payload: Dict[str, Any]) -> str:
    blob = f"{prev_hash}|{_canonical(payload)}"
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def latest_hash(db: Session) -> str:
    row = db.scalar(select(AuditEvent).order_by(AuditEvent.id.desc()).limit(1))
    return row.entry_hash if row and row.entry_hash else GENESIS


def append_audit_event(
    db: Session,
    *,
    case_id: str,
    action: str,
    summary: str = "",
    actor: str = "system",
    details: Optional[Dict[str, Any]] = None,
) -> AuditEvent:
    prev = latest_hash(db)
    ts = datetime.now(timezone.utc)
    if details:
        summary = f"{summary} | details_keys={sorted(details.keys())}".strip(" |")
    # Hash excludes timestamp so SQLite round-trips cannot break the chain.
    payload = {
        "case_id": case_id,
        "action": action,
        "actor": actor,
        "summary": summary,
    }
    entry_hash = compute_entry_hash(prev_hash=prev, payload=payload)
    event = AuditEvent(
        case_id=case_id,
        timestamp=ts,
        action=action,
        actor=actor,
        summary=summary,
        prev_hash=prev,
        entry_hash=entry_hash,
    )
    db.add(event)
    db.commit()
    db.refresh(event)
    return event


def verify_chain(db: Session, *, limit: int = 5000) -> Dict[str, Any]:
    rows = list(
        db.scalars(select(AuditEvent).order_by(AuditEvent.id.asc()).limit(limit)).all()
    )
    if not rows:
        return {
            "status": "EMPTY",
            "valid": True,
            "checked": 0,
            "network": "LOCAL_HASH_CHAIN",
            "blockchain": "NOT_CONNECTED",
            "note": "No audit events yet. Chain is local-only (not a public blockchain).",
        }

    prev = GENESIS
    for i, row in enumerate(rows):
        if row.prev_hash != prev:
            return {
                "status": "BROKEN",
                "valid": False,
                "broken_at": i,
                "event_id": row.id,
                "expected_prev": prev,
                "found_prev": row.prev_hash,
                "network": "LOCAL_HASH_CHAIN",
                "blockchain": "NOT_CONNECTED",
            }
        payload = {
            "case_id": row.case_id,
            "action": row.action,
            "actor": row.actor,
            "summary": row.summary,
        }
        expected = compute_entry_hash(prev_hash=prev, payload=payload)
        if expected != row.entry_hash:
            return {
                "status": "BROKEN",
                "valid": False,
                "broken_at": i,
                "event_id": row.id,
                "reason": "entry_hash_mismatch",
                "network": "LOCAL_HASH_CHAIN",
                "blockchain": "NOT_CONNECTED",
            }
        prev = row.entry_hash

    return {
        "status": "VALID",
        "valid": True,
        "checked": len(rows),
        "tip_hash": rows[-1].entry_hash,
        "network": "LOCAL_HASH_CHAIN",
        "blockchain": "NOT_CONNECTED",
        "note": (
            "Local SHA-256 hash chain verified. "
            "NOT a public blockchain transaction. DEMO/LOCAL only."
        ),
    }


def list_case_audit(db: Session, case_id: str, *, limit: int = 100) -> List[Dict[str, Any]]:
    rows = db.scalars(
        select(AuditEvent)
        .where(AuditEvent.case_id == case_id)
        .order_by(AuditEvent.id.asc())
        .limit(limit)
    ).all()
    return [
        {
            "id": r.id,
            "case_id": r.case_id,
            "timestamp": r.timestamp.isoformat() if r.timestamp else None,
            "action": r.action,
            "actor": r.actor,
            "summary": r.summary,
            "prev_hash": r.prev_hash,
            "entry_hash": r.entry_hash,
            "blockchain": "NOT_CONNECTED",
        }
        for r in rows
    ]


def chain_status() -> Dict[str, Any]:
    return {
        "status": "LOCAL_READY",
        "network": "LOCAL_HASH_CHAIN",
        "blockchain": "NOT_CONNECTED",
        "anchoring": "DISABLED",
        "note": "Tamper-evident local hash chain only. No fake on-chain transactions.",
    }
